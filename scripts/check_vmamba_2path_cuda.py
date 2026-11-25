"""
scripts/check_vmamba_2path_cuda.py
==================================
GPU check for JiT-S2-VMamba with the 2-path SS2D (scan_paths [0, 2]) and,
optionally, SF-Mamba's auxiliary tokens on it
(configs/tiny_imagenet/jit-s2-vmamba-2path{,-auxswap}.yaml). Run it once at
the start of the run's first Kaggle session, in the same spirit as
scripts/check_spatial_auxswap_cuda.py.

What is new on the GPU: the stock CUDA kernel over K = 2 stacked directions
(2 * d_inner channels instead of 4), and with tokens H*W + 2 positions per
path (head before cell (0, 0), tail after the last cell), the tokens'
centre-tap conv + SiLU and the head <-> tail swap between blocks.

  1. The 2-path SS2D on CUDA (fp32) matches a CPU copy of the same weights run
     through mamba_ssm's pure-PyTorch selective_scan_ref: without tokens, and
     with [head, tail].
  2. The config builds through evaluate.build_model and runs forward + backward
     under bf16 autocast, with a finite output and finite, non-zero grads into
     the last block's per-direction parameters and DWConv; with tokens, swap and
     no-swap differ from the same weights.
  3. Training-step speed (fwd+bwd+AdamW, bf16, batch 128) and peak memory for
     the config and jit-s2-vmamba-baseline (4 paths).
  4. src/flops_counter.count_complexity (needs fvcore) on the baseline, the
     2-path config and the 2-path + swap config: printed exactly, and each must
     be within 2 % of the expected ~1.423 / 1.290 / 1.317 GMACs.

Run (on GPU, from the repo root):
    python scripts/check_vmamba_2path_cuda.py --config auxswap     # or 2path
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import src.models.vmamba as vmamba_mod
from src.models.vmamba import SS2D
from scripts.evaluate import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", choices=("2path", "auxswap"), default="auxswap")
args = ap.parse_args()

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True
CFG_DIR = "configs/tiny_imagenet/"
CONFIGS = {"baseline": CFG_DIR + "jit-s2-vmamba-baseline.yaml",
           "2path": CFG_DIR + "jit-s2-vmamba-2path.yaml",
           "auxswap": CFG_DIR + "jit-s2-vmamba-2path-auxswap.yaml"}
CONFIG = CONFIGS[args.config]
EXPECTED_GMACS = {"baseline": 1.423, "2path": 1.290, "auxswap": 1.317}


# ── 1. the 2-path mixer: CUDA fp32 vs CPU reference ─────────────────────────
def compare(name, module, x):
    torch.manual_seed(0)
    cpu = module.eval()
    with torch.no_grad():
        for p in cpu.parameters():
            p.add_(0.02 * torch.randn_like(p))
    gpu = copy.deepcopy(cpu).to(dev)
    with torch.no_grad():
        y_gpu = gpu(x.to(dev), 8, 8).cpu()
        vmamba_mod.selective_scan_fn = selective_scan_ref   # CPU reference path
        try:
            y_cpu = cpu(x, 8, 8)
        finally:
            vmamba_mod.selective_scan_fn = selective_scan_fn
    e = ((y_gpu - y_cpu).abs().max() / y_cpu.abs().max()).item()
    g = e < 1e-4
    print(f"1. {name:<40} CUDA fp32 vs CPU ref: max rel |dy| = {e:.2e}  {'OK' if g else 'FAIL'}")
    return g


torch.manual_seed(1)
ok &= compare("2-path SS2D, no tokens",
              SS2D(384, d_state=16, K=2, scan_paths=(0, 2)), torch.randn(4, 64, 384))
torch.manual_seed(1)
ok &= compare("2-path SS2D, [head, tail] tokens",
              SS2D(384, d_state=16, K=2, scan_paths=(0, 2), aux_tokens=True),
              torch.randn(4, 66, 384))

# ── 2. the config end to end under bf16 ──────────────────────────────────────
cfg = yaml.safe_load(open(CONFIG))
mc = cfg["model"]
torch.manual_seed(0)
m = build_model(cfg).to(dev).train()
with torch.no_grad():                        # leave the zero (identity) init so
    for p in m.parameters():                 # every branch receives gradient
        p.add_(0.02 * torch.randn_like(p))
xi = torch.randn(2, 3, mc["input_size"], mc["input_size"], device=dev)
ti = torch.rand(2, device=dev)
yi = torch.randint(0, mc["num_classes"], (2,), device=dev)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(xi, ti, yi)
out.float().square().mean().backward()
mx = m.blocks[-1].mixer
grads = [mx.x_proj_weight.grad, mx.dt_projs_weight.grad, mx.A_logs.grad, mx.Ds.grad,
         mx.conv2d.weight.grad]
good = (out.shape == xi.shape and torch.isfinite(out).all().item()
        and tuple(mx.x_proj_weight.shape[:1]) == (2,)
        and all(g is not None and torch.isfinite(g).all().item() and g.abs().max() > 0
                for g in grads))
msg = ""
if m.aux_tokens != "none":
    m2 = copy.deepcopy(m).eval()
    m2.aux_tokens = "noswap" if m.aux_tokens == "swap" else "swap"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        differ = (m.eval()(xi, ti, yi) - m2(xi, ti, yi)).abs().max().item() > 0
    good &= differ
    msg = f", aux_tokens {m.aux_tokens} from block {m.aux_tokens_start}, differs from {m2.aux_tokens}"
    del m2
ok &= good
n = sum(p.numel() for p in m.parameters())
print(f"2. {os.path.basename(CONFIG)}: {n:,} params, scan_paths {list(m.scan_paths)}{msg}, "
      f"bf16 fwd+bwd  {'OK' if good else 'FAIL'}")
del m, out
torch.cuda.empty_cache()


# ── 3. training-step speed and peak memory ───────────────────────────────────
def step_rate(cfg, bs=128, warmup=5, iters=20):
    mc = cfg["model"]
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    m = build_model(cfg).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    xi = torch.randn(bs, 3, mc["input_size"], mc["input_size"], device=dev)
    t = torch.rand(bs, device=dev)
    y = torch.randint(0, mc["num_classes"], (bs,), device=dev)
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(xi, t, y).float().square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    rate = iters / (time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del m, opt
    torch.cuda.empty_cache()
    return rate, peak


r_ref = None
for name, path in (("4-path baseline (reference)", CONFIGS["baseline"]),
                   (os.path.basename(CONFIG), CONFIG)):
    try:
        r, peak = step_rate(yaml.safe_load(open(path)))
    except torch.cuda.OutOfMemoryError:
        ok = False
        print(f"3. {name:<36} OUT OF MEMORY at batch 128  FAIL")
        torch.cuda.empty_cache()
        continue
    r_ref = r_ref or r
    print(f"3. {name:<36} {r:6.2f} it/s ({r / r_ref:.3f}x reference)  peak {peak:5.2f} GiB")

# ── 4. MACs ──────────────────────────────────────────────────────────────────
try:
    from src.flops_counter import count_complexity
except ImportError as e:
    ok = False
    print(f"4. MACs  FAIL (fvcore missing: pip install fvcore) ({e})")
else:
    for name in ("baseline", "2path", "auxswap"):
        c = yaml.safe_load(open(CONFIGS[name]))
        torch.manual_seed(0)
        r = count_complexity(build_model(c), img_size=c["model"]["input_size"],
                             in_channels=c["model"]["in_channels"],
                             num_classes=c["model"]["num_classes"], device=dev)
        g_macs = r["macs_total"] / 1e9
        exp = EXPECTED_GMACS[name]
        g = abs(g_macs - exp) <= 0.02 * exp
        ok &= g
        print(f"4. {name:<9} {r['macs_total']:>15,} MACs = {g_macs:.4f} G "
              f"(expected ~{exp:.3f} G, {100 * (g_macs / exp - 1):+.2f} %)  {'OK' if g else 'FAIL'}")
        torch.cuda.empty_cache()

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
