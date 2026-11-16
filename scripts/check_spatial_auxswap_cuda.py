"""
scripts/check_spatial_auxswap_cuda.py
=====================================
GPU check for JiT-S2-Spatial-Mamba with SF-Mamba's auxiliary tokens
(configs/tiny_imagenet/jit-s2-spatial-nogate-aux{swap,noswap}.yaml). Run it
once at the start of the run's first Kaggle session, in the same spirit as
scripts/check_spatial_incontext_cuda.py.

What is new on the GPU: a head token scanned before cell 0 and a tail token
after the last cell (the stock CUDA kernel sees H*W + 2 positions), the
tokens' centre-tap conv + SiLU, and the head <-> tail swap between blocks.

  1. The StructureAwareSSM with the two tokens on CUDA (fp32) matches a CPU
     copy of the same weights run through mamba_ssm's pure-PyTorch
     selective_scan_ref.
  2. The config builds through evaluate.build_model and runs forward + backward
     under bf16 autocast, with a finite output and finite grads into the
     mixer's DWConv (which the tokens' centre tap reaches), SASF and the LPU;
     the swap and no-swap models differ from the same weights.
  3. Training-step speed (fwd+bwd+AdamW, bf16, batch 128) and peak memory for
     the config and the no-gate run it extends.

Run (on GPU, from the repo root):
    python scripts/check_spatial_auxswap_cuda.py --mode swap      # or noswap
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

import src.models.spatial_mamba as spatial_mod
from src.models.spatial_mamba import StructureAwareSSM
from scripts.evaluate import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("swap", "noswap"), default="swap")
args = ap.parse_args()

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True
CONFIG = f"configs/tiny_imagenet/jit-s2-spatial-nogate-aux{args.mode}.yaml"
BASE = "configs/tiny_imagenet/jit-s2-spatial-nogate.yaml"


# ── 1. mixer with head + tail tokens: CUDA fp32 vs CPU reference ─────────────
def compare(name, module, x):
    torch.manual_seed(0)
    cpu = module.eval()
    with torch.no_grad():
        for p in cpu.parameters():
            p.add_(0.02 * torch.randn_like(p))
    gpu = copy.deepcopy(cpu).to(dev)
    with torch.no_grad():
        y_gpu = gpu(x.to(dev), 8, 8).cpu()
        spatial_mod.selective_scan_fn = selective_scan_ref  # CPU reference path
        try:
            y_cpu = cpu(x, 8, 8)
        finally:
            spatial_mod.selective_scan_fn = selective_scan_fn
    e = ((y_gpu - y_cpu).abs().max() / y_cpu.abs().max()).item()
    g = e < 1e-4
    print(f"1. {name:<44} CUDA fp32 vs CPU ref: max rel |dy| = {e:.2e}  {'OK' if g else 'FAIL'}")
    return g


torch.manual_seed(1)
ok &= compare("no gate, head + tail tokens (0, 64)",
              StructureAwareSSM(384, gate=False, dilations=(1, 2, 3),
                                in_context_positions=(0, 64), token_conv=True),
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
blk = m.blocks[-1]
grads = [blk.mixer.conv2d.weight.grad, blk.mixer.state_fusion.alpha.grad,
         blk.mixer.state_fusion.kernels[-1].grad, blk.cpe1.weight.grad]
good = (out.shape == xi.shape and torch.isfinite(out).all().item()
        and all(g is not None and torch.isfinite(g).all().item() for g in grads))
other = "noswap" if args.mode == "swap" else "swap"
m2 = copy.deepcopy(m).eval()
m2.aux_tokens = other
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    differ = (m.eval()(xi, ti, yi) - m2(xi, ti, yi)).abs().max().item() > 0
good &= differ
ok &= good
n = sum(p.numel() for p in m.parameters())
print(f"2. {os.path.basename(CONFIG)}: {n:,} params, aux_tokens {m.aux_tokens} from "
      f"block {m.aux_tokens_start}, bf16 fwd+bwd, differs from {other}  "
      f"{'OK' if good else 'FAIL'}")
del m, m2, out
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
for name, path in (("spatial no gate (reference)", BASE), (f"+ aux tokens, {args.mode}", CONFIG)):
    try:
        r, peak = step_rate(yaml.safe_load(open(path)))
    except torch.cuda.OutOfMemoryError:
        ok = False
        print(f"3. {name:<30} OUT OF MEMORY at batch 128  FAIL")
        torch.cuda.empty_cache()
        continue
    r_ref = r_ref or r
    print(f"3. {name:<30} {r:6.2f} it/s ({r / r_ref:.3f}x reference)  peak {peak:5.2f} GiB")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
