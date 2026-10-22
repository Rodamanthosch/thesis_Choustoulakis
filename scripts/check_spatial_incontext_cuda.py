"""
scripts/check_spatial_incontext_cuda.py
=======================================
GPU check for JiT-S2-Spatial-Mamba with in-context class tokens placed inside
the scan (configs/tiny_imagenet/jit-s2-spatial-nogate-incontext-class3.yaml).
Run it once at the start of the run's first Kaggle session, in the same spirit
as scripts/check_spatial_cuda.py.

What is new on the GPU is the gather that interleaves the tokens into the scan
(and back), feeding the stock CUDA kernel a sequence of H*W + P positions.

  1. The StructureAwareSSM with tokens at grid cells (3, 27, 59) on CUDA (fp32)
     matches a CPU copy of the same weights run through mamba_ssm's
     pure-PyTorch selective_scan_ref -- gate on and off, and with the tokens
     all at the head (the plain prefix).
  2. The config builds through evaluate.build_model and runs forward + backward
     under bf16 autocast, with a finite output and finite grads into the
     token slots, SASF and the LPU.
  3. Training-step speed (fwd+bwd+AdamW, bf16, batch 128) and peak memory for
     the config and the no-gate run it extends.

Run (on GPU, from the repo root):  python scripts/check_spatial_incontext_cuda.py
"""
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

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True
CONFIG = "configs/tiny_imagenet/jit-s2-spatial-nogate-incontext-class3.yaml"
BASE = "configs/tiny_imagenet/jit-s2-spatial-nogate.yaml"


# ── 1. mixer with tokens: CUDA fp32 vs CPU reference ─────────────────────────
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


for gate, pos in ((False, (3, 27, 59)), (True, (3, 27, 59)), (False, (0, 0, 0))):
    torch.manual_seed(1)
    ok &= compare(f"gate={gate}, tokens before cells {pos}",
                  StructureAwareSSM(384, gate=gate, dilations=(1, 2, 3),
                                    in_context_positions=pos),
                  torch.randn(4, 64 + len(pos), 384))

# ── 2. the config end to end under bf16 ──────────────────────────────────────
cfg = yaml.safe_load(open(CONFIG))
mc = cfg["model"]
torch.manual_seed(0)
m = build_model(cfg).to(dev).train()
with torch.no_grad():                        # leave the zero (identity) init so
    for p in m.parameters():                 # every branch receives gradient
        p.add_(0.02 * torch.randn_like(p))
xi = torch.randn(2, 3, mc["input_size"], mc["input_size"], device=dev)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(xi, torch.rand(2, device=dev), torch.randint(0, mc["num_classes"], (2,), device=dev))
out.float().square().mean().backward()
blk = m.blocks[-1]
grads = [m.incontext_pos_embed.grad, blk.mixer.state_fusion.alpha.grad,
         blk.mixer.state_fusion.kernels[-1].grad, blk.cpe1.weight.grad]
good = (out.shape == xi.shape and torch.isfinite(out).all().item()
        and all(g is not None and torch.isfinite(g).all().item() for g in grads))
ok &= good
n = sum(p.numel() for p in m.parameters())
print(f"2. {os.path.basename(CONFIG)}: {n:,} params, tokens at "
      f"{m.in_context_positions} from block {m.in_context_start}, bf16 fwd+bwd  "
      f"{'OK' if good else 'FAIL'}")
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
for name, path in (("spatial no gate (reference)", BASE), ("+ 3 in-context class tokens", CONFIG)):
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
