"""
scripts/check_sasf_cuda.py
==========================
GPU check for the SASF experiments:
  - Spatial-Mamba, no gate, d_state 16 (sasf_state_kernels: shared | per_state)
    and d_state 4 (shared | per_state, no checkpointing)
  - JiT-S2-VMamba + SASF (model: vmamba_sasf, d_state 1)
Run it once at the start of each run's first Kaggle session, in the same
spirit as scripts/check_spatial_cuda.py.

  1. Stock CUDA selective_scan_fn over N * d single-state channels (B-group n
     carrying B_t[n], C = 1, D = None) returns the N-state recurrence of an
     explicit float64 loop (the d_state 16 read-out).
  2. Each mixer on CUDA (fp32) matches a CPU copy of the same weights run
     through mamba_ssm's pure-PyTorch selective_scan_ref: Spatial N = 16 with
     shared and per-state kernels, and SS2D + SASF.
  3. The three configs build through evaluate.build_model and run forward +
     backward under bf16 autocast with finite output and finite grads into
     SASF (and the LPU for Spatial).
  4. Training-step speed (fwd+bwd+AdamW, bf16, batch 128) and peak memory for
     the three configs and their reference runs, for the complexity table and
     to confirm batch 128 fits.

Run (on GPU, from the repo root):  python scripts/check_sasf_cuda.py
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import src.models.spatial_mamba as spatial_mod
import src.models.vmamba_sasf as sasf_mod
from src.models.spatial_mamba import StructureAwareSSM
from src.models.vmamba_sasf import SS2DSASF
from scripts.evaluate import build_model

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True

# ── 1. N-state read-out through the stock CUDA kernel ────────────────────────
torch.manual_seed(42)
Bb, d, N, L = 2, 384, 16, 64
u = torch.randn(Bb, d, L, dtype=torch.float64)
dt = 0.5 * torch.randn(Bb, d, L, dtype=torch.float64)
A_cn = -torch.exp(torch.log(torch.arange(1, N + 1, dtype=torch.float64)).repeat(d, 1)
                  + 0.1 * torch.randn(d, N, dtype=torch.float64))
Bt = torch.randn(Bb, N, L, dtype=torch.float64)
bias = torch.randn(d, dtype=torch.float64) - 3.0
with torch.no_grad():
    got = selective_scan_fn(
        u.repeat(1, N, 1).float().to(dev), dt.repeat(1, N, 1).float().to(dev),
        A_cn.t().reshape(N * d, 1).float().to(dev),
        Bt.unsqueeze(2).float().to(dev), torch.ones(Bb, N, 1, L, device=dev), None,
        z=None, delta_bias=bias.repeat(N).float().to(dev), delta_softplus=True,
    ).double().cpu().view(Bb, N, d, L)
delta = F.softplus(dt + bias[None, :, None])
xs, xprev = [], torch.zeros(Bb, d, N, dtype=torch.float64)
for t in range(L):
    xprev = torch.exp(delta[:, :, t, None] * A_cn[None]) * xprev \
        + delta[:, :, t, None] * Bt[:, None, :, t] * u[:, :, t, None]
    xs.append(xprev)
want = torch.stack(xs, dim=-1).permute(0, 2, 1, 3)
err = ((got - want).abs().max() / want.abs().max()).item()
good = err < 1e-4
ok &= good
print(f"1. stock CUDA scan, {N} x {d} single-state channels == explicit fp64 {N}-state "
      f"recurrence: max rel |dx| = {err:.2e}  {'OK' if good else 'FAIL'}")


# ── 2. mixers: CUDA fp32 vs CPU reference ────────────────────────────────────
def compare(name, module, mod, x):
    torch.manual_seed(0)
    cpu = module.eval()
    with torch.no_grad():
        for p in cpu.parameters():
            p.add_(0.02 * torch.randn_like(p))
    gpu = copy.deepcopy(cpu).to(dev)
    with torch.no_grad():
        y_gpu = gpu(x.to(dev), 8, 8).cpu()
        mod.selective_scan_fn = selective_scan_ref         # CPU reference path
        try:
            y_cpu = cpu(x, 8, 8)
        finally:
            mod.selective_scan_fn = selective_scan_fn
    e = ((y_gpu - y_cpu).abs().max() / y_cpu.abs().max()).item()
    g = e < 1e-4
    print(f"2. {name:<42} CUDA fp32 vs CPU ref: max rel |dy| = {e:.2e}  {'OK' if g else 'FAIL'}")
    return g


x = torch.randn(4, 64, 384)
for sk in ("shared", "per_state"):
    ok &= compare(f"Spatial no gate, d_state 16, {sk}",
                  StructureAwareSSM(384, d_state=16, gate=False, dilations=(1, 2, 3),
                                    state_kernels=sk), spatial_mod, x)
ok &= compare("VMamba SS2D + SASF (d_state 1)",
              SS2DSASF(384, dilations=(1, 2, 3)), sasf_mod, x)

# ── 3. end-to-end on the three configs ───────────────────────────────────────
CONFIGS = [
    "configs/tiny_imagenet/jit-s2-spatial-nogate-dstate16-shared.yaml",
    "configs/tiny_imagenet/jit-s2-spatial-nogate-dstate16-perstate.yaml",
    "configs/tiny_imagenet/jit-s2-spatial-nogate-dstate4-shared.yaml",
    "configs/tiny_imagenet/jit-s2-spatial-nogate-dstate4-perstate.yaml",
    "configs/tiny_imagenet/jit-s2-vmamba-sasf.yaml",
]
for path in CONFIGS:
    cfg = yaml.safe_load(open(path))
    mc = cfg["model"]
    torch.manual_seed(0)
    m = build_model(cfg).to(dev).train()
    with torch.no_grad():                    # leave the zero (identity) init so
        for p in m.parameters():             # every branch receives gradient
            p.add_(0.02 * torch.randn_like(p))
    xi = torch.randn(2, 3, mc["input_size"], mc["input_size"], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(xi, torch.rand(2, device=dev),
                torch.randint(0, mc["num_classes"], (2,), device=dev))
    out.float().square().mean().backward()
    blk = m.blocks[-1]
    sf = blk.mixer.state_fusion
    grads = [sf.alpha.grad, sf.kernels[-1].grad]
    if mc.get("lpu", "none") != "none":
        grads += [blk.cpe1.weight.grad]
    good = (out.shape == xi.shape and torch.isfinite(out).all().item()
            and all(g is not None and torch.isfinite(g).all().item() for g in grads))
    ok &= good
    n = sum(p.numel() for p in m.parameters())
    print(f"3. {os.path.basename(path):<45} {n / 1e6:.2f}M params  {'OK' if good else 'FAIL'}")
    del m, out
    torch.cuda.empty_cache()


# ── 4. training-step speed and peak memory ───────────────────────────────────
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


REFS = [
    ("vmamba d_state 16 (baseline)", "configs/tiny_imagenet/jit-s2-vmamba-baseline.yaml"),
    ("vmamba d_state 1", "configs/tiny_imagenet/jit-s2-vmamba-dstate1.yaml"),
    ("spatial no gate, d_state 1", "configs/tiny_imagenet/jit-s2-spatial-nogate.yaml"),
]
r_ref = None
for name, path in REFS + [(os.path.basename(p)[:-5], p) for p in CONFIGS]:
    try:
        r, peak = step_rate(yaml.safe_load(open(path)))
    except torch.cuda.OutOfMemoryError:
        ok = False
        print(f"4. {name:<45} OUT OF MEMORY at batch 128  FAIL")
        torch.cuda.empty_cache()
        continue
    r_ref = r_ref or r
    print(f"4. {name:<45} {r:6.2f} it/s ({r / r_ref:.3f}x baseline)  peak {peak:5.2f} GiB")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
