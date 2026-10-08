"""
scripts/check_spatial_cuda.py
=============================
GPU check for JiT-S2-Spatial-Mamba (model: spatial_mamba). Run it once at the
start of the first Kaggle session of a Spatial run, in the same spirit as
scripts/check_ffn_cuda.py.

What is new on the GPU is reading the scan STATES out of the stock mamba_ssm
CUDA kernel (C = 1, D = None), the replicate-padded dilated depthwise convs of
SASF, and the LPU convs, all under bf16 autocast.

  1. Stock CUDA selective_scan_fn with C = 1, D = None returns the states x_t
     of an explicit float64 recursion; and the full StructureAwareSSM on CUDA
     (fp32) matches a CPU copy of the same weights run through mamba_ssm's
     pure-PyTorch selective_scan_ref -- gate on and off.
  2. The two Spatial configs and the VMamba d_state=1 config build through
     evaluate.build_model and run forward + backward under bf16 autocast, with
     a finite output and finite grads into SASF (kernels, alpha) and the LPU.
  3. Training-step throughput (fwd+bwd+AdamW, bf16, batch 128) for the three
     configs and the d_state=16 baseline, for the complexity table.

Run (on GPU, from the repo root):  python scripts/check_spatial_cuda.py
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
from src.models.spatial_mamba import StructureAwareSSM
from scripts.evaluate import build_model

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True

# ── 1a. states out of the stock CUDA kernel ──────────────────────────────────
torch.manual_seed(42)
Bb, d, L = 4, 384, 64
u = torch.randn(Bb, d, L, dtype=torch.float64)
dt = 0.5 * torch.randn(Bb, d, L, dtype=torch.float64)
A = -(0.2 + torch.rand(d, 1, dtype=torch.float64))
Bm = torch.randn(Bb, 1, 1, L, dtype=torch.float64)
bias = torch.randn(d, dtype=torch.float64) - 3.0
with torch.no_grad():
    got = selective_scan_fn(
        u.float().to(dev), dt.float().to(dev), A.float().to(dev),
        Bm.float().to(dev), torch.ones_like(Bm).float().to(dev), None,
        z=None, delta_bias=bias.float().to(dev), delta_softplus=True,
    ).double().cpu()
delta = F.softplus(dt + bias[None, :, None])
xs, x_prev = [], torch.zeros(Bb, d, dtype=torch.float64)
for t in range(L):
    x_prev = torch.exp(delta[:, :, t] * A[:, 0]) * x_prev \
        + delta[:, :, t] * Bm[:, 0, 0, t, None] * u[:, :, t]
    xs.append(x_prev)
want = torch.stack(xs, dim=-1)
err = ((got - want).abs().max() / want.abs().max()).item()
good = err < 1e-4
ok &= good
print(f"1. stock CUDA scan, C=1 D=None == explicit fp64 states: "
      f"max rel |dx| = {err:.2e}  {'OK' if good else 'FAIL'}")

# ── 1b. full mixer: CUDA vs CPU (selective_scan_ref) ─────────────────────────
for gate in (True, False):
    torch.manual_seed(0)
    cpu = StructureAwareSSM(384, gate=gate, dilations=(1, 2, 3)).eval()
    with torch.no_grad():
        for p in cpu.parameters():
            p.add_(0.02 * torch.randn_like(p))
    gpu = copy.deepcopy(cpu).to(dev)
    x = torch.randn(4, 64, 384)
    with torch.no_grad():
        y_gpu = gpu(x.to(dev), 8, 8).cpu()
        spatial_mod.selective_scan_fn = selective_scan_ref    # CPU reference path
        try:
            y_cpu = cpu(x, 8, 8)
        finally:
            spatial_mod.selective_scan_fn = selective_scan_fn
    err = ((y_gpu - y_cpu).abs().max() / y_cpu.abs().max()).item()
    good = err < 1e-4
    ok &= good
    print(f"1. StructureAwareSSM gate={str(gate):<5} CUDA fp32 vs CPU ref: "
          f"max rel |dy| = {err:.2e}  {'OK' if good else 'FAIL'}")

# ── 2. end-to-end on the three configs ───────────────────────────────────────
CONFIGS = [
    "configs/tiny_imagenet/jit-s2-spatial-gate.yaml",
    "configs/tiny_imagenet/jit-s2-spatial-nogate.yaml",
    "configs/tiny_imagenet/jit-s2-vmamba-dstate1.yaml",
]
for path in CONFIGS:
    cfg = yaml.safe_load(open(path))
    mc = cfg["model"]
    res, n_cls = mc["input_size"], mc["num_classes"]
    torch.manual_seed(0)
    m = build_model(cfg).to(dev).train()
    with torch.no_grad():                    # leave the zero (identity) init so
        for p in m.parameters():             # every branch receives gradient
            p.add_(0.02 * torch.randn_like(p))
    x = torch.randn(2, 3, res, res, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x, torch.rand(2, device=dev),
                torch.randint(0, n_cls, (2,), device=dev))
    out.float().square().mean().backward()
    good = out.shape == x.shape and torch.isfinite(out).all().item()
    if cfg["experiment"]["model"] == "spatial_mamba":
        blk = m.blocks[-1]
        grads = [blk.mixer.state_fusion.alpha.grad,
                 blk.mixer.state_fusion.kernels[-1].grad]
        if mc.get("lpu", "none") != "none":
            grads += [blk.cpe1.weight.grad, blk.cpe2.weight.grad]
        good &= all(g is not None and torch.isfinite(g).all().item() for g in grads)
    ok &= good
    n = sum(p.numel() for p in m.parameters())
    print(f"2. {os.path.basename(path):<30} out {tuple(out.shape)}  "
          f"{n / 1e6:.2f}M params  {'OK' if good else 'FAIL'}")
    del m, out
    torch.cuda.empty_cache()


# ── 3. training-step throughput ──────────────────────────────────────────────
def step_rate(cfg, bs=128, warmup=5, iters=20):
    mc = cfg["model"]
    torch.manual_seed(0)
    m = build_model(cfg).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randn(bs, 3, mc["input_size"], mc["input_size"], device=dev)
    t = torch.rand(bs, device=dev)
    y = torch.randint(0, mc["num_classes"], (bs,), device=dev)
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, t, y).float().square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    rate = iters / (time.perf_counter() - t0)
    del m, opt
    torch.cuda.empty_cache()
    return rate


r_ref = step_rate(yaml.safe_load(open("configs/tiny_imagenet/jit-s2-vmamba-baseline.yaml")))
print(f"3. vmamba d_state=16 (baseline) {r_ref:6.2f} it/s (bs 128, bf16, fwd+bwd+AdamW)")
for path in CONFIGS:
    r = step_rate(yaml.safe_load(open(path)))
    print(f"3. {os.path.basename(path)[:-5]:<29}{r:6.2f} it/s  ({r / r_ref:.3f}x baseline)")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
