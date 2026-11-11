"""
scripts/check_bc_rope_cuda.py
=============================
GPU check for `a_tie` (A shared by each state pair) and `bc_rope` (RoPE on the
scan's key B and query C), on the Tiny-ImageNet configs
    jit-s2-vmamba-atie.yaml              (control: A tied, no rotation)
    jit-s2-vmamba-bc-rope-2d-axial.yaml  (A tied + B/C RoPE)
vs jit-s2-vmamba-baseline.yaml (d_state 16). Run it once at the start of the
run's first Kaggle session, like the other scripts/check_*_cuda.py. The maths is
proven on CPU by tests/test_bc_rope_cpu.py with a stubbed scan; what is new here
is the real CUDA kernel fed a tied A and rotated B / C.

  1. bc_rope_tables on CUDA (fp32) vs a float64 CPU derivation; apply_bc_rope
     keeps a bf16 input bf16; a table first built under inference_mode still
     works with autograd.
  2. The mixer (D 384, d_state 16; a_tie alone, and a_tie + bc_rope) on CUDA in
     fp32 matches a CPU copy of the same weights run through mamba_ssm's
     pure-PyTorch selective_scan_ref.
  3. On the real kernel: every coordinate shifted by (+3, +7) leaves the
     bc_rope output unchanged up to fp32 rounding (A tied), moves it with A free
     per state, and the rotation off vs on (same weights) differs.
  4. Each config builds through evaluate.build_model and runs forward + backward
     under bf16 autocast at batch 128 with a finite loss and finite gradients.
  5. Training-step speed (fwd+bwd+AdamW, bf16, batch 128) and peak memory vs the
     baseline config -- record them for the complexity table.

Run (on GPU, from the repo root):
    python scripts/check_bc_rope_cuda.py --config bc-rope
    python scripts/check_bc_rope_cuda.py --config atie
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
from src.models.vmamba import SS2D, bc_rope_tables, apply_bc_rope
from scripts.evaluate import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", choices=("atie", "bc-rope", "all"), default="all")
args = ap.parse_args()

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True
THETA, N, Dm, G = 10.0, 16, 384, 8
CFG = "configs/tiny_imagenet/"
BASE = CFG + "jit-s2-vmamba-baseline.yaml"
CONFIGS = {"atie": CFG + "jit-s2-vmamba-atie.yaml",
           "bc-rope": CFG + "jit-s2-vmamba-bc-rope-2d-axial.yaml"}
configs = list(CONFIGS) if args.config == "all" else [args.config]


def ref_angles(H, W, dr=0, dc=0):
    """(1, 4, N/2, H*W) float64 on the CPU, the formula written out again, with
    the cell each path visits taken from SS2D's scan orders."""
    omega = THETA ** (-4.0 * torch.arange(N // 4, dtype=torch.float64) / N)
    row = [(r, c) for r in range(H) for c in range(W)]
    col = [(r, c) for c in range(W) for r in range(H)]
    out = torch.zeros(4, N // 2, H * W, dtype=torch.float64)
    for k, cells in enumerate([row, row[::-1], col, col[::-1]]):
        rc = torch.tensor(cells, dtype=torch.float64)            # (L, 2)
        out[k, :N // 4] = (rc[:, 0:1] + dr).T * omega[:, None]
        out[k, N // 4:] = (rc[:, 1:2] + dc).T * omega[:, None]
    return out[None]


def perturbed(model, scale=0.02):
    """Leave the zero (identity) init, so every branch affects the output."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn_like(p))
    return model


# ── 1. tables and rotation on CUDA ───────────────────────────────────────────
cos, sin = bc_rope_tables(THETA, N, G, G, torch.device(dev), torch.float32)
ang = ref_angles(G, G)
e_tab = max((cos.double().cpu() - ang.cos()).abs().max().item(),
            (sin.double().cpu() - ang.sin()).abs().max().item())
v = torch.randn(4, 4, N, G * G, device=dev)
r16 = apply_bc_rope(v.bfloat16(), cos, sin)
with torch.inference_mode():
    bc_rope_tables(THETA, N, 5, 7, torch.device(dev), torch.float32)   # not cached yet
c2, s2 = bc_rope_tables(THETA, N, 5, 7, torch.device(dev), torch.float32)
vg = torch.randn(1, 4, N, 35, device=dev, requires_grad=True)
apply_bc_rope(vg, c2, s2).square().sum().backward()
good = (cos.device.type == "cuda" and e_tab < 1e-5 and r16.dtype == torch.bfloat16
        and torch.isfinite(vg.grad).all().item())
ok &= good
print(f"1. tables on CUDA vs float64: max|diff| = {e_tab:.1e}; bf16 in -> "
      f"{str(r16.dtype).split('.')[-1]}; autograd after inference_mode  {'OK' if good else 'FAIL'}")


# ── 2. mixer: CUDA fp32 vs CPU selective_scan_ref ────────────────────────────
def run_ref(m, x):
    vmamba_mod.selective_scan_fn = selective_scan_ref
    try:
        return m(x, G, G)
    finally:
        vmamba_mod.selective_scan_fn = selective_scan_fn


xm = torch.randn(4, G * G, Dm)
mixers = {}
for label, kw in [("a_tie", dict(a_tie="pairs")),
                  ("a_tie + bc_rope", dict(a_tie="pairs", bc_rope="2d_axial",
                                           bc_rope_theta=THETA))]:
    torch.manual_seed(1)
    cpu = perturbed(SS2D(Dm, d_state=N, **kw).eval())
    gpu = copy.deepcopy(cpu).to(dev)
    with torch.no_grad():
        y_gpu = gpu(xm.to(dev), G, G).cpu()
        y_cpu = run_ref(cpu, xm)
    e = ((y_gpu - y_cpu).abs().max() / y_cpu.abs().max()).item()
    good = e < 1e-4
    ok &= good
    mixers[label] = (gpu, y_gpu)
    print(f"2. {label}: CUDA fp32 vs CPU selective_scan_ref max rel |dy| = {e:.2e}  "
          f"{'OK' if good else 'FAIL'}")


# ── 3. relative on the real kernel ───────────────────────────────────────────
def shifted(m, dr, dc):
    def tables(theta, n, H, W, device, dtype):
        a = ref_angles(H, W, dr, dc).to(device=device, dtype=dtype)
        return a.cos(), a.sin()
    real = vmamba_mod.bc_rope_tables
    vmamba_mod.bc_rope_tables = tables
    try:
        with torch.no_grad():
            return m(xm.to(dev), G, G).cpu()
    finally:
        vmamba_mod.bc_rope_tables = real


rope, y0 = mixers["a_tie + bc_rope"]
d_tied = ((shifted(rope, 3, 7) - y0).abs().max() / y0.abs().max()).item()
free = copy.deepcopy(rope)
with torch.no_grad():
    free.A_logs = torch.nn.Parameter(
        torch.log(torch.rand(4, Dm, N, device=dev) * 15 + 1))
free.a_tie = "none"
with torch.no_grad():
    yf = free(xm.to(dev), G, G).cpu()
d_free = ((shifted(free, 3, 7) - yf).abs().max() / yf.abs().max()).item()
off = SS2D(Dm, d_state=N, a_tie="pairs").to(dev).eval()
off.load_state_dict(rope.state_dict(), strict=True)
with torch.no_grad():
    d_off = ((off(xm.to(dev), G, G).cpu() - y0).abs().max() / y0.abs().max()).item()
# fp32 rounding only when A is tied (CPU stub: ~2e-7); the free-A shift and the
# rotation itself move it by ~5e-4 / ~1e-3 at this perturbation scale.
good = d_tied < 1e-5 and d_free > 1e-4 and d_off > 1e-4
ok &= good
print(f"3. shift (+3, +7): A tied max rel |dy| = {d_tied:.1e}, A free {d_free:.2e}; "
      f"rotation off vs on {d_off:.2e}  {'OK' if good else 'FAIL'}")
del mixers, rope, free, off
torch.cuda.empty_cache()


# ── 4. the configs under bf16 at the training batch ─────────────────────────
for name in configs:
    with open(CONFIGS[name]) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    assert mc["a_tie"] == "pairs", CONFIGS[name]
    torch.manual_seed(0)
    m = perturbed(build_model(cfg).to(dev).train())
    bs = 128
    xi = torch.randn(bs, 3, mc["input_size"], mc["input_size"], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(xi, torch.rand(bs, device=dev),
                torch.randint(0, mc["num_classes"], (bs,), device=dev))
    loss = (out.float() - xi).square().mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    good = (torch.isfinite(loss).item() and len(grads) > 0
            and all(torch.isfinite(g).all().item() for g in grads))
    ok &= good
    n = sum(p.numel() for p in m.parameters())
    print(f"4. {os.path.basename(CONFIGS[name])}: {n:,} params, bf16 fwd+bwd at batch {bs}, "
          f"loss {loss.item():.4f}  {'OK' if good else 'FAIL'}")
    del m, out, loss, grads
    torch.cuda.empty_cache()


# ── 5. training-step speed and peak memory ───────────────────────────────────
def step_rate(path, bs=128, warmup=5, iters=20):
    with open(path) as f:
        c = yaml.safe_load(f)
    mc = c["model"]
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    m = build_model(c).to(dev).train()
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
for label, path in [("baseline (reference)", BASE)] + [(n, CONFIGS[n]) for n in configs]:
    try:
        r, peak = step_rate(path)
    except torch.cuda.OutOfMemoryError:
        ok = False
        print(f"5. {label:<22} OUT OF MEMORY at batch 128  FAIL")
        torch.cuda.empty_cache()
        continue
    r_ref = r_ref or r
    print(f"5. {label:<22} {r:6.2f} it/s ({r / r_ref:.3f}x reference)  peak {peak:5.2f} GiB")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
