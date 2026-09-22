"""
scripts/check_incontext_row_cuda.py
===================================
GPU check for in_context_layout="row" — run once at the start of the first
Kaggle session of a row arm, same spirit as scripts/check_stateinit_cuda.py.

The row layout hands the kernel an ORDINARY longer sequence (no C0=0 trick, no
Delta/B rescaling), so there is nothing new to prove about the kernel's algebra
— only that it is still accurate at the longer length, and that the model wires
up end to end on the real kernel under AMP.

  1. CUDA selective_scan_fn vs selective_scan_ref at L = 80 (the time_class row
     length), in the exact stacked-K format SS2D uses.
  2. Both row configs: build via evaluate.build_model, forward + backward under
     bf16 autocast, assert the output shape and the scan length the kernel
     actually sees (72 for class row, 80 for time_class row).

Run (on GPU, from the repo root):  python scripts/check_incontext_row_cuda.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import src.models.vmamba as vm
from scripts.evaluate import build_model

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True

# ── 1. kernel accuracy at the new length ─────────────────────────────────────
torch.manual_seed(42)
B, K, D, N, L = 4, 4, 96, 16, 80
u = torch.randn(B, K * D, L, device=dev)
dlt = torch.randn(B, K * D, L, device=dev) * 0.5
A = -torch.rand(K * D, N, device=dev) * 2 - 0.1
Bs = torch.randn(B, K, N, L, device=dev)
Cs = torch.randn(B, K, N, L, device=dev)
Dp = torch.randn(K * D, device=dev)
bias = torch.randn(K * D, device=dev) * 0.2

y_cuda = selective_scan_fn(u, dlt, A, Bs, Cs, Dp, z=None,
                           delta_bias=bias, delta_softplus=True)
y_ref = selective_scan_ref(u, dlt, A, Bs, Cs, Dp, z=None,
                           delta_bias=bias, delta_softplus=True)
err = (y_cuda - y_ref).abs().max().item()
print(f"1. CUDA vs reference at L={L}: max|dy| = {err:.3e}")
if err >= 1e-3:
    ok = False
    print("   FAIL -- do not trust the row arms on this kernel build")

# ── 2. end-to-end on both row configs ────────────────────────────────────────
SEEN = []
_real = vm.selective_scan_fn


def _recording(u, *a, **kw):
    SEEN.append(u.shape[-1])
    return _real(u, *a, **kw)


CONFIGS = [
    ("configs/tiny_imagenet/jit-s2-vmamba-incontext-row-class.yaml", 64 + 8),
    ("configs/tiny_imagenet/jit-s2-vmamba-incontext-row-timeclass.yaml", 64 + 16),
]

vm.selective_scan_fn = _recording
try:
    for path, want_L in CONFIGS:
        cfg = yaml.safe_load(open(path))
        m = build_model(cfg).to(dev)
        n_cls, res = cfg["model"]["num_classes"], cfg["model"]["input_size"]
        x = torch.randn(2, 3, res, res, device=dev)
        SEEN.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, torch.rand(2, device=dev),
                    torch.randint(0, n_cls, (2,), device=dev))
        out.float().mean().backward()
        got_L = sorted(set(SEEN))
        good = (out.shape == x.shape and got_L == [want_L]
                and m.incontext_pos_embed.grad is not None)
        ok &= good
        print(f"2. {os.path.basename(path):<48} out {tuple(out.shape)} "
              f"scan L {got_L} (want [{want_L}])  {'OK' if good else 'FAIL'}")
        del m, out
        torch.cuda.empty_cache()
finally:
    vm.selective_scan_fn = _real

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
