"""
scripts/check_stateinit_cuda.py
===============================
One-batch check that the CUDA selective_scan_fn agrees with selective_scan_ref
on the EXTENDED sequence (virtual write position prepended, C0 = 0), in the
exact stacked-K format SS2D uses. Run once at the start of the first Kaggle
session of a state-init arm — same spirit as the cross_merge(cross_scan(x))
== 4x verification.

Expected: max|dy| at fp32-epsilon-accumulation level (~1e-4 for L=65).

Run (on GPU):  python scripts/check_stateinit_cuda.py
"""
import sys
import torch

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

torch.manual_seed(42)
assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"

B, K, D, N, L = 4, 4, 96, 16, 64          # d_inner=96 keeps it light; L=8x8 grid
u    = torch.randn(B, K * D, L + 1, device=dev)          # +1 = virtual position
dlt  = torch.randn(B, K * D, L + 1, device=dev) * 0.5
A    = -torch.rand(K * D, N, device=dev) * 2 - 0.1
Bs   = torch.randn(B, K, N, L + 1, device=dev)
Cs   = torch.randn(B, K, N, L + 1, device=dev)
Cs[:, :, :, 0] = 0                                        # C0 = 0 (pure state write)
Dp   = torch.randn(K * D, device=dev)
bias = torch.randn(K * D, device=dev) * 0.2

y_cuda = selective_scan_fn(u, dlt, A, Bs, Cs, Dp, z=None,
                           delta_bias=bias, delta_softplus=True)
y_ref  = selective_scan_ref(u, dlt, A, Bs, Cs, Dp, z=None,
                            delta_bias=bias, delta_softplus=True)

err = (y_cuda - y_ref).abs().max().item()
print(f"CUDA vs reference on extended sequence: max|dy| = {err:.3e}")
ok = err < 1e-3
print("PASS" if ok else "FAIL — do not trust the state-init arm on this kernel build")
sys.exit(0 if ok else 1)
