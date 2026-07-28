"""
tests/test_stateinit_equivalence.py
===================================
Proof that the virtual write step used by SS2D's state_init is ALGEBRAICALLY
EXACT h_{-1} initialization under Mamba's discretization:

    h_{-1}_eff[d, n] = softplus(dt0[d] + delta_bias[d]) * B0[n] * u0[d]

Tests (all float64, tol 1e-10 -> equivalence must hold at machine epsilon):
  A. Single direction, variable B/C.
  B. Stacked K=4 grouped format — the exact call signature of SS2D's
     selective_scan_fn launch (A: (K*D, N), B/C: (B, K, N, L)).
  C. Representability: DiMSUM's paper-literal per-channel init
     h_{-1}[d, n] = w[d] (Linear(c) broadcast over states) is an exact
     special case (B0 = 1_N, u0 = w / softplus(dt0 + bias)).

The recurrence below mirrors mamba_ssm's selective_scan_ref:
    deltaA   = exp(delta * A);  deltaB_u = delta * B * u
    x_i      = deltaA_i * x_{i-1} + deltaB_u_i;   y_i = C_i . x_i + D * u_i
If mamba_ssm is installed, Test A additionally cross-checks against the
installed selective_scan_ref in float32.

Run:  python tests/test_stateinit_equivalence.py
"""
import sys
import torch
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


def ref_scan(u, delta, A, B, C, D, delta_bias, h_init=None):
    """selective_scan_ref recurrence with optional nonzero initial state.
    u, delta: (B,D,L); A: (D,N); B,C: (B,N,L) or grouped (B,G,N,L); D: (D,)."""
    delta = F.softplus(delta + delta_bias[None, :, None])
    batch, dim, L = u.shape
    N = A.shape[1]
    if B.dim() == 4:
        H = dim // B.shape[1]
        B = B.repeat_interleave(H, dim=1)
        C = C.repeat_interleave(H, dim=1)
        deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
        read = lambda x, i: torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
    else:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        read = lambda x, i: torch.einsum("bdn,bn->bd", x, C[:, :, i])
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    x = torch.zeros(batch, dim, N) if h_init is None else h_init.clone()
    ys = []
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(read(x, i))
    return torch.stack(ys, dim=2) + u * D[None, :, None], x


def main():
    errs = {}

    # ---------------------------------------------------------------- A
    Bt, D, N, L = 3, 8, 4, 16
    u    = torch.randn(Bt, D, L)
    dlt  = torch.randn(Bt, D, L) * 0.5
    A    = -torch.rand(D, N) * 2 - 0.1
    Bv   = torch.randn(Bt, N, L)
    Cv   = torch.randn(Bt, N, L)
    Dp   = torch.randn(D)
    bias = torch.randn(D) * 0.2
    u0   = torch.randn(Bt, D)
    B0   = torch.randn(Bt, N)
    dt0  = torch.randn(Bt, D)

    u_e   = torch.cat([u0[:, :, None],  u],   2)
    dlt_e = torch.cat([dt0[:, :, None], dlt], 2)
    B_e   = torch.cat([B0[:, :, None],  Bv],  2)
    C_e   = torch.cat([torch.zeros(Bt, N, 1), Cv], 2)
    y_v, h_v = ref_scan(u_e, dlt_e, A, B_e, C_e, Dp, bias)
    y_v = y_v[:, :, 1:]

    h_init = F.softplus(dt0 + bias[None, :])[:, :, None] * B0[:, None, :] * u0[:, :, None]
    y_i, h_i = ref_scan(u, dlt, A, Bv, Cv, Dp, bias, h_init=h_init)
    errs["A single-dir y"]  = (y_v - y_i).abs().max().item()
    errs["A single-dir h"]  = (h_v - h_i).abs().max().item()

    # ---------------------------------------------------------------- B
    K = 4
    uK, dltK = torch.randn(Bt, K * D, L), torch.randn(Bt, K * D, L) * 0.5
    AK  = -torch.rand(K * D, N) * 2 - 0.1
    BK, CK = torch.randn(Bt, K, N, L), torch.randn(Bt, K, N, L)
    DK, bK = torch.randn(K * D), torch.randn(K * D) * 0.2
    u0K  = torch.randn(Bt, D).repeat(1, K)
    B0K  = torch.randn(K, N).expand(Bt, K, N).clone()
    dt0K = torch.randn(K, D).reshape(K * D).expand(Bt, K * D).clone()

    u_e   = torch.cat([u0K[:, :, None],  uK],   2)
    dlt_e = torch.cat([dt0K[:, :, None], dltK], 2)
    B_e   = torch.cat([B0K[:, :, :, None], BK], 3)
    C_e   = torch.cat([torch.zeros(Bt, K, N, 1), CK], 3)
    yv, hv = ref_scan(u_e, dlt_e, AK, B_e, C_e, DK, bK)
    yv = yv[:, :, 1:]

    B0_full = B0K.repeat_interleave(D, dim=1)
    h_initK = F.softplus(dt0K + bK[None, :])[:, :, None] * B0_full * u0K[:, :, None]
    yi, hi = ref_scan(uK, dltK, AK, BK, CK, DK, bK, h_init=h_initK)
    errs["B stacked-K4 y"] = (yv - yi).abs().max().item()
    errs["B stacked-K4 h"] = (hv - hi).abs().max().item()

    # ---------------------------------------------------------------- C
    w    = torch.randn(Bt, D)
    dt0c = torch.zeros(Bt, D)
    u0c  = w / F.softplus(dt0c + bias[None, :])
    B0c  = torch.ones(Bt, N)
    u_e   = torch.cat([u0c[:, :, None], u], 2)
    dlt_e = torch.cat([dt0c[:, :, None], dlt], 2)
    B_e   = torch.cat([B0c[:, :, None], Bv], 2)
    C_e   = torch.cat([torch.zeros(Bt, N, 1), Cv], 2)
    y_v, _ = ref_scan(u_e, dlt_e, A, B_e, C_e, Dp, bias)
    y_d, _ = ref_scan(u, dlt, A, Bv, Cv, Dp, bias,
                      h_init=w[:, :, None].expand(Bt, D, N).clone())
    errs["C dimsum-literal y"] = (y_v[:, :, 1:] - y_d).abs().max().item()

    # ---------------------------------------------------- optional CUDA-repo ref
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as ssr
        y32, _ = ref_scan(u.float(), dlt.float(), A.float(), Bv.float(),
                          Cv.float(), Dp.float(), bias.float(), h_init.float())
        yr = ssr(torch.cat([u0[:, :, None], u], 2).float(),
                 torch.cat([dt0[:, :, None], dlt], 2).float(),
                 A.float(),
                 torch.cat([B0[:, :, None], Bv], 2).float(),
                 torch.cat([torch.zeros(Bt, N, 1), Cv], 2).float(),
                 Dp.float(), None, bias.float(), True)
        errs["X vs mamba_ssm ref (fp32)"] = (yr[:, :, 1:] - y32).abs().max().item()
        fp32_keys = {"X vs mamba_ssm ref (fp32)"}
    except ImportError:
        fp32_keys = set()

    ok = True
    for k, e in errs.items():
        tol = 1e-4 if k in fp32_keys else 1e-10
        flag = "ok " if e < tol else "FAIL"
        ok &= e < tol
        print(f"  [{flag}] {k:28s} max|err| = {e:.3e}")
    print("ALL TESTS PASS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
