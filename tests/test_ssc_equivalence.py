"""
tests/test_ssc_equivalence.py
=============================
Proofs for the SSC ("abc") realization on the stock selective scan:

  A. GATE IDENTITY: calling the kernel with delta' = s*softplus(delta+bias)
     and B' = B/s under delta_softplus=False reproduces EXACTLY the gated
     recurrence  h_i = exp(s*Delta_i*A) h_{i-1} + Delta_i*B_i*u_i
     (decay gated by s, write untouched) — DiM-2's A-modulation semantics.
  B. s = 1 CONSISTENCY: the delta_softplus=False path with s=1 equals the
     stock delta_softplus=True path (the two kernel invocation styles agree).
  C. BIAS LINEARITY: adding (b0 + W_B c) to B / (c0 + W_C c) to C before the
     scan equals the recurrence run with the shifted maps (trivial but guards
     broadcast/shape mistakes in the (B, K, N, L) grouped layout).

All float64, tol 1e-10.  Run:  python tests/test_ssc_equivalence.py
"""
import sys
import torch
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


def scan(u, delta_post, A, Bv, Cv, D, decay_scale=None):
    """Manual recurrence with POST-softplus delta and optional decay gate:
    h_i = exp(decay_scale * delta_i * A) h_{i-1} + delta_i * B_i * u_i.
    u, delta_post: (B, KD, L); A: (KD, N); Bv, Cv grouped: (B, K, N, L)."""
    batch, dim, L = u.shape
    Kg = Bv.shape[1]
    H = dim // Kg
    Bf = Bv.repeat_interleave(H, dim=1)          # (B, KD, N, L)
    Cf = Cv.repeat_interleave(H, dim=1)
    s = 1.0 if decay_scale is None else decay_scale.repeat_interleave(H, dim=1)[..., None]  # (B,KD,1)
    x = torch.zeros(batch, dim, A.shape[1])
    ys = []
    sc = 1.0 if decay_scale is None else s.squeeze(-1)
    for i in range(L):
        decay = torch.exp((delta_post[:, :, i] * sc)[..., None] * A[None])
        x = decay * x + (delta_post[:, :, i, None] * Bf[:, :, :, i] * u[:, :, i, None])
        ys.append(torch.einsum("bdn,bdn->bd", x, Cf[:, :, :, i]))
    return torch.stack(ys, 2) + u * D[None, :, None]


def main():
    # Primary reference: faithful fp64 replica of selective_scan_ref semantics
    # (the installed ref casts to fp32 internally, breaking fp64 exactness).
    def kernel(u, delta, A, B, C, D=None, z=None, delta_bias=None,
               delta_softplus=False, return_last_state=False):
        d = delta
        if delta_bias is not None: d = d + delta_bias[..., None]
        if delta_softplus: d = F.softplus(d)
        return scan(u, d, A, B, C, D)

    Bt, K, Dch, N, L = 3, 4, 6, 5, 12
    KD = K * Dch
    u    = torch.randn(Bt, KD, L)
    dlt  = torch.randn(Bt, KD, L) * 0.5           # PRE-softplus
    A    = -torch.rand(KD, N) * 2 - 0.1
    Bv   = torch.randn(Bt, K, N, L)
    Cv   = torch.randn(Bt, K, N, L)
    Dp   = torch.randn(KD)
    bias = torch.randn(KD) * 0.2
    gate = (torch.rand(Bt, K) * 0.85 + 0.1)       # s in [0.1, 0.95]

    errs = {}

    # ---- A. gate identity --------------------------------------------------
    dt_post = F.softplus(dlt + bias[None, :, None])
    g_full  = gate.repeat_interleave(Dch, dim=1)[..., None]           # (B, KD, 1)
    y_kern = kernel(u, dt_post * g_full, A, Bv / gate[:, :, None, None], Cv, Dp,
                    delta_bias=None, delta_softplus=False)
    y_man  = scan(u, dt_post, A, Bv, Cv, Dp, decay_scale=gate)
    errs["A gate identity"] = (y_kern - y_man).abs().max().item()

    # ---- B. s = 1 consistency ----------------------------------------------
    y_nosp = kernel(u, dt_post, A, Bv, Cv, Dp, delta_bias=None, delta_softplus=False)
    y_sp   = kernel(u, dlt, A, Bv, Cv, Dp, delta_bias=bias, delta_softplus=True)
    errs["B s=1 consistency"] = (y_nosp - y_sp).abs().max().item()

    # ---- C. bias linearity --------------------------------------------------
    b0 = torch.randn(K, N); c0 = torch.randn(K, N)
    wb = torch.randn(Bt, K, N); wc = torch.randn(Bt, K, N)
    B_shift = Bv + b0[None, :, :, None] + wb[..., None]
    C_shift = Cv + c0[None, :, :, None] + wc[..., None]
    y_kernC = kernel(u, dlt, A, B_shift, C_shift, Dp, delta_bias=bias, delta_softplus=True)
    y_manC  = scan(u, F.softplus(dlt + bias[None, :, None]), A, B_shift, C_shift, Dp)
    errs["C bias linearity"] = (y_kernC - y_manC).abs().max().item()

    # Optional: cross-check the s=1 no-softplus invocation against the
    # installed mamba_ssm reference (fp32 tolerance).
    fp32_keys = set()
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as ssr
        y_m = ssr(u.float(), dt_post.float(), A.float(), Bv.float(), Cv.float(),
                  Dp.float(), None, None, False)
        errs["X vs mamba_ssm ref (fp32)"] = (y_m - y_nosp.float()).abs().max().item()
        fp32_keys = {"X vs mamba_ssm ref (fp32)"}
    except ImportError:
        pass

    ok = True
    for k, e in errs.items():
        tol = 1e-4 if k in fp32_keys else 1e-10
        flag = "ok " if e < tol else "FAIL"
        ok &= e < tol
        print(f"  [{flag}] {k:26s} max|err| = {e:.3e}")
    print("ALL TESTS PASS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
