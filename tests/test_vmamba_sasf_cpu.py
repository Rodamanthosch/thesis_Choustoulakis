"""
tests/test_vmamba_sasf_cpu.py
=============================
CPU-only verification of JiT-S2-VMamba + SASF (src/models/vmamba_sasf.py):
vmamba.py's SS2D at d_state 1 with Spatial-Mamba's state fusion inside each of
the four scan directions.

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, kept
in float64 when fed float64 so the fidelity checks are exact).

Checks:
  A. cross_unscan: undoes cross_scan for every direction, and summing it
     reproduces vmamba.py's cross_merge.
  B. DirectionalStateFusion == 4 independent spatial_mamba.StateFusion modules
     (one per direction, own kernels and alpha), float64.
  C. reduction: with identity fusion (centre tap 1, alpha = (1, 0, 0)) the
     mixer equals vmamba.py's SS2D at d_state 1 with the same weights.
  D. fidelity: the mixer equals a naive transcription -- per-direction scan in
     that direction's order, states and C put back on the grid through an
     independent index map, replicate-clamped neighbour sums with that
     direction's kernels, y = sum_k C^k h^k + D^k u, LayerNorm, out_proj.
  E. model: JiTVMambaSASF has exactly JiTVMamba(d_state=1)'s weights for every
     shared parameter (same seed), and with identity fusion the same output
     (float32 rounding only; the mixer-level float64 proof is check C).
  F. param counts: SS2D + SASF 422,412 per block at D=384 (380,928 + 41,484);
     Tiny-ImageNet model 30,699,984.
  G. smoke + init: forward + backward, grads reach every direction's kernels
     and alpha; every block is the identity at init; CPU bf16 autocast runs.
  H. guards: d_state != 1 and a token count != H*W fail.
  I. wiring: the config builds IDENTICAL models through both build_models
     (needs pyyaml; skipped otherwise).

Run from the repo root:
    PYTHONPATH=. python tests/test_vmamba_sasf_cpu.py
"""
import ast
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:                                    # pragma: no cover
    print(f"SKIP (torch not importable here): {e}")
    sys.exit(0)


# ── stub mamba_ssm with a faithful reference selective scan ──────────────────
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, g, n, l) grouped.
    Computes in float64 when u is float64, float32 otherwise."""
    ct = torch.float64 if u.dtype == torch.float64 else torch.float32
    b, d, l = u.shape
    delta = delta.to(ct)
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(ct)
    if delta_softplus:
        delta = F.softplus(delta)
    u = u.to(ct)
    A = A.to(ct)
    g = B.shape[1]
    rep = d // g
    Bf = B.to(ct).repeat_interleave(rep, dim=1)
    Cf = C.to(ct).repeat_interleave(rep, dim=1)
    n = A.shape[1]
    h = u.new_zeros(b, d, n)
    ys = []
    for i in range(l):
        dAi = torch.exp(delta[:, :, i].unsqueeze(-1) * A[None])
        dBu = delta[:, :, i].unsqueeze(-1) * Bf[:, :, :, i] * u[:, :, i].unsqueeze(-1)
        h = dAi * h + dBu
        ys.append((h * Cf[:, :, :, i]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D[None, :, None].to(ct) * u
    return y


if "mamba_ssm" not in sys.modules:
    stub = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = selective_scan_ref
    iface.selective_scan_ref = selective_scan_ref
    sys.modules["mamba_ssm"] = stub
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = iface

from src.models.vmamba import SS2D, JiTVMamba, cross_scan, cross_merge   # noqa: E402
from src.models.spatial_mamba import StateFusion                          # noqa: E402
from src.models.vmamba_sasf import (                                      # noqa: E402
    JiTVMambaSASF, SS2DSASF, DirectionalStateFusion, cross_unscan,
)

KW = dict(input_size=16, patch_size=4, hidden_size=32, depth=2, num_classes=7,
          bottleneck_dim=16)
H = W = KW["input_size"] // KW["patch_size"]       # 4
HW = H * W
Bsz = 2
torch.manual_seed(0)
x_img = torch.randn(Bsz, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(Bsz)
y_in = torch.randint(0, KW["num_classes"], (Bsz,))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def perturb(module, scale=0.05):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))
    return module


def set_identity_fusion(mx):
    sf = mx.state_fusion
    with torch.no_grad():
        for k in sf.kernels:
            k.zero_()
        sf.kernels[0][:, 0, 1, 1] = 1.0
        sf.alpha.zero_()
        sf.alpha[:, 0] = 1.0


# ═══ A. cross_unscan ═════════════════════════════════════════════════════════
for (hh, ww) in ((4, 4), (3, 5)):
    g = torch.randn(2, 6, hh, ww, dtype=torch.float64)
    un = cross_unscan(cross_scan(g), hh, ww)
    flat = g.reshape(2, 6, hh * ww)
    assert all(torch.equal(un[:, k], flat) for k in range(4)), (hh, ww)
    ys = torch.randn(2, 4, 6, hh * ww, dtype=torch.float64)
    assert (cross_unscan(ys, hh, ww).sum(1) - cross_merge(ys, hh, ww)).abs().max() < 1e-12
print("A  cross_unscan undoes cross_scan per direction; its sum == cross_merge "
      "(4x4 and 3x5 grids)")


# ═══ B. DirectionalStateFusion == 4 StateFusion modules ══════════════════════
worst_b = 0.0
for dil in ((1, 2, 3), (1, 3, 5)):
    torch.manual_seed(1)
    dsf = perturb(DirectionalStateFusion(4, 6, dil), 0.3).double()
    h = torch.randn(2, 4 * 6, H, W, dtype=torch.float64)
    with torch.no_grad():
        got = dsf(h)
        for k in range(4):
            sf = StateFusion(6, dil).double()
            for i in range(len(dil)):
                sf.kernels[i].copy_(dsf.kernels[i][k * 6:(k + 1) * 6])
            sf.alpha.copy_(dsf.alpha[k])
            ref = sf(h[:, k * 6:(k + 1) * 6])
            worst_b = max(worst_b, (got[:, k * 6:(k + 1) * 6] - ref).abs().max().item())
assert worst_b < 1e-12, worst_b
print(f"B  DirectionalStateFusion == 4 independent StateFusion modules "
      f"(max|diff| = {worst_b:.1e})")


# ═══ C. identity fusion == vmamba.py SS2D (d_state 1) ════════════════════════
worst_c = 0.0
for D_ in (32, 24):
    torch.manual_seed(2)
    ss = perturb(SS2D(D_, d_state=1, K=4), 0.1).double()
    mx = SS2DSASF(D_).double()
    mx.load_state_dict(ss.state_dict(), strict=False)
    set_identity_fusion(mx)
    x = torch.randn(2, HW, D_, dtype=torch.float64)
    with torch.no_grad():
        ref = ss(x, H, W)
        worst_c = max(worst_c, ((mx(x, H, W) - ref).abs().max() / ref.abs().max()).item())
assert worst_c < 1e-12, worst_c
print(f"C  identity fusion == vmamba.py SS2D at d_state 1 (max relative diff = {worst_c:.1e})")


# ═══ D. naive transcription ══════════════════════════════════════════════════
def naive_mixer(mx, x, Hh, Ww):
    Bb, Ll, _ = x.shape
    K, di, R = mx.K, mx.d_inner, mx.dt_rank
    u = mx.act(mx.conv2d((x @ mx.in_proj.weight.T).reshape(Bb, Hh, Ww, di).permute(0, 3, 1, 2)))
    u = u.reshape(Bb, di, Ll)
    # independent index map: idx[k, s] = grid position read at scan step s
    idx = cross_scan(torch.arange(Ll, dtype=torch.float64).view(1, 1, Hh, Ww))[0, :, 0].long()
    sf = mx.state_fusion
    y = torch.zeros(Bb, di, Ll, dtype=x.dtype)
    for k in range(K):
        us = u[:, :, idx[k]]                                         # scan order
        dbl = mx.x_proj_weight[k] @ us                               # (B, R+2, L)
        dt_r, Bt, Ct = dbl[:, :R], dbl[:, R], dbl[:, R + 1]
        delta = F.softplus(mx.dt_projs_weight[k] @ dt_r
                           + mx.dt_projs_bias[k].float().to(x.dtype)[None, :, None])
        Av = -torch.exp(mx.A_logs[k].float()).to(x.dtype)[:, 0]
        xprev, st = torch.zeros(Bb, di, dtype=x.dtype), torch.zeros(Bb, di, Ll, dtype=x.dtype)
        for s in range(Ll):
            xprev = torch.exp(delta[:, :, s] * Av) * xprev + delta[:, :, s] * Bt[:, s, None] * us[:, :, s]
            st[:, :, idx[k, s]] = xprev                              # back to the grid
        Cg = torch.zeros(Bb, Ll, dtype=x.dtype)
        Cg[:, idx[k]] = Ct
        st = st.reshape(Bb, di, Hh, Ww)
        h = torch.zeros_like(st)
        for r in range(Hh):
            for c in range(Ww):
                acc = 0
                for i, (kk, dil) in enumerate(zip(sf.kernels, sf.dilations)):
                    kd = kk[k * di:(k + 1) * di, 0]
                    for a in (-1, 0, 1):
                        for b in (-1, 0, 1):
                            rr = min(max(r + a * dil, 0), Hh - 1)
                            cc = min(max(c + b * dil, 0), Ww - 1)
                            acc = acc + sf.alpha[k, i] * kd[:, a + 1, b + 1] * st[:, :, rr, cc]
                h[:, :, r, c] = acc
        y = y + h.reshape(Bb, di, Ll) * Cg[:, None, :] + mx.Ds[k][None, :, None] * u
    y = F.layer_norm(y.transpose(1, 2), (di,), mx.out_norm.weight, mx.out_norm.bias, mx.out_norm.eps)
    return y @ mx.out_proj.weight.T


worst_d = 0.0
for dil in ((1, 2, 3), (1, 3, 5)):
    torch.manual_seed(3)
    mx = perturb(SS2DSASF(16, dilations=dil), 0.1).double()
    x = torch.randn(2, HW, 16, dtype=torch.float64)
    with torch.no_grad():
        ref = naive_mixer(mx, x, H, W)
        worst_d = max(worst_d, ((mx(x, H, W) - ref).abs().max() / ref.abs().max()).item())
assert worst_d < 1e-12, worst_d
print(f"D  mixer == naive per-direction transcription, dilations (1,2,3) and (1,3,5) "
      f"(max relative diff = {worst_d:.1e})")


# ═══ E. model == JiTVMamba(d_state=1) ═════════════════════════════════════════
for ffn in ("swiglu", "convglu"):
    torch.manual_seed(4)
    vm = JiTVMamba(d_state=1, ffn=ffn, **KW)
    torch.manual_seed(4)
    sm = JiTVMambaSASF(ffn=ffn, sasf_dilations=(1, 2, 3), **KW)
    sv, ssd = vm.state_dict(), sm.state_dict()
    extra = set(ssd) - set(sv)
    assert set(sv) <= set(ssd) and all(".state_fusion." in k for k in extra), extra
    assert all(torch.equal(sv[k], ssd[k]) for k in sv), ffn
    perturb(vm, 0.02)
    sm.load_state_dict(vm.state_dict(), strict=False)
    for blk in sm.blocks:
        set_identity_fusion(blk.mixer)
    vm, sm = vm.eval(), sm.eval()           # float32: JiT's t-embedder is fp32-only
    with torch.no_grad():
        ref = vm(x_img, t_in, y_in)
        de = ((sm(x_img, t_in, y_in) - ref).abs().max() / ref.abs().max()).item()
    assert de < 1e-5, (ffn, de)
print(f"E  model: JiTVMamba(d_state=1)'s weights for every shared parameter; identity "
      f"fusion -> same output (swiglu, convglu; max relative diff = {de:.1e})")


# ═══ F. param counts ═════════════════════════════════════════════════════════
p_mx = n_params(SS2DSASF(384, dilations=(1, 2, 3)))
assert p_mx == 380_928 + 41_484, p_mx
p_tiny = n_params(JiTVMambaSASF(input_size=64, patch_size=8, hidden_size=384, depth=12,
                                num_classes=200, bottleneck_dim=128, sasf_dilations=(1, 2, 3)))
assert p_tiny == 30_699_984, p_tiny
print(f"F  SS2D + SASF per block at D=384: {p_mx:,} (380,928 + 41,484); "
      f"Tiny-IN model {p_tiny / 1e6:.2f}M ({p_tiny:,})")


# ═══ G. smoke, grads, identity at init, bf16 ═════════════════════════════════
torch.manual_seed(5)
m = JiTVMambaSASF(sasf_dilations=(1, 2, 3), **KW)
xb, cb = torch.randn(Bsz, HW, KW["hidden_size"]), torch.randn(Bsz, KW["hidden_size"])
with torch.no_grad():
    for blk in m.blocks:
        assert torch.equal(blk(xb, cb, H, W), xb)
perturb(m, 0.02).train()
out = m(x_img, t_in, y_in)
assert out.shape == x_img.shape and torch.isfinite(out).all()
out.square().mean().backward()
sf = m.blocks[-1].mixer.state_fusion
assert (sf.alpha.grad.abs() > 0).all(), "some direction's alpha gets no gradient"
for k in sf.kernels:
    gk = k.grad.view(4, -1)
    assert torch.isfinite(gk).all() and (gk.abs().amax(dim=1) > 0).all()
with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
    assert torch.isfinite(m(x_img, t_in, y_in).float()).all()
print("G  fwd+bwd OK; grads reach every direction's kernels and alpha; blocks identity "
      "at init; bf16 OK")


# ═══ H. guards ═══════════════════════════════════════════════════════════════
for bad, what in ((lambda: JiTVMambaSASF(d_state=16, **KW), "d_state != 1"),
                  (lambda: SS2DSASF(32)(torch.randn(1, HW + 2, 32), H, W), "prefix")):
    try:
        bad()
        raise SystemExit(f"guard missing: {what}")
    except AssertionError:
        pass
print("H  d_state != 1 and a prefix (L != H*W) are rejected")


# ═══ I. both build_model copies agree on the config ══════════════════════════
try:
    import yaml                                             # noqa: E402
except ImportError as e:                                    # pragma: no cover
    print(f"I  SKIPPED (pyyaml not importable here: {e})")
    print("ALL CHECKS PASSED")
    sys.exit(0)


def load_build_model(rel):
    path = os.path.join(REPO_ROOT, rel)
    src = open(path, encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_model")
    ns = {}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns["build_model"]


cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "tiny_imagenet",
                                       "jit-s2-vmamba-sasf.yaml"), encoding="utf-8"))
torch.manual_seed(0)
mt = load_build_model("scripts/run_experiment.py")(cfg)
torch.manual_seed(0)
me = load_build_model("scripts/evaluate.py")(cfg)
st, se = mt.state_dict(), me.state_dict()
assert type(mt) is JiTVMambaSASF and set(st) == set(se)
assert all(torch.equal(st[k], se[k]) for k in st)
assert mt.sasf_dilations == (1, 2, 3) and n_params(mt) == 30_699_984, n_params(mt)
print(f"I  jit-s2-vmamba-sasf.yaml: train == eval build_model, {n_params(mt) / 1e6:.2f}M params")

print("ALL CHECKS PASSED")
