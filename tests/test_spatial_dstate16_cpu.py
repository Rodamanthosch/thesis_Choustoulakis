"""
tests/test_spatial_dstate16_cpu.py
==================================
CPU-only verification of Spatial-Mamba with d_state > 1
(src/models/spatial_mamba.py, `d_state` + `sasf_state_kernels`).

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, kept
in float64 when fed float64 so the fidelity checks are exact).

Checks:
  A. d_state = 1 is BIT-IDENTICAL to HEAD: the model exec'd out of
     `git show HEAD:src/models/spatial_mamba.py` and the current one, built
     from the same seed, have identical state_dicts and outputs (gate on/off,
     lpu none/zero), so the finished d_state = 1 runs are untouched.
  B. state read-out at N > 1: the scan over N * d single-state channels
     (u, dt repeated; A[c, n]; B-group n) returns the N-state Mamba-1
     recurrence x_t[c, n] of an independent explicit loop (float64).
  C. mixer fidelity at N = 16: equals a naive per-token transcription --
     N-state recurrence, SASF on every (channel, state) map with
     replicate-clamped neighbours, y = sum_n C[n] h[c, n] + D u, LayerNorm,
     optional SiLU(z), out_proj -- for shared and per-state kernels, gate
     on/off (float64).
  D. "shared" == "per_state" when every state's kernels are a copy of the
     channel's (the two options differ only in kernel tying).
  E. param counts at D = 384: no-gate mixer 347,523 (shared) / 503,043
     (per-state); gated 494,979 / 650,499; Tiny-ImageNet no-gate + LPU models
     29.89M / 31.76M.
  F. smoke + init: forward + backward at N = 16 for both options, grads reach
     every state's A, the SASF kernels and alpha, and the LPU; every block is
     the identity at init (lpu zero); CPU bf16 autocast runs.
  G. guards: unknown state_kernels fails.
  I. grad_checkpoint (recompute each mixer in backward; on in the d_state-16
     configs, which do not fit a 15 GB T4 at batch 128 otherwise) changes
     nothing: same parameters, BIT-IDENTICAL output and gradients.
  H. wiring: the d_state-16 and d_state-4 configs build IDENTICAL models through
     scripts/run_experiment.py::build_model and scripts/evaluate.py::build_model
     (needs pyyaml; skipped otherwise).

Run from the repo root:
    PYTHONPATH=. python tests/test_spatial_dstate16_cpu.py
"""
import ast
import os
import subprocess
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
    Bf = B.to(ct).repeat_interleave(rep, dim=1)       # (b, d, n, l)
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

from src.models.spatial_mamba import (                      # noqa: E402  (after the stub)
    JiTSpatialMamba, StructureAwareSSM,
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


# ═══ A. d_state = 1 bit-identical to HEAD ═════════════════════════════════════
try:
    head_src = subprocess.run(
        ["git", "show", "HEAD:src/models/spatial_mamba.py"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8")
except Exception as e:                                      # pragma: no cover
    head_src = None
    print(f"A  SKIPPED (cannot read HEAD:src/models/spatial_mamba.py: {e})")

if head_src is not None:
    head_mod = types.ModuleType("spatial_mamba_head")
    exec(compile(head_src, "<HEAD:src/models/spatial_mamba.py>", "exec"), head_mod.__dict__)
    n_cfg = 0
    for gate in (True, False):
        for lpu in ("none", "zero"):
            kw = dict(gate=gate, lpu=lpu, sasf_dilations=(1, 2, 3), **KW)
            torch.manual_seed(1)
            old = head_mod.JiTSpatialMamba(**kw).eval()
            torch.manual_seed(1)
            new = JiTSpatialMamba(**kw).eval()
            so, sn = old.state_dict(), new.state_dict()
            assert set(so) == set(sn), (gate, lpu, set(so) ^ set(sn))
            assert all(torch.equal(so[k], sn[k]) for k in so), (gate, lpu)
            perturb(old, 0.02)
            new.load_state_dict(old.state_dict(), strict=True)
            with torch.no_grad():
                d = (old(x_img, t_in, y_in) - new(x_img, t_in, y_in)).abs().max().item()
            assert d == 0.0, (gate, lpu, d)
            n_cfg += 1
    print(f"A  d_state=1 bit-identical to HEAD (init AND output) for {n_cfg} configs")


# ═══ B. N * d single-state channels == the N-state recurrence ════════════════
torch.manual_seed(2)
d, N, L = 5, 16, 9
u = torch.randn(2, d, L, dtype=torch.float64)
dt = 0.5 * torch.randn(2, d, L, dtype=torch.float64)
Alog = torch.randn(d, N, dtype=torch.float64) * 0.3
A_cn = -torch.exp(Alog)                                     # (d, N)
Bt = torch.randn(2, N, L, dtype=torch.float64)
bias = torch.randn(d, dtype=torch.float64)
got = selective_scan_ref(
    u.repeat(1, N, 1), dt.repeat(1, N, 1), A_cn.t().reshape(N * d, 1),
    Bt.unsqueeze(2), torch.ones(2, N, 1, L, dtype=torch.float64), None,
    delta_bias=bias.repeat(N), delta_softplus=True,
).view(2, N, d, L)
delta = F.softplus(dt + bias[None, :, None])
xs, xprev = [], torch.zeros(2, d, N, dtype=torch.float64)
for t in range(L):
    xprev = torch.exp(delta[:, :, t, None] * A_cn[None]) * xprev \
        + delta[:, :, t, None] * Bt[:, None, :, t] * u[:, :, t, None]
    xs.append(xprev)
want = torch.stack(xs, dim=-1).permute(0, 2, 1, 3)          # (b, N, d, L)
err_b = (got - want).abs().max().item()
assert err_b < 1e-12, err_b
print(f"B  N*d single-state channels == N-state recurrence (N={N}, max|diff| = {err_b:.1e})")


# ═══ C. N = 16 mixer == naive per-token transcription ════════════════════════
def naive_mixer(mx, x, Hh, Ww):
    Bb, Ll, _ = x.shape
    di, Ns, R = mx.d_inner, mx.d_state, mx.dt_rank
    xz = x @ mx.in_proj.weight.T
    u, z = (xz[..., :di], xz[..., di:]) if mx.gate else (xz, None)
    u = mx.act(mx.conv2d(u.reshape(Bb, Hh, Ww, di).permute(0, 3, 1, 2)))
    u = u.reshape(Bb, di, Ll)
    dbl = torch.einsum("od,bdl->bol", mx.x_proj_weight, u)
    dt_r, Bt, Ct = dbl[:, :R], dbl[:, R:R + Ns], dbl[:, R + Ns:]
    delta = F.softplus(torch.einsum("dr,brl->bdl", mx.dt_projs_weight, dt_r)
                       + mx.dt_projs_bias.float().to(x.dtype)[None, :, None])
    Av = -torch.exp(mx.A_logs.float()).to(x.dtype)            # (d, N)
    st, xprev = [], torch.zeros(Bb, di, Ns, dtype=x.dtype)
    for t in range(Ll):
        xprev = torch.exp(delta[:, :, t, None] * Av[None]) * xprev \
            + delta[:, :, t, None] * Bt[:, None, :, t] * u[:, :, t, None]
        st.append(xprev)
    st = torch.stack(st, dim=-1).reshape(Bb, di, Ns, Hh, Ww)
    sf = mx.state_fusion
    h = torch.zeros_like(st)
    for r in range(Hh):
        for c in range(Ww):
            acc = 0
            for a, k, dil in zip(sf.alpha, sf.kernels, sf.dilations):
                if mx.state_kernels == "per_state":
                    kk = k[:, 0].reshape(Ns, di, 3, 3).permute(1, 0, 2, 3)   # (d, N, 3, 3)
                else:
                    kk = k[:, 0][:, None].expand(di, Ns, 3, 3)
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        rr = min(max(r + i * dil, 0), Hh - 1)
                        cc = min(max(c + j * dil, 0), Ww - 1)
                        acc = acc + a * kk[:, :, i + 1, j + 1] * st[:, :, :, rr, cc]
            h[:, :, :, r, c] = acc
    h = h.reshape(Bb, di, Ns, Ll)
    y = (h * Ct[:, None, :, :]).sum(dim=2) + u * mx.Ds[None, :, None]
    y = F.layer_norm(y.transpose(1, 2), (di,), mx.out_norm.weight, mx.out_norm.bias,
                     mx.out_norm.eps)
    if z is not None:
        y = y * F.silu(z)
    return y @ mx.out_proj.weight.T


worst_c = 0.0
for gate in (True, False):
    for sk in ("shared", "per_state"):
        torch.manual_seed(3)
        mx = perturb(StructureAwareSSM(16, d_state=16, gate=gate, dilations=(1, 2, 3),
                                       state_kernels=sk), 0.1).double()
        x = torch.randn(2, HW, 16, dtype=torch.float64)
        with torch.no_grad():
            ref = naive_mixer(mx, x, H, W)
            dc = ((mx(x, H, W) - ref).abs().max() / ref.abs().max()).item()
        assert dc < 1e-12, (gate, sk, dc)
        worst_c = max(worst_c, dc)
print(f"C  N=16 mixer == naive transcription (shared / per-state kernels, gate on/off; "
      f"max relative diff = {worst_c:.1e})")


# ═══ D. shared == per_state with kernels tied across states ══════════════════
torch.manual_seed(4)
ms = perturb(StructureAwareSSM(16, d_state=16, gate=False, dilations=(1, 2, 3),
                               state_kernels="shared"), 0.1).double()
mp = StructureAwareSSM(16, d_state=16, gate=False, dilations=(1, 2, 3),
                       state_kernels="per_state").double()
sd = {k: v for k, v in ms.state_dict().items() if not k.startswith("state_fusion.kernels")}
mp.load_state_dict(sd, strict=False)
with torch.no_grad():
    for ks, kp in zip(ms.state_fusion.kernels, mp.state_fusion.kernels):
        kp.copy_(ks.repeat(16, 1, 1, 1))                    # channel n*d + c <- c
    x = torch.randn(2, HW, 16, dtype=torch.float64)
    dd = (ms(x, H, W) - mp(x, H, W)).abs().max().item()
assert dd < 1e-12, dd
print(f"D  shared == per_state with tied kernels (max|diff| = {dd:.1e})")


# ═══ E. param counts ═════════════════════════════════════════════════════════
pc = {(g, sk): n_params(StructureAwareSSM(384, d_state=16, gate=g, state_kernels=sk))
      for g in (True, False) for sk in ("shared", "per_state")}
assert pc == {(True, "shared"): 494_979, (True, "per_state"): 650_499,
              (False, "shared"): 347_523, (False, "per_state"): 503_043}, pc
TINY = dict(input_size=64, patch_size=8, hidden_size=384, depth=12, num_classes=200,
            bottleneck_dim=128, sasf_dilations=(1, 2, 3), gate=False, lpu="zero", d_state=16)
pm = {sk: n_params(JiTSpatialMamba(sasf_state_kernels=sk, **TINY)) for sk in ("shared", "per_state")}
assert pm == {"shared": 29_893_476, "per_state": 31_759_716}, pm
print(f"E  mixer/block (N=16): no gate {pc[(False, 'shared')]:,} / {pc[(False, 'per_state')]:,}, "
      f"gate {pc[(True, 'shared')]:,} / {pc[(True, 'per_state')]:,} (shared / per-state); "
      f"Tiny-IN no-gate + LPU: {pm['shared'] / 1e6:.2f}M / {pm['per_state'] / 1e6:.2f}M")


# ═══ F. smoke, grads, identity at init, bf16 ═════════════════════════════════
for sk in ("shared", "per_state"):
    torch.manual_seed(5)
    m = JiTSpatialMamba(d_state=16, gate=False, lpu="zero", sasf_state_kernels=sk,
                        sasf_dilations=(1, 2, 3), **KW)
    xb = torch.randn(Bsz, HW, KW["hidden_size"])
    cb = torch.randn(Bsz, KW["hidden_size"])
    with torch.no_grad():
        for blk in m.blocks:
            assert torch.equal(blk(xb, cb, H, W), xb), sk
    perturb(m, 0.02).train()
    out = m(x_img, t_in, y_in)
    assert out.shape == x_img.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    mx = m.blocks[-1].mixer
    assert mx.A_logs.shape == (KW["hidden_size"], 16)
    assert (mx.A_logs.grad.abs() > 0).all(dim=0).all(), "some state's A gets no gradient"
    for g in (mx.state_fusion.alpha.grad, mx.state_fusion.kernels[2].grad,
              m.blocks[-1].cpe1.weight.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0, sk
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(m(x_img, t_in, y_in).float()).all()
print("F  N=16 fwd+bwd OK (shared / per-state); grads reach every state's A, SASF, "
      "alpha, LPU; blocks identity at init; bf16 OK")


# ═══ G. guards ═══════════════════════════════════════════════════════════════
for bad in (lambda: StructureAwareSSM(32, d_state=16, state_kernels="per_map"),
            lambda: JiTSpatialMamba(d_state=16, sasf_state_kernels="per_map", **KW)):
    try:
        bad()
        raise SystemExit("guard missing: unknown state_kernels")
    except AssertionError:
        pass
print("G  unknown state_kernels rejected")


# ═══ I. grad_checkpoint: same output and gradients ═══════════════════════════
for sk in ("shared", "per_state"):
    outs, grads = [], []
    for ckpt in (False, True):
        torch.manual_seed(6)
        m = JiTSpatialMamba(d_state=16, gate=False, lpu="zero", sasf_state_kernels=sk,
                            sasf_dilations=(1, 2, 3), grad_checkpoint=ckpt, **KW)
        perturb(m, 0.02).train()
        out = m(x_img, t_in, y_in)
        out.square().mean().backward()
        outs.append(out.detach())
        grads.append({k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None})
    assert torch.equal(outs[0], outs[1]), sk
    assert grads[0].keys() == grads[1].keys(), sk
    assert all(torch.equal(grads[0][k], grads[1][k]) for k in grads[0]), sk
print("I  grad_checkpoint: bit-identical output and gradients (shared / per-state)")


# ═══ H. both build_model copies agree on the new configs ═════════════════════
try:
    import yaml                                             # noqa: E402
except ImportError as e:                                    # pragma: no cover
    print(f"H  SKIPPED (pyyaml not importable here: {e})")
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


bm_train = load_build_model("scripts/run_experiment.py")
bm_eval = load_build_model("scripts/evaluate.py")
for name, sk, n_expect, N, ckpt in (
        ("jit-s2-spatial-nogate-dstate16-shared.yaml", "shared", 29_893_476, 16, True),
        ("jit-s2-spatial-nogate-dstate16-perstate.yaml", "per_state", 31_759_716, 16, True),
        ("jit-s2-spatial-nogate-dstate4-shared.yaml", "shared", 29_727_588, 4, False),
        ("jit-s2-spatial-nogate-dstate4-perstate.yaml", "per_state", 30_100_836, 4, False)):
    cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "tiny_imagenet", name),
                              encoding="utf-8"))
    torch.manual_seed(0)
    mt = bm_train(cfg)
    torch.manual_seed(0)
    me = bm_eval(cfg)
    st, se = mt.state_dict(), me.state_dict()
    assert set(st) == set(se) and all(torch.equal(st[k], se[k]) for k in st), name
    assert (mt.d_state, mt.sasf_state_kernels, mt.gate, mt.lpu) == (N, sk, False, "zero"), name
    assert mt.sasf_dilations == (1, 2, 3) and n_params(mt) == n_expect, (name, n_params(mt))
    assert mt.grad_checkpoint == ckpt and all(b.grad_checkpoint == ckpt for b in mt.blocks), name
    print(f"H  {name}: train == eval build_model, {n_params(mt) / 1e6:.2f}M params")

print("ALL CHECKS PASSED")
