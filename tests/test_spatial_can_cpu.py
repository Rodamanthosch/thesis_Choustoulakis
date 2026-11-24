"""
tests/test_spatial_can_cpu.py
=============================
CPU-only verification of the CAN-style conditional kernels of
JiT-S2-Spatial-Mamba (src/models/spatial_mamba.py: dwconv_can / sasf_can,
CANGenerator, can_dwconv, set_can_zero; our adaptation of Cai et al., CVPR
2024, arXiv 2404.01143).

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, kept
in float64 when fed float64). Model-level checks run the whole model in
float64: the timestep embedder's sin-cos table is built in fp32 by design, so
`to64` casts it to float64 before its MLP -- for every model compared alike.
Only the plain no-gate setup is exercised (d_state 1, lpu zero, swiglu).

Checks:
  A. both flags off are BIT-IDENTICAL to the last commit before the feature
     (pinned 9b454ff; CAN_REFERENCE=<rev> overrides): state_dicts and outputs
     for the no-gate setup and gate / lpu variants.
  B. batched == naive per-sample loop, in the mixer (float64): sample b through
     a copy whose DWConv weight is K + dK_dw[b] and SASF kernels K_j +
     dK_sasf[b, j], without CAN -- each site alone and both.
  C. the model end to end (float64, perturbed generators) == a per-sample
     loop over flags-off models whose blocks' static kernels are set to
     K_l + G(c_b).
  D. zero-init: same seed, flags on vs off -> every shared weight identical,
     generators zero, outputs equal (max|diff| printed); set_can_zero on both
     sites == the flags-off model EXACTLY; zeroing one site == the model with
     only the other site's flag, exactly.
  E. G(c) is computed once per forward: each generator's forward runs once
     for a depth-4 model and every block's mixer receives the SAME dK tensors.
  F. params: +9 D d_inner (dwconv) and +9 n_dil D d_inner (sasf); gradients
     into both generators; grad_checkpoint bit-identical; under CPU bf16
     autocast the CAN convs run in fp32 with autocast off; the DWConv output
     has the static conv's dtype and the SASF output its input's dtype.
  G. guards: tokens, aux tokens, d_state > 1, feature_rope, ssm_norm, Derf,
     conv FFNs; set_can_zero on a site that is off.
  H. scripts/measure_can.py: the Gauss-Hermite nodes integrate the training
     t density (trapezoid on z, 1e-9); zero generators -> NaN shares; the
     constant / time / class / interaction energy shares == a direct two-way
     decomposition of the full (t, class) grid and sum to 1; size ratios;
     the time component T(t) = mean_y dK(t, .) - m: its SVD energy == the time
     energy, the per-channel / one-scale projections on K_l + m, the centre
     share and ||T(t)|| / ||K_l + m|| == naive loops; a rank-1 generator along
     K_l gives one direction, projection share 1 and the same s(t) in every
     block; a centre-only generator gives centre share 1; reports written
     (also with NaN shares); a dwconv-only model has only the dwconv site.
  I. set_can_freeze (evaluate.py --can_freeze): "all" at both sites == ONE
     flags-off model with static kernels K_l + E_t mean_y G; every other case
     ("class", "time", "all" at one site) == a per-sample loop over static
     models with the naive averages (repeated t, null class, either site alone
     or both); None restores the normal output exactly; the "all" table's
     energy == the constant part; weights changed after the call are refused;
     guards.
  J. wiring: the config builds IDENTICAL models through both build_model
     copies, 34,994,532 params = jit-s2-spatial-nogate + 5,308,416 (needs
     pyyaml).

Run from the repo root:
    PYTHONPATH=. python tests/test_spatial_can_cpu.py
"""
import ast
import copy
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

import src.models.spatial_mamba as spatial_mod              # noqa: E402  (after the stub)
from src.models.spatial_mamba import JiTSpatialMamba, StructureAwareSSM   # noqa: E402

# The plain no-gate setup of the Tiny-IN run, at toy size.
KW = dict(input_size=16, patch_size=4, hidden_size=32, depth=4, num_classes=7,
          bottleneck_dim=16, sasf_dilations=(1, 2, 3), d_state=1, gate=False,
          lpu="zero", ffn="swiglu")
H = W = KW["input_size"] // KW["patch_size"]       # 4
D = KW["hidden_size"]
N_DIL = len(KW["sasf_dilations"])
Bsz = 3
BOTH = dict(dwconv_can=True, sasf_can=True)
torch.manual_seed(0)
x_img = torch.randn(Bsz, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(Bsz)
y_in = torch.randint(0, KW["num_classes"], (Bsz,))
x64 = x_img.double()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def perturb(module, scale=0.05):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))
    return module


def to64(m):
    """Whole model in float64 (the fp32 sin-cos table cast before the MLP)."""
    m.double()
    te = m.t_embedder
    te.forward = lambda t: te.mlp(te.timestep_embedding(t, te.frequency_embedding_size).double())
    return m


def build(seed=1, **kw):
    torch.manual_seed(seed)
    return JiTSpatialMamba(**dict(KW, **kw))


# ═══ A. flags off == the pre-feature commit, bit for bit ══════════════════════
REF = os.environ.get("CAN_REFERENCE", "9b454ff")
try:
    head_src = subprocess.run(
        ["git", "show", f"{REF}:src/models/spatial_mamba.py"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8")
except Exception as e:                                      # pragma: no cover
    head_src = None
    print(f"A  SKIPPED (cannot read {REF}:src/models/spatial_mamba.py: {e})")

if head_src is not None:
    head_mod = types.ModuleType("spatial_mamba_ref")
    exec(compile(head_src, f"<{REF}:src/models/spatial_mamba.py>", "exec"), head_mod.__dict__)
    n_cfg = 0
    for extra in ({}, dict(gate=True), dict(lpu="none"), dict(lpu="paper")):
        kw = dict(KW, **extra)
        torch.manual_seed(1)
        old = head_mod.JiTSpatialMamba(**kw).eval()
        torch.manual_seed(1)
        new = JiTSpatialMamba(**kw).eval()
        so, sn = old.state_dict(), new.state_dict()
        assert set(so) == set(sn), (kw, set(so) ^ set(sn))
        assert all(torch.equal(so[k], sn[k]) for k in so), kw
        perturb(old, 0.02)
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            d = (old(x_img, t_in, y_in) - new(x_img, t_in, y_in)).abs().max().item()
        assert d == 0.0, (kw, d)
        n_cfg += 1
    print(f"A  flags off: bit-identical to {REF} (init AND output) for {n_cfg} configs")


# ═══ B. mixer: batched == per-sample loop (float64) ═══════════════════════════
def static_copy(mx, dk_dw, dk_sasf, b):
    """A CAN-free copy of the mixer whose static kernels are K + dK[b]."""
    m = copy.deepcopy(mx)
    with torch.no_grad():
        if dk_dw is not None:
            m.conv2d.weight.add_(dk_dw[b].unsqueeze(1))
        if dk_sasf is not None:
            for j, k in enumerate(m.state_fusion.kernels):
                k.add_(dk_sasf[b, j].unsqueeze(1))
    return m


torch.manual_seed(2)
mx = perturb(StructureAwareSSM(12, d_state=1, gate=False, dilations=(1, 2)), 0.2).double().eval()
xin = torch.randn(Bsz, HW := H * W, 12, dtype=torch.float64)
dk_dw = 0.3 * torch.randn(Bsz, 12, 3, 3, dtype=torch.float64)
dk_sasf = 0.3 * torch.randn(Bsz, 2, 12, 3, 3, dtype=torch.float64)
worst = 0.0
for name, can in (("dwconv", (dk_dw, None)), ("sasf", (None, dk_sasf)), ("both", (dk_dw, dk_sasf))):
    with torch.no_grad():
        got = mx(xin, H, W, can)
        want = torch.cat([static_copy(mx, *can, b)(xin[b:b + 1], H, W) for b in range(Bsz)])
        static = mx(xin, H, W)
    err = (got - want).abs().max().item()
    assert err < 1e-12, (name, err)
    assert (got - static).abs().max() > 1e-3, name                   # dK does something
    worst = max(worst, err)
print(f"B  mixer: batched per-sample kernels == per-sample loop for dwconv, sasf, both "
      f"(max|diff| = {worst:.1e})")


# ═══ C. the model end to end == per-sample loop over static models (float64) ══
m_can = to64(perturb(build(**BOTH), 0.05)).eval()
with torch.no_grad():
    got = m_can(x64, t_in.double(), y_in)
    c = m_can.t_embedder(t_in.double()) + m_can.y_embedder(y_in)
    dk_dw, dk_sasf = m_can.can_deltas(c)
    assert dk_dw.dtype == dk_sasf.dtype == torch.float64
    want = []
    for b in range(Bsz):
        ref = to64(build())
        missing, unexpected = ref.load_state_dict(m_can.state_dict(), strict=False)
        assert not missing and set(unexpected) == {"can_dwconv.weight", "can_sasf.weight"}
        for blk in ref.blocks:
            blk.mixer.conv2d.weight.add_(dk_dw[b].unsqueeze(1))
            for j, k in enumerate(blk.mixer.state_fusion.kernels):
                k.add_(dk_sasf[b, j].unsqueeze(1))
        want.append(ref.eval()(x64[b:b + 1], t_in[b:b + 1].double(), y_in[b:b + 1]))
    want = torch.cat(want)
err_c = (got - want).abs().max().item()
assert err_c < 1e-12, err_c
print(f"C  model: per-sample K_l + G(c_b) in every block == loop over static models "
      f"(max|diff| = {err_c:.1e})")


# ═══ D. zero-init == the baseline; set_can_zero == static path ════════════════
m_off, m_on = build(), build(**BOTH)
so, sn = m_off.state_dict(), m_on.state_dict()
assert set(sn) - set(so) == {"can_dwconv.weight", "can_sasf.weight"} and set(so) <= set(sn)
assert all(torch.equal(so[k], sn[k]) for k in so)                      # no RNG drawn
assert not m_on.can_dwconv.weight.any() and not m_on.can_sasf.weight.any()
perturb(m_off, 0.05)
m_on.load_state_dict(m_off.state_dict(), strict=False)                  # generators stay 0
to64(m_off).eval(), to64(m_on).eval()
with torch.no_grad():
    o_off = m_off(x64, t_in.double(), y_in)
    err_d = (m_on(x64, t_in.double(), y_in) - o_off).abs().max().item()
assert err_d < 1e-12, err_d

perturb(m_on.can_dwconv, 0.1), perturb(m_on.can_sasf, 0.1)
m_on.set_can_zero(dwconv=True, sasf=True)
with torch.no_grad():
    assert torch.equal(m_on(x64, t_in.double(), y_in), o_off)           # static path, exact
m_sasf_only = to64(build(sasf_can=True))
m_sasf_only.load_state_dict(m_on.state_dict(), strict=False)
m_on.set_can_zero(dwconv=True)
m_dw_only = to64(build(dwconv_can=True))
m_dw_only.load_state_dict(m_on.state_dict(), strict=False)
with torch.no_grad():
    assert torch.equal(m_on(x64, t_in.double(), y_in), m_sasf_only.eval()(x64, t_in.double(), y_in))
    m_on.set_can_zero(sasf=True)
    assert torch.equal(m_on(x64, t_in.double(), y_in), m_dw_only.eval()(x64, t_in.double(), y_in))
    m_on.set_can_zero()
    assert (m_on(x64, t_in.double(), y_in) - o_off).abs().max() > 1e-6
print(f"D  zero-init: shared weights identical, generators 0, output == flags off "
      f"(max|diff| = {err_d:.1e}); set_can_zero both == flags off exactly, one site "
      f"== the other site's flag alone exactly")


# ═══ E. G(c) once per forward, the same dK in every block ═════════════════════
m = perturb(build(**BOTH), 0.05).eval()
calls, seen = {"can_dwconv": 0, "can_sasf": 0}, []
for name in calls:
    getattr(m, name).register_forward_hook(
        lambda mod, inp, out, name=name: calls.__setitem__(name, calls[name] + 1))
for blk in m.blocks:
    blk.mixer.register_forward_pre_hook(lambda mod, args: seen.append(args[3]))
with torch.no_grad():
    m(x_img, t_in, y_in)
assert calls == {"can_dwconv": 1, "can_sasf": 1}, calls
assert len(seen) == KW["depth"]
assert all(s[0] is seen[0][0] and s[1] is seen[0][1] for s in seen)
print(f"E  each generator runs once per forward; all {KW['depth']} mixers get the same dK tensors")


# ═══ F. params, grads, grad_checkpoint, dtypes under bf16 autocast ════════════
base = build()
assert n_params(build(dwconv_can=True)) - n_params(base) == 9 * D * D
assert n_params(build(sasf_can=True)) - n_params(base) == 9 * N_DIL * D * D
assert n_params(build(**BOTH)) - n_params(base) == 9 * D * D * (1 + N_DIL)

res = []
for ck in (False, True):
    m = perturb(build(**BOTH, grad_checkpoint=ck), 0.05).train()
    out = m(x_img, t_in, y_in)
    out.square().mean().backward()
    g = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    for n in ("can_dwconv.weight", "can_sasf.weight"):
        assert torch.isfinite(g[n]).all() and g[n].abs().sum() > 0, n
    res.append((out.detach(), g))
assert torch.equal(res[0][0], res[1][0]) and set(res[0][1]) == set(res[1][1])
assert all(torch.equal(res[0][1][n], res[1][1][n]) for n in res[0][1])

conv_calls, inside = [], [False]
orig_conv, orig_can = F.conv2d, spatial_mod.can_dwconv


def spy_conv(x, w, *a, **k):
    if inside[0]:
        conv_calls.append((x.dtype, w.dtype, torch.is_autocast_enabled("cpu")))
    return orig_conv(x, w, *a, **k)


def spy_can(*a, **k):
    inside[0] = True
    try:
        return orig_can(*a, **k)
    finally:
        inside[0] = False


def site_dtypes(model):
    rec = {}
    blk = model.blocks[0].mixer
    h1 = blk.act.register_forward_hook(lambda mod, inp, out: rec.__setitem__("dw", inp[0].dtype))
    h2 = blk.state_fusion.register_forward_hook(
        lambda mod, inp, out: rec.update(sasf_in=inp[0].dtype, sasf_out=out.dtype))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x_img, t_in, y_in)
    h1.remove(), h2.remove()
    assert torch.isfinite(out.float()).all()
    return rec


m_on = perturb(build(**BOTH), 0.05).eval()
m_off = build()
m_off.load_state_dict(m_on.state_dict(), strict=False)
spatial_mod.can_dwconv, F.conv2d = spy_can, spy_conv
try:
    rec_on = site_dtypes(m_on)
finally:
    spatial_mod.can_dwconv, F.conv2d = orig_can, orig_conv
rec_off = site_dtypes(m_off.eval())
assert conv_calls and all(xd == wd == torch.float32 and not ac for xd, wd, ac in conv_calls), conv_calls
# DWConv: CAN output dtype == the static conv's under autocast. SASF: CAN casts
# back to the states' dtype (on the GPU the kernel returns u's dtype, so the
# static conv gives the same; the CPU stub returns fp32 states).
assert rec_on["dw"] == rec_off["dw"] == torch.bfloat16, (rec_on, rec_off)
assert rec_on["sasf_out"] == rec_on["sasf_in"], rec_on
print(f"F  params +9*D*d_inner (dwconv) / +27*D*d_inner (sasf, 3 dilations); grads into "
      f"both generators; grad_checkpoint bit-identical; bf16 autocast: {len(conv_calls)} CAN "
      f"convs in fp32 with autocast off; DWConv output {rec_on['dw']} as the static path, "
      f"SASF output = its input dtype ({rec_on['sasf_in']})")


# ═══ G. guards ════════════════════════════════════════════════════════════════
def fails(**kw):
    try:
        JiTSpatialMamba(**dict(KW, **kw))
    except (AssertionError, ValueError):
        return True
    return False


for flag in ("dwconv_can", "sasf_can"):
    on = {flag: True}
    assert fails(**on, in_context_len=1)
    assert fails(**on, aux_tokens="swap")
    assert fails(**on, d_state=2)
    assert fails(**on, feature_rope="1d")
    assert fails(**on, ssm_norm="dt")
    assert fails(**on, norm="derf")
    assert fails(**on, ffn="convglu")
    assert not fails(**on)
for site, other in (("dwconv", "sasf_can"), ("sasf", "dwconv_can")):
    try:
        build(**{other: True}).set_can_zero(**{site: True})
        raise RuntimeError("set_can_zero accepted a site that is off")
    except AssertionError:
        pass
print("G  guards: tokens, aux tokens, d_state > 1, feature_rope, ssm_norm, Derf, conv FFN; "
      "set_can_zero on an off site")


# ═══ H. the diagnostic: energy decomposition on the training t density ═══════
import tempfile                                             # noqa: E402
import math                                                 # noqa: E402
from scripts.measure_can import measure, write_reports     # noqa: E402
from src.models.spatial_mamba import logit_normal_nodes    # noqa: E402

PM, PS, NN, NC = -0.8, 0.8, 8, KW["num_classes"]           # the configs' P_mean / P_std

# The quadrature is the training density t = sigmoid(N(P_mean, P_std)).
ts_q, ws_q = logit_normal_nodes(PM, PS, 64)
z = torch.linspace(PM - 12 * PS, PM + 12 * PS, 400001, dtype=torch.float64)
pdf = torch.exp(-0.5 * ((z - PM) / PS) ** 2) / (PS * math.sqrt(2 * math.pi))
for f in (lambda t: t, lambda t: t ** 2, lambda t: torch.cos(7 * t)):
    exact = torch.trapezoid(f(torch.sigmoid(z)) * pdf, z).item()
    assert abs((ws_q * f(ts_q)).sum().item() - exact) < 1e-9, exact
assert abs(ws_q.sum().item() - 1) < 1e-14 and ((ts_q > 0) & (ts_q < 1)).all()

res0 = measure(build(**BOTH).eval(), PM, PS, NN)
assert set(res0["sites"]) == {"dwconv", "sasf_d1", "sasf_d2", "sasf_d3", "sasf"}
for a in res0["sites"].values():
    assert a["energy"] == 0 and all(math.isnan(v) for v in a["shares"].values())

m_h = perturb(build(**BOTH), 0.05).eval()
with torch.no_grad():
    m_h.can_dwconv.weight.normal_(0, 0.05), m_h.can_sasf.weight.normal_(0, 0.05)
to64(m_h)
res = measure(m_h, PM, PS, NN)

ts, ws = logit_normal_nodes(PM, PS, NN)                     # the full grid, directly
with torch.no_grad():
    c_grid = torch.stack([m_h.t_embedder(torch.full((NC,), t)) + m_h.y_embedder(torch.arange(NC))
                          for t in ts.tolist()])            # (T, NC, D)
    g_dw = m_h.can_dwconv(c_grid.flatten(0, 1)).view(NN, NC, -1)
    g_sa = m_h.can_sasf(c_grid.flatten(0, 1)).view(NN, NC, N_DIL, -1)
grids = {"dwconv": g_dw, "sasf_d2": g_sa[:, :, 1], "sasf": g_sa.flatten(2)}
w3 = ws[:, None, None]
for site, G in grids.items():
    mean = (w3 * G).sum(0).mean(0)                          # m
    a_t = G.mean(1) - mean                                  # (T, P)
    b_y = (w3 * G).sum(0) - mean                            # (NC, P)
    r = G - mean - a_t[:, None] - b_y[None]
    E = (ws[:, None] * G.pow(2).sum(-1)).sum(0).mean().item()
    parts = {"constant": mean.pow(2).sum().item(),
             "time": (ws * a_t.pow(2).sum(-1)).sum().item(),
             "class": b_y.pow(2).sum(-1).mean().item(),
             "interaction": (ws[:, None] * r.pow(2).sum(-1)).sum(0).mean().item()}
    assert abs(sum(parts.values()) - E) < 1e-12 * E                         # orthogonal
    got = res["sites"][site]
    assert abs(got["energy"] - E) < 1e-12 * E, site
    for k, v in parts.items():
        assert abs(got["shares"][k] - v / E) < 1e-10, (site, k, got["shares"][k], v / E)
    assert all(0 <= v <= 1 for v in got["shares"].values())
    assert abs(sum(got["shares"].values()) - 1) < 1e-12
K3 = m_h.blocks[3].mixer.conv2d.weight[:, 0].double()
assert abs(res["sites"]["dwconv"]["size"][3]["rms_ratio"]
           - math.sqrt(res["sites"]["dwconv"]["energy"]) / K3.norm().item()) < 1e-12

# The time component T(t) = mean_y dK(t, .) - m, against M_l = K_l + m: naive recomputation.
statics_h = {"dwconv": [b.mixer.conv2d.weight[:, 0].double().flatten() for b in m_h.blocks],
             "sasf_d2": [b.mixer.state_fusion.kernels[1][:, 0].double().flatten() for b in m_h.blocks],
             "sasf": [torch.stack([k[:, 0] for k in b.mixer.state_fusion.kernels]).double().flatten()
                      for b in m_h.blocks]}
for site, G in grids.items():
    mean = (w3 * G).sum(0).mean(0)
    Tn = G.mean(1) - mean                                   # (T, P)
    tm = res["sites"][site]["time"]
    e_t = (ws * Tn.pow(2).sum(-1)).sum().item()
    assert abs(tm["energy"] - e_t) < 1e-12 * e_t
    assert abs(tm["energy"] - res["sites"][site]["shares"]["time"] * res["sites"][site]["energy"]) < 1e-10 * e_t
    sv = torch.tensor(tm["rank"]["sv"], dtype=torch.float64)
    assert abs(sv.pow(2).sum().item() - e_t) < 1e-10 * e_t                  # SVD holds the time energy
    assert abs(sum(tm["rank"]["frac"]) - 1) < 1e-12 and 1 <= tm["rank"]["n90"] <= tm["rank"]["n99"]
    cum = torch.tensor(tm["rank"]["frac"]).cumsum(0)
    assert cum[tm["rank"]["n90"] - 1] >= 0.9 - 1e-12 and (tm["rank"]["n90"] == 1 or cum[tm["rank"]["n90"] - 2] < 0.9)
    T3 = Tn.view(NN, -1, 9)
    for i in (0, NN // 2, NN - 1):
        tt = T3[i].pow(2).sum().item()
        assert abs(tm["centre"]["per_t"][i] - T3[i, :, 4].pow(2).sum().item() / tt) < 1e-12
        for l in (0, 2):
            M3 = (statics_h[site][l] + mean).view(-1, 9)
            dots = [(T3[i, ch] @ M3[ch]).item() for ch in range(M3.shape[0])]       # per channel
            p_ch = sum(d * d / float(M3[ch] @ M3[ch]) for ch, d in enumerate(dots))
            a = sum(dots) / M3.pow(2).sum().item()
            b = tm["blocks"][l]
            assert abs(b["per_t"]["share_channel"][i] - p_ch / tt) < 1e-10
            assert abs(b["per_t"]["scale"][i] - a) < 1e-10 * max(1, abs(a))
            assert abs(b["per_t"]["share_one"][i] - a * a * M3.pow(2).sum().item() / tt) < 1e-10
            assert abs(b["per_t"]["ratio"][i] - math.sqrt(tt) / M3.norm().item()) < 1e-12
            assert b["per_t"]["share_one"][i] <= b["per_t"]["share_channel"][i] + 1e-12 <= 1 + 2e-12

# Constructed case: dK = (v . SiLU(c)) u (rank-1 generator) and K_l = lam_l u, so
# T(t) = s(t) u and M_l = (lam_l + mu) u: one direction, fully along M_l, with
# a_l(t) (lam_l + mu) = s(t) the same in every block; centre-only u -> centre share 1.
m_k = to64(build(dwconv_can=True).eval())
P_dw = m_k.can_dwconv.weight.shape[0]
torch.manual_seed(5)
for centre_only in (False, True):
    u = torch.randn(P_dw, dtype=torch.float64).view(-1, 9)
    if centre_only:
        u[:, [0, 1, 2, 3, 5, 6, 7, 8]] = 0
    v = torch.randn(D, dtype=torch.float64)
    lam = [20.0 * (l + 1) for l in range(len(m_k.blocks))]     # lam_l + mu kept away from 0
    with torch.no_grad():
        m_k.can_dwconv.weight.copy_(u.flatten()[:, None] * v[None])
        for l, b in enumerate(m_k.blocks):
            b.mixer.conv2d.weight.copy_(lam[l] * u.view_as(b.mixer.conv2d.weight[:, 0]).unsqueeze(1))
    tk = measure(m_k, PM, PS, NN)["sites"]["dwconv"]["time"]
    assert tk["rank"]["n90"] == tk["rank"]["n99"] == 1 and abs(tk["rank"]["frac"][0] - 1) < 1e-12
    s_l = []
    for b in tk["blocks"]:
        assert abs(b["share_one"] - 1) < 1e-10 and abs(b["share_channel"] - 1) < 1e-10
        s_l.append(torch.tensor(b["per_t"]["scale"], dtype=torch.float64) * b["M_norm"] / u.norm().item())   # = |lam_l+mu| a_l(t)
    assert all(torch.allclose(s_l[0].abs(), s.abs(), rtol=1e-10, atol=1e-14) for s in s_l)
    if centre_only:
        assert abs(tk["centre"]["share"] - 1) < 1e-12
        assert all(abs(c - 1) < 1e-12 for c in tk["centre"]["per_t"])

with tempfile.TemporaryDirectory() as tmp:
    write_reports(res, tmp)
    write_reports(res0, tmp + "/zero")                      # NaN shares do not break the reports
    for f in ("can_energy.csv", "can_size.csv", "can_time_rank.csv", "can_time_alpha.csv",
              "can_time_centre.csv", "can_time_ratio.csv", "summary.json"):
        assert os.path.getsize(os.path.join(tmp, f)) > 0, f
    import json as _json                                    # noqa: E402
    assert "per_t" not in open(os.path.join(tmp, "summary.json"), encoding="utf-8").read()
    _json.load(open(os.path.join(tmp, "summary.json"), encoding="utf-8"))
assert set(measure(build(dwconv_can=True).eval(), PM, PS, 3)["sites"]) == {"dwconv"}
sh = res["sites"]["sasf"]["shares"]
print(f"H  measure_can: quadrature == the training t density (1e-9); zero generators -> "
      f"NaN shares; constant / time / class / interaction == a direct two-way decomposition "
      f"of the full grid, summing to 1 (e.g. sasf {sh['constant']:.3f} / {sh['time']:.3f} / "
      f"{sh['class']:.3f} / {sh['interaction']:.3f}); size ratios; time component: SVD "
      f"holds the time energy, projections / centre share / ||T||/||M|| == naive loops, "
      f"rank-1 generator along K_l -> 1 direction, share 1, same s(t) in every block; "
      f"centre-only -> centre share 1; reports written")


# ═══ I. --can_freeze: all / class / time on the same weights ══════════════════
def g_one(m, site, t, y):
    """G(c(t, y)) for one (t, y), one row at a time (float64 model)."""
    c = m.t_embedder(torch.tensor([float(t)])) + m.y_embedder(torch.tensor([int(y)]))
    return (m.can_dwconv if site == "dwconv" else m.can_sasf)(c)[0]


def naive_frozen(m, site, mode, t, y):
    if mode == "all":
        return sum(w * torch.stack([g_one(m, site, ti, yy) for yy in range(NC)]).mean(0)
                   for ti, w in zip(ts.tolist(), ws.tolist()))
    if mode == "class":
        return torch.stack([g_one(m, site, t, yy) for yy in range(NC)]).mean(0)
    return sum(w * g_one(m, site, ti, y) for ti, w in zip(ts.tolist(), ws.tolist()))


def static_ref(src, dk_dw, dk_sasf):
    """A flags-off model with src's weights and static kernels K_l + dK."""
    ref = to64(build())
    ref.load_state_dict(src.state_dict(), strict=False)
    with torch.no_grad():
        for blk in ref.blocks:
            blk.mixer.conv2d.weight.add_(dk_dw.unsqueeze(1))
            for j, k in enumerate(blk.mixer.state_fusion.kernels):
                k.add_(dk_sasf[j].unsqueeze(1))
    return ref.eval()


m_f = to64(perturb(build(**BOTH), 0.05)).eval()
t_f = torch.tensor([0.3, 0.7, 0.3], dtype=torch.float64)    # a repeated t (class mode)
y_f = torch.tensor([1, 4, NC])                              # and the null class (time mode)
with torch.no_grad():
    o_plain = m_f(x64, t_f, y_f)
worst, n_i = 0.0, 0
for mode in ("all", "class", "time"):
    for sites in (("dwconv", "sasf"), ("sasf",), ("dwconv",)):
        m_f.set_can_freeze(mode, sites=sites, P_mean=PM, P_std=PS, n_nodes=NN)
        with torch.no_grad():
            got = m_f(x64, t_f, y_f)
            if mode == "all" and len(sites) == 2:
                # one fixed kernel per site: ONE flags-off model for the batch
                want = static_ref(m_f, naive_frozen(m_f, "dwconv", "all", None, None),
                                  naive_frozen(m_f, "sasf", "all", None, None))(x64, t_f, y_f)
            else:                       # per sample (the unfrozen site reads G(c_b))
                rows = []
                for b in range(Bsz):
                    tb, yb = t_f[b].item(), y_f[b].item()
                    dks = {s: (naive_frozen(m_f, s, mode, tb, yb) if s in sites
                               else g_one(m_f, s, tb, yb)) for s in ("dwconv", "sasf")}
                    rows.append(static_ref(m_f, dks["dwconv"], dks["sasf"])(
                        x64[b:b + 1], t_f[b:b + 1], y_f[b:b + 1]))
                want = torch.cat(rows)
        err = (got - want).abs().max().item()
        assert err < 1e-12, (mode, sites, err)
        assert (got - o_plain).abs().max() > 1e-6, (mode, sites)     # it does change dK
        worst, n_i = max(worst, err), n_i + 1
m_f.set_can_freeze(None)
with torch.no_grad():
    assert torch.equal(m_f(x64, t_f, y_f), o_plain)

# the "all" table is the decomposition's constant part, on the same nodes
m_h.set_can_freeze("all", P_mean=PM, P_std=PS, n_nodes=NN)
for s in ("dwconv", "sasf"):
    e_c = res["sites"][s]["shares"]["constant"] * res["sites"][s]["energy"]
    assert abs(m_h.can_freeze["tables"][s].pow(2).sum().item() - e_c) < 1e-10 * e_c, s
m_h.set_can_freeze(None)

# weights changed after set_can_freeze (e.g. an EMA swap) -> refuse to run
m_f.set_can_freeze("time", P_mean=PM, P_std=PS, n_nodes=NN)
with torch.no_grad():
    m_f.can_sasf.weight.add_(0.0)
try:
    with torch.no_grad():
        m_f(x64, t_f, y_f)
    raise RuntimeError("stale freeze tables were used")
except AssertionError:
    pass
m_f.set_can_freeze(None)


def refused(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


assert refused(lambda: m_f.set_can_freeze("both", P_mean=PM, P_std=PS))
assert refused(lambda: m_f.set_can_freeze("all"))                         # no density
assert refused(lambda: m_f.set_can_freeze("time", P_std=PS))
assert refused(lambda: build(dwconv_can=True).set_can_freeze("class", sites=("sasf",)))
m_f.set_can_zero(dwconv=True)
assert refused(lambda: m_f.set_can_freeze("class"))
m_f.set_can_zero()
m_f.set_can_freeze("class")                                               # no density needed
assert refused(lambda: m_f.set_can_zero(sasf=True))
m_f.set_can_freeze(None)
print(f"I  can_freeze all / class / time x (both, sasf, dwconv) == static models with the "
      f"naive averages ({n_i} configs, max|diff| = {worst:.1e}; all == ONE flags-off model "
      f"with K_l + mean dK); off == normal exactly; 'all' table == the constant part; "
      f"stale weights refused; guards")


# ═══ J. both build_model copies agree on the config ═══════════════════════════
try:
    import yaml                                             # noqa: E402
except ImportError as e:                                    # pragma: no cover
    print(f"J  SKIPPED (pyyaml not importable here: {e})")
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
cfg_dir = os.path.join(REPO_ROOT, "configs", "tiny_imagenet")
base_cfg = yaml.safe_load(open(os.path.join(cfg_dir, "jit-s2-spatial-nogate.yaml"), encoding="utf-8"))
name = "jit-s2-spatial-nogate-can.yaml"
cfg = yaml.safe_load(open(os.path.join(cfg_dir, name), encoding="utf-8"))
assert {k: v for k, v in cfg["model"].items() if not k.endswith("_can")} == base_cfg["model"]
torch.manual_seed(0)
mt = bm_train(cfg)
torch.manual_seed(0)
me = bm_eval(cfg)
st, se = mt.state_dict(), me.state_dict()
assert set(st) == set(se) and all(torch.equal(st[k], se[k]) for k in st), name
assert (mt.dwconv_can, mt.sasf_can, me.dwconv_can, me.sasf_can) == (True,) * 4
torch.manual_seed(0)
mb = bm_train(base_cfg)
n_base = n_params(mb)
assert n_base == 29_686_116, n_base
assert n_params(mt) == 34_994_532 == n_base + 9 * 384 * 384 + 27 * 384 * 384, n_params(mt)
sb = mb.state_dict()
assert all(torch.equal(sb[k], st[k]) for k in sb)               # same init at the same seed
print(f"J  {name}: train == eval build_model, {n_params(mt):,} params "
      f"(no-gate {n_base:,} + 5,308,416); every baseline weight identical at seed 0")

print("ALL CHECKS PASSED")
