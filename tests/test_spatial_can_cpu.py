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
  H. scripts/measure_can.py: zero generators -> every ratio 0, fractions NaN;
     perturbed -> ratios and fractions finite and > 0, one (t, class, block)
     entry of each quantity == a direct recomputation; the reports are written
     and a dwconv-only model has only the dwconv site.
  I. wiring: the config builds IDENTICAL models through both build_model
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


# ═══ H. the diagnostic script ════════════════════════════════════════════════
import tempfile                                             # noqa: E402
import numpy as np                                          # noqa: E402
from scripts.measure_can import measure, write_reports     # noqa: E402

res0 = measure(build(**BOTH).eval(), t_steps=5)
assert set(res0["sites"]) == {"dwconv", "sasf_d1", "sasf_d2", "sasf_d3", "sasf"}
for a in res0["sites"].values():
    assert (a["ratio"] == 0).all() and np.isnan(a["class_frac"]).all() and np.isnan(a["time_frac"]).all()

m = perturb(build(**BOTH), 0.05).eval()
with torch.no_grad():
    m.can_dwconv.weight.normal_(0, 0.05), m.can_sasf.weight.normal_(0, 0.05)
to64(m)                          # float64 embedders: batched == one-by-one c
res = measure(m, t_steps=5)
for a in res["sites"].values():
    for k in ("ratio", "class_frac", "time_frac"):
        assert np.isfinite(a[k]).all() and (a[k] > 0).all(), k

ti, yi, li, NC = 2, 1, 3, KW["num_classes"]                 # one entry, recomputed directly
with torch.no_grad():
    ts = torch.linspace(0, 1, 5)

    def dk_at(t, y):
        c = m.t_embedder(torch.tensor([float(t)])) + m.y_embedder(torch.tensor([y]))
        return m.can_dwconv(c)[0], m.can_sasf(c)[0]

    dw, sa = dk_at(ts[ti], yi)
    K_dw = m.blocks[li].mixer.conv2d.weight[:, 0].double()
    K_s2 = m.blocks[li].mixer.state_fusion.kernels[1][:, 0].double()
    mean_y = torch.stack([dk_at(ts[ti], y)[0] for y in range(NC)]).mean(0)
    mean_t = torch.stack([dk_at(t, yi)[1] for t in ts]).mean(0)
    want = {("dwconv", "ratio"): (dw.norm() / K_dw.norm()).item(),
            ("sasf_d2", "ratio"): (sa[1].norm() / K_s2.norm()).item(),
            ("dwconv", "class_frac"): ((dw - mean_y).norm() / dw.norm()).item(),
            ("sasf", "time_frac"): ((sa - mean_t).norm() / sa.norm()).item()}
for (site, k), v in want.items():
    got_v = res["sites"][site][k][li, ti, yi] if k == "ratio" else res["sites"][site][k][ti, yi]
    assert abs(got_v - v) < 1e-10 * max(1.0, abs(v)), (site, k, got_v, v)

with tempfile.TemporaryDirectory() as tmp:
    summary = write_reports(res, tmp, NC)
    for f in ("can_ratio.csv", "can_split.csv", "can_split_by_class.csv",
              "can_diagnostic.npz", "summary.json"):
        assert os.path.getsize(os.path.join(tmp, f)) > 0, f
assert len(summary["sites"]["dwconv"]["ratio_mean_per_block"]) == KW["depth"]
assert set(measure(build(dwconv_can=True).eval(), 3)["sites"]) == {"dwconv"}
print("H  measure_can: zero generators -> ratios 0, fractions NaN; trained-like -> finite > 0; "
      "ratio, class and time fractions == direct recomputation; reports written")


# ═══ I. both build_model copies agree on the config ═══════════════════════════
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
print(f"I  {name}: train == eval build_model, {n_params(mt):,} params "
      f"(no-gate {n_base:,} + 5,308,416); every baseline weight identical at seed 0")

print("ALL CHECKS PASSED")
