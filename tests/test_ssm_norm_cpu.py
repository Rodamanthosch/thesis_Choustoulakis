"""
tests/test_ssm_norm_cpu.py
==========================
CPU-only verification of `ssm_norm` -- Jamba's RMSNorm on the input-dependent
SSM parameters (Lieber et al., arXiv 2403.19887, Sec. 6.4; transformers'
JambaMambaMixer dt_layernorm / b_layernorm / c_layernorm), for
  vmamba        : "none" | "dt" | "bcdt"  (the run: bcdt at d_state 16)
  spatial_mamba : "none" | "dt"           (the run: dt at d_state 1)

Runs WITHOUT a GPU and WITHOUT mamba_ssm: the selective scan is stubbed with a
reference recurrence that keeps float64 (so the transcriptions below compare
at ~1e-12) and records the dtypes it is called with.

Checks:
  A. ssm_norm="none" (the default) is BIT-IDENTICAL to HEAD for both models
     and every arm / variant the flag sits next to (same init at the same
     seed, and the same output after a random perturbation of every weight),
     and src/primitives.py only ADDS lines vs HEAD.
  B. ssm_rms_norm equals a verbatim copy of transformers' JambaRMSNorm
     (fp32 and bf16 max|diff| == 0), and with an fp32 weight on a bf16 input
     it returns bf16 (the CUDA scan needs B, C, dt in u's dtype).
  C. The SS2D mixer with "bcdt" / "dt" equals a naive float64 transcription
     of VMamba's SS2D with Jamba's norms inserted after each direction's
     x_proj split (dt_r before dt_proj), random gains, max|diff| < 1e-10.
  D. The Spatial-Mamba mixer (no gate, lpu zero, d_state 1) with "dt" equals
     a naive float64 transcription (state recurrence, SASF, y = C h + D u).
  E. What the norm does: scaling x_proj's dt / B / C rows by 5 leaves the
     "bcdt" output unchanged (scale invariance -- the property QK-norm buys)
     and moves the "none" output; for Spatial "dt" the same with the dt rows.
  F. Parameters: exactly {dt,B,C}_norm_weight (vmamba) / dt_norm_weight
     (spatial) are added, 1-D, ones-init, every baseline weight bit-identical
     at the same seed, +K*(R+2N) (+R) per block; all in add_weight_decay's
     no-decay group; gradients reach them.
  G. Under bf16 autocast the scan receives exactly the dtypes the "none"
     model passes (no new dtype reaches the kernel).
  H. Guards: bcdt at d_state 1, bcdt on spatial, ssm_norm with SSC or
     state-init, and unknown names are rejected.
  I. scripts/run_experiment.py::build_model == scripts/evaluate.py::build_model
     on both new configs, which differ from their base configs ONLY by
     ssm_norm; param counts 31,034,304 and 29,686,404.

Run from the repo root:
    PYTHONPATH=. python tests/test_ssm_norm_cpu.py
"""
import ast
import difflib
import math
import os
import subprocess
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import yaml
except ImportError as e:                                    # pragma: no cover
    print(f"SKIP (torch / yaml not importable here): {e}")
    sys.exit(0)


# ── stub mamba_ssm: Mamba-1 selective scan as a recurrence (keeps float64) ───
SCAN_DTYPES = []          # (u, delta, B, C) dtypes of every call, for check G


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, g, n, l) grouped."""
    SCAN_DTYPES.append((u.dtype, delta.dtype, B.dtype, C.dtype))
    wt = torch.promote_types(u.dtype, torch.float32)
    b, d, l = u.shape
    u, delta, A = u.to(wt), delta.to(wt), A.to(wt)
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(wt)
    if delta_softplus:
        delta = F.softplus(delta)
    rep = d // B.shape[1]
    Bf = B.to(wt).repeat_interleave(rep, dim=1)      # (b, d, n, l)
    Cf = C.to(wt).repeat_interleave(rep, dim=1)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for i in range(l):
        h = torch.exp(delta[:, :, i, None] * A[None]) * h \
            + delta[:, :, i, None] * Bf[:, :, :, i] * u[:, :, i, None]
        ys.append((h * Cf[:, :, :, i]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D[None, :, None].to(wt) * u
    return y


if "mamba_ssm" not in sys.modules:
    for name in ("mamba_ssm", "mamba_ssm.ops"):
        sys.modules[name] = types.ModuleType(name)
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = selective_scan_ref
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = iface

from src.primitives import ssm_rms_norm, SSM_NORMS           # noqa: E402  (after the stub)
from src.models.vmamba import JiTVMamba, SS2D                # noqa: E402
from src.models.spatial_mamba import (                       # noqa: E402
    JiTSpatialMamba, StructureAwareSSM)
from src.utils import add_weight_decay                       # noqa: E402

KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32)
D, DEPTH = KW["hidden_size"], KW["depth"]
R = math.ceil(D / 16)                     # dt_rank
B = 3
torch.manual_seed(0)
x_img = torch.randn(B, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(B)
y_in = torch.randint(0, KW["num_classes"], (B,))

VM = dict(d_state=4)
SP = dict(gate=False, lpu="zero", sasf_dilations=(1, 2, 3))


def build(cls, **kw):
    torch.manual_seed(0)
    return cls(**KW, **kw).eval()


def perturb(model, seed=1):
    """Seeded noise on every parameter, so adaLN-Zero's zero output can't hide
    a difference."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def head_source(rel):
    try:
        return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT,
                              capture_output=True, check=True).stdout.decode("utf-8")
    except Exception as e:                                  # pragma: no cover
        print(f"   (cannot read HEAD:{rel}: {e})")
        return None


# ═══ A. ssm_norm="none" is bit-identical to HEAD ═════════════════════════════
prim_head = head_source("src/primitives.py")
if prim_head is not None:
    cur = open(os.path.join(REPO_ROOT, "src", "primitives.py"), encoding="utf-8").read()
    removed = [l for l in difflib.ndiff(prim_head.splitlines(), cur.splitlines())
               if l.startswith("- ")]
    assert not removed, f"src/primitives.py changed existing lines: {removed[:3]}"

CASES = [
    ("src/models/vmamba.py", "JiTVMamba", JiTVMamba, "baseline", dict(VM)),
    ("src/models/vmamba.py", "JiTVMamba", JiTVMamba, "in-context class K=2",
     dict(VM, in_context_len=2, in_context_content="class")),
    ("src/models/vmamba.py", "JiTVMamba", JiTVMamba, "in-context row",
     dict(VM, in_context_len=4, in_context_content="class", in_context_layout="row")),
    ("src/models/vmamba.py", "JiTVMamba", JiTVMamba, "ssc abc", dict(VM, ssc="abc")),
    ("src/models/vmamba.py", "JiTVMamba", JiTVMamba, "state-init learned",
     dict(VM, state_init="learned")),
    ("src/models/vmamba.py", "JiTVMamba", JiTVMamba, "d_state 1 + derf",
     dict(d_state=1, norm="derf")),
    ("src/models/spatial_mamba.py", "JiTSpatialMamba", JiTSpatialMamba, "no gate",
     dict(SP)),
    ("src/models/spatial_mamba.py", "JiTSpatialMamba", JiTSpatialMamba, "gate, lpu none",
     dict(gate=True, lpu="none", sasf_dilations=(1, 2, 3))),
    ("src/models/spatial_mamba.py", "JiTSpatialMamba", JiTSpatialMamba, "d_state 4 per_state",
     dict(SP, d_state=4, sasf_state_kernels="per_state")),
    ("src/models/spatial_mamba.py", "JiTSpatialMamba", JiTSpatialMamba, "in-context 2 tokens",
     dict(SP, in_context_len=2, in_context_start=1, in_context_positions=[[0, 1], [2, 2]])),
]
n_checked = 0
for rel, name, cls, label, kw in CASES:
    src = head_source(rel)
    if src is None:
        print(f"A  SKIPPED for {rel}")
        continue
    head_mod = types.ModuleType(f"head_{name}")
    exec(compile(src, f"<HEAD:{rel}>", "exec"), head_mod.__dict__)
    old = build(getattr(head_mod, name), **kw)
    new = build(cls, **kw)                                  # ssm_norm defaults to none
    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn), f"state_dict keys changed for {name} {label}"
    assert all(torch.equal(so[k], sn[k]) for k in so), f"init changed for {name} {label}"
    perturb(old)
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.no_grad():
        out_old = old(x_img, t_in, y_in)
        d = (out_old - new(x_img, t_in, y_in)).abs().max().item()
    assert out_old.abs().max() > 0, f"perturbed output still zero for {name} {label}"
    assert d == 0.0, f"ssm_norm='none' not bit-identical to HEAD for {name} {label}: {d}"
    n_checked += 1
print(f"A  ssm_norm='none' bit-identical to HEAD for {n_checked} model/arm builds "
      f"(init + perturbed output, max|diff| = 0); primitives.py only adds lines")


# ═══ B. ssm_rms_norm == transformers' JambaRMSNorm ═══════════════════════════
class JambaRMSNorm(nn.Module):
    """Verbatim from transformers/models/jamba/modeling_jamba.py."""
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        JambaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


torch.manual_seed(2)
ref = JambaRMSNorm(16)
with torch.no_grad():
    ref.weight.normal_()
xb = 3 * torch.randn(5, 7, 16)
with torch.no_grad():
    d32 = (ssm_rms_norm(xb, ref.weight, dim=-1) - ref(xb)).abs().max().item()
    # Jamba's layout (B, L, n) normalised over the last dim == ours over dim 1 of (B, n, L)
    dT = (ssm_rms_norm(xb.transpose(1, 2), ref.weight[None, :, None], dim=1).transpose(1, 2)
          - ref(xb)).abs().max().item()
    ref16 = JambaRMSNorm(16).bfloat16()
    ref16.weight.data.copy_(ref.weight.data)
    x16 = xb.bfloat16()
    o16 = ssm_rms_norm(x16, ref16.weight, dim=-1)
    d16 = (o16.float() - ref16(x16).float()).abs().max().item()
    mixed = ssm_rms_norm(x16, ref.weight, dim=-1)          # fp32 weight, bf16 input
    dmix = (mixed.float() - ref(x16).bfloat16().float()).abs().max().item()
assert d32 == 0.0 and dT == 0.0, (d32, dT)
assert o16.dtype == torch.bfloat16 and d16 == 0.0, d16
assert mixed.dtype == torch.bfloat16 and dmix == 0.0, dmix
print(f"B  ssm_rms_norm == JambaRMSNorm: fp32 max|diff| = {d32} (also over dim 1 of "
      f"(B, n, L)), bf16 = {d16}; fp32 weight on bf16 input -> bf16, Jamba's result "
      f"rounded once (max|diff| = {dmix})")


# ═══ C. SS2D mixer == naive float64 transcription ════════════════════════════
def jrms(v, g, eps=1e-6):
    """JambaRMSNorm over dim 1 of (B, n, L), written out."""
    return g[None, :, None] * v / torch.sqrt(v.pow(2).mean(1, keepdim=True) + eps)


def naive_ss2d(m, x, H, W):
    """VMamba SS2D (4 directions) with Jamba's norms after each x_proj split."""
    Bn, L, _ = x.shape
    K, d, N, Rk = m.K, m.d_inner, m.d_state, m.dt_rank
    z = x @ m.in_proj.weight.t()
    z = z.view(Bn, H, W, d).permute(0, 3, 1, 2)
    z = F.silu(F.conv2d(z, m.conv2d.weight, m.conv2d.bias, padding=1, groups=d))
    row = z.reshape(Bn, d, L)                               # row-major
    col = z.transpose(2, 3).reshape(Bn, d, L)               # column-major
    seqs = [row, row.flip(-1), col, col.flip(-1)]
    out = 0
    for k in range(K):
        u = seqs[k]
        xd = torch.einsum("od,bdl->bol", m.x_proj_weight[k], u)
        dt_r, Bs, Cs = xd[:, :Rk], xd[:, Rk:Rk + N], xd[:, Rk + N:]
        if m.ssm_norm != "none":
            dt_r = jrms(dt_r, m.dt_norm_weight[k * Rk:(k + 1) * Rk])
        if m.ssm_norm == "bcdt":
            Bs = jrms(Bs, m.B_norm_weight[k * N:(k + 1) * N])
            Cs = jrms(Cs, m.C_norm_weight[k * N:(k + 1) * N])
        # SS2D hands A, D and the dt bias to the kernel in fp32 by design
        # (".float()"), so the transcription rounds them the same way.
        f32 = lambda p: p.float().to(u.dtype)               # noqa: E731
        delta = F.softplus(torch.einsum("dr,brl->bdl", m.dt_projs_weight[k], dt_r)
                           + f32(m.dt_projs_bias[k])[None, :, None])
        A = -f32(torch.exp(m.A_logs[k].float()))            # (d, N)
        Dk = f32(m.Ds[k])
        h = u.new_zeros(Bn, d, N)
        ys = []
        for t in range(L):
            h = torch.exp(delta[:, :, t, None] * A) * h \
                + delta[:, :, t, None] * Bs[:, None, :, t] * u[:, :, t, None]
            ys.append((h * Cs[:, None, :, t]).sum(-1) + Dk[None] * u[:, :, t])
        y = torch.stack(ys, -1)                              # scan order of direction k
        if k in (1, 3):
            y = y.flip(-1)
        if k in (2, 3):                                      # column-major -> row-major
            y = y.reshape(Bn, d, W, H).transpose(2, 3).reshape(Bn, d, L)
        out = out + y
    out = F.layer_norm(out.transpose(1, 2), (d,), m.out_norm.weight, m.out_norm.bias,
                       m.out_norm.eps)
    return out @ m.out_proj.weight.t()


Hg = Wg = 4
for mode in ("bcdt", "dt"):
    torch.manual_seed(3)
    mix = perturb(SS2D(d_model=D, d_state=4, ssm_norm=mode), seed=4).double().eval()
    xm = torch.randn(2, Hg * Wg, D, dtype=torch.float64)
    with torch.no_grad():
        dC = (mix(xm, Hg, Wg) - naive_ss2d(mix, xm, Hg, Wg)).abs().max().item()
    assert dC < 1e-10, (mode, dC)
    print(f"C  SS2D ssm_norm={mode!r} == naive float64 transcription "
          f"(4 directions, random gains): max|diff| = {dC:.1e}")

# A normalised and an un-normalised transcription must disagree (else C is vacuous).
torch.manual_seed(3)
mix = perturb(SS2D(d_model=D, d_state=4, ssm_norm="bcdt"), seed=4).double().eval()
with torch.no_grad():
    a = naive_ss2d(mix, xm, Hg, Wg)
    mix.ssm_norm = "none"
    b = naive_ss2d(mix, xm, Hg, Wg)
assert (a - b).abs().max() > 1e-3
print(f"C  (non-vacuous: with vs without the norms the transcriptions differ by "
      f"{(a - b).abs().max().item():.2e})")


# ═══ D. Spatial-Mamba mixer == naive float64 transcription ═══════════════════
def naive_spatial(m, x, H, W):
    """Structure-aware SSM, d_state 1, no gate, Jamba's dt norm before dt_proj."""
    Bn, L, _ = x.shape
    d, Rk = m.d_inner, m.dt_rank
    u = x @ m.in_proj.weight.t()
    u = u.view(Bn, H, W, d).permute(0, 3, 1, 2)
    u = F.silu(F.conv2d(u, m.conv2d.weight, m.conv2d.bias, padding=1, groups=d))
    u = u.reshape(Bn, d, L)
    xd = torch.einsum("od,bdl->bol", m.x_proj_weight, u)
    dt_r, Bs, Cs = xd[:, :Rk], xd[:, Rk:Rk + 1], xd[:, Rk + 1:]
    if m.ssm_norm == "dt":
        dt_r = jrms(dt_r, m.dt_norm_weight)
    # the module hands A and the dt bias to the kernel in fp32 (".float()")
    delta = F.softplus(torch.einsum("dr,brl->bdl", m.dt_projs_weight, dt_r)
                       + m.dt_projs_bias.float().to(u.dtype)[None, :, None])
    A = -torch.exp(m.A_logs[:, 0].float()).to(u.dtype)      # (d,)
    h = u.new_zeros(Bn, d)
    states = []
    for t in range(L):
        h = torch.exp(delta[:, :, t] * A) * h + delta[:, :, t] * Bs[:, :, t] * u[:, :, t]
        states.append(h)
    states = torch.stack(states, -1)                        # (Bn, d, L)
    fused = m.state_fusion(states.view(Bn, d, H, W)).reshape(Bn, d, L)
    y = fused * Cs + u * m.Ds[None, :, None]
    y = F.layer_norm(y.transpose(1, 2), (d,), m.out_norm.weight, m.out_norm.bias,
                     m.out_norm.eps)
    return y @ m.out_proj.weight.t()


torch.manual_seed(5)
smix = perturb(StructureAwareSSM(d_model=D, d_state=1, gate=False, dilations=(1, 2, 3),
                                 ssm_norm="dt"), seed=6).double().eval()
xs_ = torch.randn(2, Hg * Wg, D, dtype=torch.float64)
with torch.no_grad():
    dD = (smix(xs_, Hg, Wg) - naive_spatial(smix, xs_, Hg, Wg)).abs().max().item()
    a = naive_spatial(smix, xs_, Hg, Wg)
    smix.ssm_norm = "none"
    b = naive_spatial(smix, xs_, Hg, Wg)
    smix.ssm_norm = "dt"
assert dD < 1e-10, dD
assert (a - b).abs().max() > 1e-3
print(f"D  Spatial ssm_norm='dt' == naive float64 transcription (d_state 1, SASF): "
      f"max|diff| = {dD:.1e} (vs no norm: {(a - b).abs().max().item():.2e})")


# ═══ E. scale invariance -- what the norm buys ═══════════════════════════════
def scaled_diff(model, rows, c=5.0):
    """max|out(x_proj rows * c) - out| / max|out| for the given row slices of
    every block's x_proj."""
    with torch.no_grad():
        base_out = model(x_img, t_in, y_in)
        saved = [blk.mixer.x_proj_weight.clone() for blk in model.blocks]
        for blk in model.blocks:
            w = blk.mixer.x_proj_weight
            for sl in rows:
                w[..., sl, :] *= c
        out = model(x_img, t_in, y_in)
        for blk, s in zip(model.blocks, saved):
            blk.mixer.x_proj_weight.copy_(s)
    return ((out - base_out).abs().max() / base_out.abs().max()).item()


N4 = VM["d_state"]
vm_rows = [slice(0, R), slice(R, R + N4), slice(R + N4, R + 2 * N4)]
on = perturb(build(JiTVMamba, ssm_norm="bcdt", **VM))
off = perturb(build(JiTVMamba, **VM))
e_on, e_off = scaled_diff(on, vm_rows), scaled_diff(off, vm_rows)
assert e_on < 1e-4 and e_off > 1e-2, (e_on, e_off)
sp_rows = [slice(0, R)]
son = perturb(build(JiTSpatialMamba, ssm_norm="dt", **SP))
soff = perturb(build(JiTSpatialMamba, **SP))
s_on, s_off = scaled_diff(son, sp_rows), scaled_diff(soff, sp_rows)
assert s_on < 1e-4 and s_off > 1e-2, (s_on, s_off)
print(f"E  x_proj dt/B/C rows x5: vmamba bcdt output moves {e_on:.1e} (eps only) vs "
      f"none {e_off:.2e}; spatial dt rows x5: dt {s_on:.1e} vs none {s_off:.2e}")


# ═══ F. parameters, weight decay, gradients ══════════════════════════════════
for cls, kw, mode, per_block in [
        (JiTVMamba, VM, "bcdt", 4 * (R + 2 * VM["d_state"])),
        (JiTVMamba, VM, "dt", 4 * R),
        (JiTSpatialMamba, SP, "dt", R)]:
    base, normd = build(cls, **kw), build(cls, ssm_norm=mode, **kw)
    sb, sd = base.state_dict(), normd.state_dict()
    assert all(torch.equal(sb[k], sd[k]) for k in sb), f"{cls.__name__} {mode}: a weight moved"
    names = ["dt_norm_weight"] + (["B_norm_weight", "C_norm_weight"] if mode == "bcdt" else [])
    assert set(sd) - set(sb) == {f"blocks.{i}.mixer.{n}" for i in range(DEPTH) for n in names}
    assert n_params(normd) - n_params(base) == DEPTH * per_block
    gains = [p for n, p in normd.named_parameters() if n.endswith("_norm_weight")]
    assert all(p.dim() == 1 and torch.equal(p, torch.ones_like(p)) for p in gains)
    no_decay, decay = add_weight_decay(normd, weight_decay=0.05)
    ids = {id(p) for p in gains}
    assert ids <= {id(p) for p in no_decay["params"]}
    assert not ids & {id(p) for p in decay["params"]}
    m = perturb(normd).train()
    m(x_img, t_in, y_in).pow(2).mean().backward()
    for n, p in m.named_parameters():
        if n.endswith("_norm_weight") or n.endswith("x_proj_weight"):
            assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad on {n}"
    print(f"F  {cls.__name__} {mode}: +{DEPTH} x {per_block} params ({', '.join(names)}), "
          f"1-D ones-init, no weight decay, gradients reach them; other weights unchanged")


# ═══ G. bf16 autocast: the scan sees the same dtypes as without the norm ═════
def scan_dtypes(model):
    SCAN_DTYPES.clear()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(x_img, t_in, y_in)
    return list(SCAN_DTYPES)


for cls, kw, mode in [(JiTVMamba, VM, "bcdt"), (JiTSpatialMamba, SP, "dt")]:
    a = scan_dtypes(perturb(build(cls, **kw)))
    b = scan_dtypes(perturb(build(cls, ssm_norm=mode, **kw)))
    assert a == b and a, (cls.__name__, a, b)
    print(f"G  {cls.__name__} {mode} under bf16 autocast: scan dtypes (u, dt, B, C) "
          f"= {tuple(str(t).split('.')[-1] for t in b[0])}, same as without the norm")


# ═══ H. guards ═══════════════════════════════════════════════════════════════
def rejected(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


assert rejected(lambda: build(JiTVMamba, d_state=1, ssm_norm="bcdt"))
assert rejected(lambda: build(JiTSpatialMamba, ssm_norm="bcdt", **SP))
assert rejected(lambda: build(JiTVMamba, ssm_norm="bcdt", ssc="bc", **VM))
assert rejected(lambda: build(JiTVMamba, ssm_norm="dt", state_init="dimsum", **VM))
assert rejected(lambda: build(JiTVMamba, ssm_norm="qk", **VM))
assert not rejected(lambda: build(JiTVMamba, d_state=1, ssm_norm="dt"))
assert SSM_NORMS == ("none", "dt", "bcdt")
print("H  rejected: bcdt at d_state 1, bcdt on spatial_mamba, with SSC, with "
      "state-init, unknown name; dt at d_state 1 accepted")


# ═══ I. build_model wiring (train == eval) and the configs ═══════════════════
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


def same_build(cfg):
    torch.manual_seed(0)
    mt = bm_train(cfg)
    torch.manual_seed(0)
    me = bm_eval(cfg)
    st, se = mt.state_dict(), me.state_dict()
    assert set(st) == set(se) and all(torch.equal(st[k], se[k]) for k in st)
    return mt


def load_cfg(name):
    return yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "tiny_imagenet", name),
                               encoding="utf-8"))


for new_name, base_name, mode, expect in [
        ("jit-s2-vmamba-ssmnorm-bcdt.yaml", "jit-s2-vmamba-baseline.yaml", "bcdt", 31_034_304),
        ("jit-s2-spatial-nogate-ssmnorm-dt.yaml", "jit-s2-spatial-nogate.yaml", "dt", 29_686_404)]:
    cfg, base_cfg = load_cfg(new_name), load_cfg(base_name)
    assert cfg["model"]["ssm_norm"] == mode
    assert {k: v for k, v in cfg["model"].items() if k != "ssm_norm"} == base_cfg["model"], \
        f"{new_name} differs from {base_name} beyond ssm_norm"
    for sec in ("training", "diffusion", "cfg", "sampling"):
        assert cfg[sec] == base_cfg[sec], (new_name, sec)
    assert cfg["experiment"]["seed"] == base_cfg["experiment"]["seed"]
    mt = same_build(cfg)
    mb = same_build(base_cfg)
    assert mt.ssm_norm == mode and all(b.mixer.ssm_norm == mode for b in mt.blocks)
    n_new, n_base = n_params(mt), n_params(mb)
    assert n_new == expect, (new_name, n_new)
    print(f"I  {new_name}: train == eval build_model, ssm_norm={mode!r}, "
          f"{n_new:,} params (base {n_base:,} + {n_new - n_base:,}); only ssm_norm differs")

print("ALL CHECKS PASSED")
