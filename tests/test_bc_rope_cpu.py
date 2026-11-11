"""
tests/test_bc_rope_cpu.py
=========================
CPU-only verification of `a_tie` (A shared by each state pair (2m, 2m+1)) and
`bc_rope` (RoPE on the scan's key B and query C) for JiT-S2-VMamba (SS2D).

The scan computes  y_i = sum_j C_i^T (prod_k exp(dt_k A)) B_j dt_j x_j.  With B
and C rotated in d_state pairs by their cell's 2-D position,
    (R(phi_i) C_i)^T  decay  R(phi_j) B_j  =  C_i^T  decay  R(phi_j - phi_i) B_j
when the decay is the same for both states of every rotated pair (a_tie).

Runs WITHOUT a GPU and WITHOUT mamba_ssm: the selective scan is stubbed with a
reference recurrence that keeps float64 and records the dtypes and A it gets.

Checks:
  A. a_tie="none", bc_rope="none" (the defaults) are BIT-IDENTICAL to the
     reference commit (the last one before the feature; override with
     BC_ROPE_REFERENCE=<rev>) for JiTVMamba under every arm / modifier the
     flags sit next to, and for JiTVMambaSASF (which inherits SS2D.__init__):
     same init at the same seed and the same output after a random
     perturbation of every weight.
  B. bc_rope_tables == an independent float64 derivation on a rectangular grid
     (omega_j = theta^(-4j/N); pair j < N/4 by row, the rest by column; the
     cell visited at step s of each path written out from SS2D's scan orders,
     not from cross_scan); apply_bc_rope == explicit 2x2 rotation matrices,
     bf16 in -> bf16 out; a table first built under inference_mode still works
     with autograd.
  C. The SS2D mixer (a_tie pairs; with and without bc_rope; and with ssm_norm
     bcdt) equals a naive float64 transcription of the 4-direction scan with A
     tied and B, C multiplied by each cell's rotation matrix, < 1e-10.
  D. What the rotation buys: shifting every coordinate by a constant (rows +3,
     columns +7, tables patched into the real mixer) leaves the output
     unchanged (relative) with A tied, and moves it with A free per state; and
     the rotation itself changes the output (same weights, bc_rope off vs on).
  E. Parameters: a_tie changes ONLY A_logs, to (K, d_inner, N/2) == the
     baseline init's even states, every other weight bit-identical at the same
     seed, -depth*K*d_inner*N/2 params; bc_rope adds nothing (state_dict
     bit-identical to the a_tie model); the A reaching the scan is tied in
     pairs; A_logs is in the baseline's weight-decay group; gradients reach
     A_logs and x_proj.
  F. Under bf16 autocast the scan receives exactly the dtypes the baseline
     passes.
  G. Guards: bc_rope without a_tie, d_state not divisible by 4, with in-context
     tokens / state-init / SSC, odd d_state with a_tie, unknown names, bad
     theta are rejected; bc_rope + ssm_norm bcdt / feature_rope and a_tie with
     the in-context row layout are accepted and run.
  H. scripts/run_experiment.py, evaluate.py and profile_model.py build the same
     model from both new configs, which differ from the baseline config only
     by their new keys; 30,884,160 params each.

Run from the repo root:
    PYTHONPATH=. python tests/test_bc_rope_cpu.py
"""
import ast
import math
import os
import subprocess
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import torch
    import torch.nn.functional as F
    import yaml
except ImportError as e:                                    # pragma: no cover
    print(f"SKIP (torch / yaml not importable here): {e}")
    sys.exit(0)


# ── stub mamba_ssm: Mamba-1 selective scan as a recurrence (keeps float64) ───
SCAN_CALLS = []           # (u, delta, B, C) dtypes and A of every call


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, g, n, l) grouped."""
    SCAN_CALLS.append(((u.dtype, delta.dtype, B.dtype, C.dtype), A.detach().clone()))
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

import src.models.vmamba as vmamba_mod                       # noqa: E402  (after the stub)
from src.models.vmamba import (                              # noqa: E402
    JiTVMamba, SS2D, bc_rope_tables, apply_bc_rope, BC_ROPE_MODES, A_TIE_MODES)
from src.models.vmamba_sasf import JiTVMambaSASF             # noqa: E402
from src.utils import add_weight_decay                       # noqa: E402

REFERENCE = os.environ.get("BC_ROPE_REFERENCE", "118940d")
KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32)
D, DEPTH, K = KW["hidden_size"], KW["depth"], 4
N = 8                                    # d_state: 2 frequencies per axis
VM = dict(d_state=N)
ON = dict(VM, a_tie="pairs", bc_rope="2d_axial", bc_rope_theta=10.0)
B = 3
torch.manual_seed(0)
x_img = torch.randn(B, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(B)
y_in = torch.randint(0, KW["num_classes"], (B,))


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


def ref_source(rel):
    try:
        return subprocess.run(["git", "show", f"{REFERENCE}:{rel}"], cwd=REPO_ROOT,
                              capture_output=True, check=True).stdout.decode("utf-8")
    except Exception as e:                                  # pragma: no cover
        print(f"   (cannot read {REFERENCE}:{rel}: {e})")
        return None


# ═══ A. the defaults are bit-identical to the reference commit ═══════════════
def ref_module(rel, name, overrides=None):
    src = ref_source(rel)
    if src is None:
        return None
    mod = types.ModuleType(name)
    saved = {k: sys.modules.get(k) for k in (overrides or {})}
    sys.modules.update(overrides or {})
    try:
        exec(compile(src, f"<{REFERENCE}:{rel}>", "exec"), mod.__dict__)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def same_as_ref(old_cls, new_cls, kw, label):
    old, new = build(old_cls, **kw), build(new_cls, **kw)
    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn), f"state_dict keys changed: {label}"
    assert all(torch.equal(so[k], sn[k]) for k in so), f"init changed: {label}"
    perturb(old)
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.no_grad():
        out_old = old(x_img, t_in, y_in)
        d = (out_old - new(x_img, t_in, y_in)).abs().max().item()
    assert out_old.abs().max() > 0, f"perturbed output still zero: {label}"
    assert d == 0.0, f"defaults not bit-identical to {REFERENCE}: {label}: {d}"


ref_vm = ref_module("src/models/vmamba.py", "ref_vmamba")
CASES = [
    ("baseline", dict(VM)),
    ("d_state 16", dict(d_state=16)),
    ("in-context prefix class K=2", dict(VM, in_context_len=2, in_context_content="class")),
    ("in-context row", dict(VM, in_context_len=4, in_context_content="class",
                            in_context_layout="row")),
    ("in-context diagonal", dict(VM, in_context_len=2, in_context_content="class",
                                 in_context_layout="diagonal", in_context_start=1)),
    ("ssc abc", dict(VM, ssc="abc")),
    ("ssc bc, adaln_cond none, z mlp", dict(VM, ssc="bc", adaln_cond="none", ssc_z_mlp=True)),
    ("state-init learned", dict(VM, state_init="learned")),
    ("ssm_norm bcdt", dict(VM, ssm_norm="bcdt")),
    ("feature_rope 2d_axial", dict(VM, feature_rope="2d_axial")),
    ("glumbconv + derf", dict(VM, ffn="glumbconv", norm="derf")),
    ("d_state 1", dict(d_state=1)),
]
n_checked = 0
if ref_vm is None:
    print("A  SKIPPED (no reference commit in this clone)")
else:
    for label, kw in CASES:
        same_as_ref(ref_vm.JiTVMamba, JiTVMamba, kw, f"JiTVMamba {label}")
        n_checked += 1
    # JiTVMambaSASF at the reference, built on the reference vmamba.py
    ref_sasf = ref_module("src/models/vmamba_sasf.py", "ref_vmamba_sasf",
                          {"src.models.vmamba": ref_vm})
    if ref_sasf is not None:
        same_as_ref(ref_sasf.JiTVMambaSASF, JiTVMambaSASF,
                    dict(d_state=1, sasf_dilations=(1, 2)), "JiTVMambaSASF")
        n_checked += 1
    print(f"A  a_tie / bc_rope defaults bit-identical to {REFERENCE} for {n_checked} "
          f"model/arm builds (init + perturbed output, max|diff| = 0)")


# ═══ B. the tables and the rotation ══════════════════════════════════════════
def scan_cells(H, W):
    """The cell (r, c) visited at each step of SS2D's four paths, written out:
    0 row-major, 1 its reverse, 2 column-major, 3 its reverse."""
    row = [(r, c) for r in range(H) for c in range(W)]
    col = [(r, c) for c in range(W) for r in range(H)]
    return [row, row[::-1], col, col[::-1]]


def ref_angles(theta, n, H, W, dr=0, dc=0):
    """(4, n/2, H*W) float64 angles, from the formula; coordinates shifted by (dr, dc)."""
    omega = [theta ** (-4.0 * j / n) for j in range(n // 4)]
    out = torch.zeros(4, n // 2, H * W, dtype=torch.float64)
    for k, cells in enumerate(scan_cells(H, W)):
        for s, (r, c) in enumerate(cells):
            for j, w in enumerate(omega):
                out[k, j, s] = (r + dr) * w
                out[k, n // 4 + j, s] = (c + dc) * w
    return out


Hr, Wr, theta = 3, 5, 7.0
cos, sin = bc_rope_tables(theta, 12, Hr, Wr, torch.device("cpu"), torch.float64)
ang = ref_angles(theta, 12, Hr, Wr)
dB = max((cos[0] - ang.cos()).abs().max().item(), (sin[0] - ang.sin()).abs().max().item())
assert cos.shape == (1, 4, 6, Hr * Wr) and dB < 1e-12, (cos.shape, dB)


def rot_matrix(angles):
    """Block-diagonal (n, n) rotation for one cell, from its n/2 pair angles."""
    n = 2 * angles.numel()
    Rm = torch.zeros(n, n, dtype=torch.float64)
    for j, a in enumerate(angles.tolist()):
        Rm[2 * j:2 * j + 2, 2 * j:2 * j + 2] = torch.tensor(
            [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=torch.float64)
    return Rm


v = torch.randn(2, 4, 12, Hr * Wr, dtype=torch.float64)
rv = apply_bc_rope(v, cos, sin)
explicit = torch.stack([torch.stack([rot_matrix(ang[k, :, s]) @ v[:, k, :, s].T
                                     for s in range(Hr * Wr)], -1)
                        for k in range(4)], 0).permute(2, 0, 1, 3)   # (2, 4, 12, L)
dR = (rv - explicit).abs().max().item()
assert dR < 1e-12, dR
c32, s32 = bc_rope_tables(theta, 12, Hr, Wr, torch.device("cpu"), torch.float32)
v16 = v.bfloat16()
r16 = apply_bc_rope(v16, c32, s32)
assert r16.dtype == torch.bfloat16
d16 = ((r16.double() - apply_bc_rope(v16.double(), cos, sin)).abs()
       / (explicit.abs() + 1)).max().item()
assert d16 < 1e-2, d16
with torch.inference_mode():                                 # first build of this key
    ci, si = bc_rope_tables(theta, 12, 2, 7, torch.device("cpu"), torch.float32)
vg = torch.randn(1, 4, 12, 14, requires_grad=True)
apply_bc_rope(vg, ci, si).square().sum().backward()
assert torch.isfinite(vg.grad).all()
print(f"B  bc_rope_tables == independent float64 angles on a {Hr}x{Wr} grid (max|diff| "
      f"{dB:.1e}); apply_bc_rope == explicit 2x2 rotations ({dR:.1e}); bf16 in -> bf16 out; "
      f"autograd after inference_mode")


# ═══ C. SS2D mixer == naive float64 transcription ════════════════════════════
def jrms(v, g, eps=1e-6):
    return g[None, :, None] * v / torch.sqrt(v.pow(2).mean(1, keepdim=True) + eps)


def naive_ss2d(m, x, H, W, A_full=None, dr=0, dc=0):
    """VMamba SS2D (4 directions); A tied as stored; B and C multiplied by the
    rotation matrix of the cell each scan step visits (coordinates + (dr, dc))."""
    Bn, L, _ = x.shape
    Kk, d, n, Rk = m.K, m.d_inner, m.d_state, m.dt_rank
    z = x @ m.in_proj.weight.t()
    z = z.view(Bn, H, W, d).permute(0, 3, 1, 2)
    z = F.silu(F.conv2d(z, m.conv2d.weight, m.conv2d.bias, padding=1, groups=d))
    row = z.reshape(Bn, d, L)
    col = z.transpose(2, 3).reshape(Bn, d, L)
    seqs = [row, row.flip(-1), col, col.flip(-1)]
    if A_full is None:
        # SS2D hands A to the kernel in fp32 by design (".float()"), so the
        # transcription rounds it the same way.
        A_logs = m.A_logs.float()
        if m.a_tie == "pairs":
            A_logs = torch.stack([A_logs, A_logs], -1).flatten(-2)   # [a0, a0, a1, a1, ...]
        A_full = -torch.exp(A_logs).to(x.dtype)                      # (K, d, n)
    angles = ref_angles(m.bc_rope_theta, n, H, W, dr, dc) if m.bc_rope != "none" else None
    out = 0
    for k in range(Kk):
        u = seqs[k]
        xd = torch.einsum("od,bdl->bol", m.x_proj_weight[k], u)
        dt_r, Bs, Cs = xd[:, :Rk], xd[:, Rk:Rk + n], xd[:, Rk + n:]
        if m.ssm_norm != "none":
            dt_r = jrms(dt_r, m.dt_norm_weight[k * Rk:(k + 1) * Rk])
        if m.ssm_norm == "bcdt":
            Bs = jrms(Bs, m.B_norm_weight[k * n:(k + 1) * n])
            Cs = jrms(Cs, m.C_norm_weight[k * n:(k + 1) * n])
        if angles is not None:
            Rs = [rot_matrix(angles[k, :, s]) for s in range(L)]
            Bs = torch.stack([Bs[:, :, s] @ Rs[s].T for s in range(L)], -1)
            Cs = torch.stack([Cs[:, :, s] @ Rs[s].T for s in range(L)], -1)
        f32 = lambda p: p.float().to(u.dtype)               # noqa: E731
        delta = F.softplus(torch.einsum("dr,brl->bdl", m.dt_projs_weight[k], dt_r)
                           + f32(m.dt_projs_bias[k])[None, :, None])
        A = A_full[k]
        Dk = f32(m.Ds[k])
        h = u.new_zeros(Bn, d, n)
        ys = []
        for t in range(L):
            h = torch.exp(delta[:, :, t, None] * A) * h \
                + delta[:, :, t, None] * Bs[:, None, :, t] * u[:, :, t, None]
            ys.append((h * Cs[:, None, :, t]).sum(-1) + Dk[None] * u[:, :, t])
        y = torch.stack(ys, -1)
        if k in (1, 3):
            y = y.flip(-1)
        if k in (2, 3):
            y = y.reshape(Bn, d, W, H).transpose(2, 3).reshape(Bn, d, L)
        out = out + y
    out = F.layer_norm(out.transpose(1, 2), (d,), m.out_norm.weight, m.out_norm.bias,
                       m.out_norm.eps)
    return out @ m.out_proj.weight.t()


def mixer(seed=3, **kw):
    torch.manual_seed(seed)
    return perturb(SS2D(d_model=D, d_state=N, **kw), seed=seed + 1).double().eval()


Hg, Wg = 4, 5
torch.manual_seed(7)
xm = torch.randn(2, Hg * Wg, D, dtype=torch.float64)
for label, kw in [("a_tie", dict(a_tie="pairs")),
                  ("a_tie + bc_rope", dict(a_tie="pairs", bc_rope="2d_axial", bc_rope_theta=3.0)),
                  ("a_tie + bc_rope + ssm_norm bcdt",
                   dict(a_tie="pairs", bc_rope="2d_axial", bc_rope_theta=3.0, ssm_norm="bcdt"))]:
    mix = mixer(**kw)
    with torch.no_grad():
        dC = (mix(xm, Hg, Wg) - naive_ss2d(mix, xm, Hg, Wg)).abs().max().item()
    assert dC < 1e-10, (label, dC)
    print(f"C  SS2D {label} == naive float64 transcription ({Hg}x{Wg} grid, 4 directions): "
          f"max|diff| = {dC:.1e}")


# ═══ D. relative: a constant coordinate shift leaves the output unchanged ═════
def shifted_tables(dr, dc):
    def tables(theta_, n, H, W, device, dtype):
        a = ref_angles(theta_, n, H, W, dr, dc)[None].to(device=device, dtype=dtype)
        return a.cos(), a.sin()
    return tables


def run_shifted(m, dr, dc):
    real = vmamba_mod.bc_rope_tables
    vmamba_mod.bc_rope_tables = shifted_tables(dr, dc)
    try:
        with torch.no_grad():
            return m(xm, Hg, Wg)
    finally:
        vmamba_mod.bc_rope_tables = real


rope = mixer(a_tie="pairs", bc_rope="2d_axial", bc_rope_theta=3.0)
with torch.no_grad():
    y0 = rope(xm, Hg, Wg)
scale = y0.abs().max().item()
d_tied = (run_shifted(rope, 3, 7) - y0).abs().max().item() / scale
d_zero = (run_shifted(rope, 0, 0) - y0).abs().max().item() / scale
# Same mixer with A free per state (Mamba-1 as is): the shift moves the output.
free = mixer(a_tie="pairs", bc_rope="2d_axial", bc_rope_theta=3.0)
with torch.no_grad():
    free.A_logs = torch.nn.Parameter(
        torch.log(torch.rand(K, free.d_inner, N, dtype=torch.float64) * 8 + 0.5))
free.a_tie = "none"
with torch.no_grad():
    yf = free(xm, Hg, Wg)
    dnaive = (yf - naive_ss2d(free, xm, Hg, Wg)).abs().max().item()
d_free = (run_shifted(free, 3, 7) - yf).abs().max().item() / yf.abs().max().item()
# Same weights, rotation off: the rotation is not a no-op.
off = mixer(a_tie="pairs")
off.load_state_dict(rope.state_dict(), strict=True)
with torch.no_grad():
    d_off = (off(xm, Hg, Wg) - y0).abs().max().item() / scale
assert d_zero < 1e-12 and d_tied < 1e-12, (d_zero, d_tied)
assert dnaive < 1e-10 and d_free > 1e-3 and d_off > 1e-3, (dnaive, d_free, d_off)
print(f"D  coordinates shifted by (+3, +7): A tied max rel |dy| = {d_tied:.1e} (relative), "
      f"A free per state {d_free:.2e}; rotation off vs on, same weights: {d_off:.2e}")


# ═══ E. parameters, weight decay, gradients, the A the scan gets ═════════════
base, tied, roped = build(JiTVMamba, **VM), build(JiTVMamba, **VM, a_tie="pairs"), build(JiTVMamba, **ON)
sb, st, sr = base.state_dict(), tied.state_dict(), roped.state_dict()
assert list(sb) == list(st) == list(sr)
for k in sb:
    if k.endswith("mixer.A_logs"):
        assert st[k].shape == (K, D, N // 2) and torch.equal(st[k], sb[k][..., 0::2]), k
    else:
        assert torch.equal(sb[k], st[k]), f"a_tie moved {k}"
assert all(torch.equal(st[k], sr[k]) for k in st), "bc_rope changed a weight"
assert n_params(base) - n_params(tied) == DEPTH * K * D * N // 2
assert n_params(roped) == n_params(tied)
assert torch.allclose(torch.exp(st["blocks.0.mixer.A_logs"][0, 0]),
                      torch.arange(1, N, 2, dtype=torch.float32))
SCAN_CALLS.clear()
with torch.no_grad():
    roped(x_img, t_in, y_in)
A_seen = SCAN_CALLS[0][1]
assert A_seen.shape == (K * D, N) and torch.equal(A_seen[:, 0::2], A_seen[:, 1::2])
# add_weight_decay groups by ndim, not by _no_weight_decay: the 3-D A_logs is in
# the decay group for the baseline too (the configs use weight_decay 0.0). The
# tied A_logs must land in the same group as the baseline's.
def a_group(model):
    no_decay, decay = add_weight_decay(model, weight_decay=0.05)
    ids = {id(b.mixer.A_logs) for b in model.blocks}
    return ("no_decay" if ids <= {id(p) for p in no_decay["params"]} else
            "decay" if ids <= {id(p) for p in decay["params"]} else "split")


assert a_group(roped) == a_group(base) != "split", (a_group(roped), a_group(base))
m = perturb(build(JiTVMamba, **ON)).train()
m(x_img, t_in, y_in).pow(2).mean().backward()
for n_, p in m.named_parameters():
    if n_.endswith("A_logs") or n_.endswith("x_proj_weight"):
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad on {n_}"
print(f"E  a_tie: only A_logs changes, to (K, d_inner, N/2) = the baseline's even states "
      f"(A = -[1, 3, ..., {N - 1}]), -{DEPTH} x {K * D * N // 2} params; bc_rope adds none; the "
      f"scan's A is tied in pairs; A_logs in the baseline's weight-decay group "
      f"({a_group(base)}); gradients reach A_logs and x_proj")


# ═══ F. bf16 autocast: the scan sees the baseline's dtypes ═══════════════════
def scan_dtypes(model):
    SCAN_CALLS.clear()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(x_img, t_in, y_in)
    return [c[0] for c in SCAN_CALLS]


a = scan_dtypes(perturb(build(JiTVMamba, **VM)))
b = scan_dtypes(perturb(build(JiTVMamba, **ON)))
assert a == b and a, (a, b)
print(f"F  bf16 autocast: scan dtypes (u, dt, B, C) = "
      f"{tuple(str(t).split('.')[-1] for t in b[0])}, same as the baseline")


# ═══ G. guards ═══════════════════════════════════════════════════════════════
def rejected(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return True
    return False


assert rejected(lambda: build(JiTVMamba, **VM, bc_rope="2d_axial"))            # no a_tie
assert rejected(lambda: build(JiTVMamba, d_state=6, a_tie="pairs", bc_rope="2d_axial"))
assert rejected(lambda: build(JiTVMamba, d_state=5, a_tie="pairs"))
assert rejected(lambda: build(JiTVMamba, **ON, in_context_len=2, in_context_content="class"))
assert rejected(lambda: build(JiTVMamba, **ON, state_init="dimsum"))
assert rejected(lambda: build(JiTVMamba, **ON, ssc="bc"))
assert rejected(lambda: build(JiTVMamba, **dict(ON, bc_rope="1d")))
assert rejected(lambda: build(JiTVMamba, **dict(ON, a_tie="quads")))
for bad in (0.0, -1.0, float("inf"), "x", True):
    assert rejected(lambda: build(JiTVMamba, **dict(ON, bc_rope_theta=bad))), bad
assert rejected(lambda: SS2D(D, d_state=N, a_tie="pairs", bc_rope="2d_axial", K=2))
tok = SS2D(D, d_state=N, a_tie="pairs", bc_rope="2d_axial").eval()
assert rejected(lambda: tok(torch.randn(1, 2 + 16, D), 4, 4))      # tokens in the scan
for kw in (dict(ON, ssm_norm="bcdt"), dict(ON, feature_rope="2d_axial"),
           dict(VM, a_tie="pairs", in_context_len=4, in_context_content="class",
                in_context_layout="row"),
           dict(ON, bc_rope_theta="1e4")):
    with torch.no_grad():
        out = perturb(build(JiTVMamba, **kw))(x_img, t_in, y_in)
    assert torch.isfinite(out).all(), kw
assert BC_ROPE_MODES == ("none", "2d_axial") and A_TIE_MODES == ("none", "pairs")
print("G  rejected: bc_rope without a_tie, d_state 6, odd d_state with a_tie, with in-context "
      "/ state-init / SSC, unknown names, bad theta, K != 4, tokens reaching the mixer; "
      "accepted and finite: + ssm_norm bcdt, + feature_rope, a_tie + in-context row")


# ═══ H. build_model wiring (train == eval == profile) and the configs ════════
def load_build_model(rel):
    path = os.path.join(REPO_ROOT, rel)
    src = open(path, encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_model")
    ns = {}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns["build_model"]


builders = {rel: load_build_model(rel) for rel in
            ("scripts/run_experiment.py", "scripts/evaluate.py", "scripts/profile_model.py")}


def load_cfg(name):
    return yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "tiny_imagenet", name),
                               encoding="utf-8"))


base_cfg = load_cfg("jit-s2-vmamba-baseline.yaml")
for name, keys in [
        ("jit-s2-vmamba-atie.yaml", dict(a_tie="pairs")),
        ("jit-s2-vmamba-bc-rope-2d-axial.yaml",
         dict(a_tie="pairs", bc_rope="2d_axial", bc_rope_theta=10.0))]:
    cfg = load_cfg(name)
    assert {k: cfg["model"][k] for k in keys} == keys, name
    assert {k: v for k, v in cfg["model"].items() if k not in keys} == base_cfg["model"], \
        f"{name} differs from the baseline beyond {sorted(keys)}"
    for sec in ("training", "diffusion", "cfg", "sampling"):
        assert cfg[sec] == base_cfg[sec], (name, sec)
    assert cfg["experiment"]["seed"] == base_cfg["experiment"]["seed"]
    models = {}
    for rel, bm in builders.items():
        torch.manual_seed(0)
        models[rel] = bm(cfg)
    ref_sd = models["scripts/run_experiment.py"].state_dict()
    for rel, mdl in models.items():
        sd = mdl.state_dict()
        assert set(sd) == set(ref_sd) and all(torch.equal(sd[k], ref_sd[k]) for k in sd), rel
        assert mdl.a_tie == keys["a_tie"] and mdl.bc_rope == keys.get("bc_rope", "none"), rel
        assert all(b.mixer.bc_rope_theta == (10.0 if "bc_rope" in keys else None)
                   for b in mdl.blocks), rel
    n = n_params(models["scripts/evaluate.py"])
    assert n == 30_884_160, (name, n)
    print(f"H  {name}: run_experiment == evaluate == profile_model build_model, "
          f"{n:,} params; differs from the baseline config only by {sorted(keys)}")

print("ALL CHECKS PASSED")
