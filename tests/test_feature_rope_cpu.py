"""
tests/test_feature_rope_cpu.py
==============================
CPU-only verification of `feature_rope` (src/feature_rope.py), the feature-level
RoPE of the VMamba and Spatial-Mamba mixers: `none` | `1d` | `2d_axial`.

Runs WITHOUT a GPU and WITHOUT mamba_ssm: the selective scan is stubbed with a
reference recurrence, so this exercises the model code, not the CUDA kernel.

Checks:
  A. feature_rope="none" (and the flag being absent) is BIT-IDENTICAL to the
     pre-edit implementation: same init, RNG stream and state_dict; every
     mixer's output compared directly, and the full model after a perturbation
     of every weight (adaLN-Zero and the zero output layer cannot hide a
     difference). src/primitives.py is unchanged.
  B. apply_feature_rope against an independent loop-based reference on a
     rectangular grid: formula, pair ordering, row/column allocation, identity
     at the origin, norm preservation, JiT's VisionRotaryEmbeddingFast, dtypes
     under bf16 autocast, the cache, inference mode, and the input guards.
  C. Placement. Each pre-edit mixer with the rotation spliced in right after
     its SiLU equals the new mixer (bit-exact with the module's rotation, and
     within tolerance with the independent one). The tensors reaching the scan
     are the already-rotated cells at their expected scan positions in all four
     paths, dt/B/C come from them, condition tokens are not rotated, and token
     insertion does not move any image coordinate.
  D. Forward + backward of both active modes in VMamba d_state 16 and
     Spatial-Mamba d_state 1 (both gate settings, with and without tokens).
  E. No new parameters or checkpoint entries, strict loading both ways, the
     train / eval / profile builders, and the four Tiny-ImageNet configs, which
     differ from their sources only by the RoPE keys and run identifiers.

Pre-edit reference: the directory in $FEATURE_ROPE_REFERENCE_DIR when set (a
copy of the working tree taken before the feature was written), otherwise the
last commit before it, pinned below. Never `git show HEAD`, which would compare
the feature against itself once committed.

Run from the repo root:
    PYTHONPATH=. python tests/test_feature_rope_cpu.py
"""
import ast
import copy
import math
import os
import subprocess
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import yaml
except ImportError as e:                                    # pragma: no cover
    print(f"SKIP (torch / numpy / yaml not importable here): {e}")
    sys.exit(0)

PRE_FEATURE_COMMIT = "09b6efb"
REFERENCE_DIR = os.environ.get("FEATURE_ROPE_REFERENCE_DIR")


# ── stub mamba_ssm: Mamba-1 selective scan as a recurrence ───────────────────
SCAN_CALLS = []            # filled only while CAPTURING
SCAN_DTYPES = []           # (u, delta, B, C) dtypes of every call
CAPTURING = [False]


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, g, n, l) grouped."""
    SCAN_DTYPES.append((u.dtype, delta.dtype, B.dtype, C.dtype))
    if CAPTURING[0]:
        SCAN_CALLS.append({k: v.detach().clone()
                           for k, v in dict(u=u, delta=delta, B=B, C=C).items()})
    wt = torch.promote_types(u.dtype, torch.float32)
    b, d, l = u.shape
    u, delta, A = u.to(wt), delta.to(wt), A.to(wt)
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(wt)
    if delta_softplus:
        delta = F.softplus(delta)
    rep = d // B.shape[1]
    Bf = B.to(wt).repeat_interleave(rep, dim=1)
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

from src import feature_rope as fr                          # noqa: E402  (after the stub)
from src.feature_rope import apply_feature_rope, validate_feature_rope  # noqa: E402
from src.models.jit import VisionRotaryEmbeddingFast        # noqa: E402
from src.models.vmamba import (                             # noqa: E402
    JiTVMamba, SS2D, cross_scan, diagonal_scan_index)
from src.models.spatial_mamba import (                      # noqa: E402
    JiTSpatialMamba, StructureAwareSSM)
from src.utils import load_config                           # noqa: E402

MODES = ("1d", "2d_axial")
THETA = 10000.0
TOL = 1e-4                 # module (fp32 angles) vs the float64 reference

KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32)
D, GRID = KW["hidden_size"], KW["input_size"] // KW["patch_size"]
BATCH = 3
VM = dict(d_state=16)
SP = dict(d_state=1, lpu="zero", sasf_dilations=(1, 2))

torch.manual_seed(0)
x_img = torch.randn(BATCH, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(BATCH)
y_in = torch.randint(0, KW["num_classes"], (BATCH,))
cond_in = torch.randn(BATCH, D)


def build(cls, **kw):
    torch.manual_seed(0)
    return cls(**KW, **kw).eval()


def perturb(module, seed=1):
    """Seeded noise on every parameter, so zero-init gates can't hide a difference."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return module


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def reference_source(rel):
    if REFERENCE_DIR:
        with open(os.path.join(REFERENCE_DIR, rel), encoding="utf-8") as f:
            return f.read()
    return subprocess.run(
        ["git", "show", f"{PRE_FEATURE_COMMIT}:{rel}"], cwd=REPO_ROOT,
        capture_output=True, check=True).stdout.decode("utf-8")


def load_reference(rel, name):
    src = reference_source(rel)
    assert "feature_rope" not in src, f"reference {rel} already contains the feature"
    module = types.ModuleType(name)
    exec(compile(src, f"<pre-edit:{rel}>", "exec"), module.__dict__)
    return module


REF_VM = load_reference("src/models/vmamba.py", "reference_vmamba")
REF_SP = load_reference("src/models/spatial_mamba.py", "reference_spatial")
print("reference:", REFERENCE_DIR or f"git commit {PRE_FEATURE_COMMIT}")


# ── Independent reference: plain loops over cells and channel pairs, float64 ─
def ref_rotate_np(x, mode, theta):
    B, dim, H, W = x.shape
    out = np.empty(x.shape, dtype=np.float64)
    for b in range(B):
        for r in range(H):
            for c in range(W):
                v = x[b, :, r, c]
                new = np.empty(dim, dtype=np.float64)
                if mode == "1d":
                    spans = [(0, dim // 2, r * W + c, -2.0 / dim)]
                else:
                    spans = [(0, dim // 4, r, -4.0 / dim),
                             (dim // 2, dim // 4, c, -4.0 / dim)]
                for start, pairs, coord, exponent in spans:
                    for j in range(pairs):
                        a = coord * theta ** (exponent * j)
                        i0, i1 = start + 2 * j, start + 2 * j + 1
                        new[i0] = v[i0] * math.cos(a) - v[i1] * math.sin(a)
                        new[i1] = v[i0] * math.sin(a) + v[i1] * math.cos(a)
                out[b, :, r, c] = new
    return out


def ref_rotate(x, mode, theta=THETA):
    return torch.from_numpy(ref_rotate_np(x.detach().double().numpy(), mode, theta))


def pair_angle(mode, dim, H, W, r, c, pair, theta=THETA):
    """Angle of the adjacent pair `pair` (0 .. dim/2 - 1) at cell (r, c)."""
    if mode == "1d":
        return (r * W + c) * theta ** (-2.0 * pair / dim)
    if pair < dim // 4:
        return r * theta ** (-4.0 * pair / dim)
    return c * theta ** (-4.0 * (pair - dim // 4) / dim)


class SiLUThenRotate(nn.Module):
    """Spliced in for a pre-edit mixer's `act`: its last step before the scan."""
    def __init__(self, rotate):
        super().__init__()
        self.rotate = rotate

    def forward(self, x):
        return self.rotate(F.silu(x))


def splice(module, rotate):
    for m in module.modules():
        if isinstance(m, (REF_VM.SS2D, REF_SP.StructureAwareSSM)):
            m.act = SiLUThenRotate(rotate)
    return module


def module_rotation(mode, theta=THETA):
    return lambda z: apply_feature_rope(z, mode, theta)


def independent_rotation(mode, theta=THETA):
    return lambda z: ref_rotate(z, mode, theta).to(z.dtype)


def mixer_out(kind, mixer, x, H, W):
    return mixer(x, H, W, cond=cond_in.to(x.dtype)) if kind == "vm" else mixer(x, H, W)


CASES = [
    ("vm", "d_state 16 baseline", dict(VM), 0),
    ("vm", "prefix tokens", dict(VM, in_context_len=2, in_context_start=1,
                                 in_context_content="class"), 2),
    ("vm", "row tokens", dict(VM, in_context_len=GRID, in_context_start=1,
                              in_context_content="class", in_context_layout="row"), GRID),
    ("vm", "diagonal tokens", dict(VM, in_context_len=2, in_context_start=1,
                                   in_context_content="class",
                                   in_context_layout="diagonal"), 2),
    ("vm", "ssc bc", dict(VM, ssc="bc"), 0),
    ("vm", "state-init learned", dict(VM, state_init="learned"), 0),
    ("sp", "no gate", dict(SP, gate=False), 0),
    ("sp", "gate", dict(SP, gate=True), 0),
    ("sp", "no gate + tokens", dict(SP, gate=False, in_context_len=2, in_context_start=1,
                                    in_context_positions=[[0, 1], [2, 2]]), 2),
    ("sp", "gate + tokens", dict(SP, gate=True, in_context_len=2, in_context_start=1,
                                 in_context_positions=[[0, 1], [2, 2]]), 2),
]
CLS = {"vm": JiTVMamba, "sp": JiTSpatialMamba}
REF_CLS = {"vm": REF_VM.JiTVMamba, "sp": REF_SP.JiTSpatialMamba}


# ═══ A. Disabled path: bit-identical to the pre-edit implementation ═════════
prim_ref = reference_source("src/primitives.py").replace("\r\n", "\n")
with open(os.path.join(REPO_ROOT, "src", "primitives.py"), encoding="utf-8") as f:
    assert f.read().replace("\r\n", "\n") == prim_ref, "src/primitives.py changed"

for kind, label, kw, P in CASES:
    torch.manual_seed(0)
    old = REF_CLS[kind](**KW, **kw).eval()
    rng_old = torch.get_rng_state()
    torch.manual_seed(0)
    new_absent = CLS[kind](**KW, **kw).eval()
    rng_new = torch.get_rng_state()
    new_none = build(CLS[kind], feature_rope="none", **kw)
    assert torch.equal(rng_old, rng_new), f"init consumed different RNG: {label}"
    so = old.state_dict()
    for new in (new_absent, new_none):
        sn = new.state_dict()
        assert list(so) == list(sn), f"state_dict keys changed: {label}"
        assert all(torch.equal(so[k], sn[k]) for k in so), f"init changed: {label}"

    perturb(old)
    for new in (new_absent, new_none):
        new.load_state_dict(old.state_dict(), strict=True)
    assert old.blocks[0].adaLN_modulation[-1].weight.abs().max() > 0
    assert old.final_layer.linear.weight.abs().max() > 0

    x_mix = torch.randn(BATCH, P + GRID * GRID, D)
    rng_before = torch.get_rng_state()
    with torch.no_grad():
        mix_old = mixer_out(kind, old.blocks[-1].mixer, x_mix, GRID, GRID)
        out_old = old(x_img, t_in, y_in)
        for new in (new_absent, new_none):
            mix_new = mixer_out(kind, new.blocks[-1].mixer, x_mix, GRID, GRID)
            out_new = new(x_img, t_in, y_in)
            assert mix_old.abs().max() > 0 and out_old.abs().max() > 0, f"vacuous: {label}"
            d_mix, d_out = max_diff(mix_old, mix_new), max_diff(out_old, out_new)
            assert d_mix == 0.0 and d_out == 0.0, f"{label}: mixer {d_mix}, model {d_out}"
    assert torch.equal(rng_before, torch.get_rng_state()), f"forward used RNG: {label}"
    print(f"A  {kind} {label}: none/absent == pre-edit (mixer and model max|diff| = 0)")


# ═══ B. The rotation itself ═════════════════════════════════════════════════
H_R, W_R, D_R = 3, 5, 16
torch.manual_seed(2)
x64 = torch.randn(2, D_R, H_R, W_R, dtype=torch.float64)

for mode in MODES:
    for theta in (THETA, 100.0):
        got = apply_feature_rope(x64, mode, theta)
        want = ref_rotate(x64, mode, theta)
        assert got.shape == x64.shape and got.dtype == torch.float64
        d = (got - want).abs().max().item()
        assert d < TOL, f"{mode} theta {theta}: {d}"
        assert (got - x64).abs().max() > 0.1, "the rotation did nothing"

    # Pair ordering: each basis vector goes where the 2x2 matrix sends it.
    for pair in range(D_R // 2):
        for second in (False, True):
            basis = torch.zeros(1, D_R, H_R, W_R)
            basis[:, 2 * pair + int(second)] = 1.0
            rot = apply_feature_rope(basis, mode, THETA)
            for r in range(H_R):
                for c in range(W_R):
                    a = pair_angle(mode, D_R, H_R, W_R, r, c, pair)
                    want0, want1 = (-math.sin(a), math.cos(a)) if second \
                        else (math.cos(a), math.sin(a))
                    v = rot[0, :, r, c]
                    assert abs(v[2 * pair].item() - want0) < TOL
                    assert abs(v[2 * pair + 1].item() - want1) < TOL
                    others = torch.cat([v[:2 * pair], v[2 * pair + 2:]])
                    assert others.abs().max() == 0, "leak into another pair"

    # Identity at the origin, and rotations preserve every pair's norm.
    x32 = x64.float()
    got32 = apply_feature_rope(x32, mode, THETA)
    assert got32.dtype == torch.float32
    assert torch.equal(got32[:, :, 0, 0], x32[:, :, 0, 0]), "origin is not the identity"
    got = apply_feature_rope(x64, mode, THETA)
    pair_ratio = got.reshape(2, -1, 2, H_R, W_R).norm(dim=2) \
        / x64.reshape(2, -1, 2, H_R, W_R).norm(dim=2)
    assert (pair_ratio - 1).abs().max() < 1e-6, "norm not preserved"
print(f"B  rectangular {H_R}x{W_R}, D={D_R}: formula, pair order, origin, norms == reference")

# Row / column allocation of 2d_axial, on a spatially constant input.
const = torch.randn(1, D_R, 1, 1).expand(1, D_R, H_R, W_R)
rot = apply_feature_rope(const, "2d_axial", THETA)
half = D_R // 2
assert (rot[:, :half] - rot[:, :half, :, 0, None]).abs().max() == 0, "row half varies with column"
assert (rot[:, half:] - rot[:, half:, 0, None, :]).abs().max() == 0, "column half varies with row"
assert (rot[:, :half, 1, 0] - rot[:, :half, 0, 0]).abs().max() > 1e-3, "row half ignores the row"
assert (rot[:, half:, 0, 1] - rot[:, half:, 0, 0]).abs().max() > 1e-3, "column half ignores the column"
assert torch.equal(rot[:, half:, 2, 0], const[:, half:, 2, 0]), "column 0 must leave its half alone"
assert torch.equal(rot[:, :half, 0, 3], const[:, :half, 0, 3]), "row 0 must leave its half alone"
rot1 = apply_feature_rope(const, "1d", THETA)
assert (rot1[:, :, 1, 0] - rot1[:, :, 0, W_R - 1]).abs().max() > 1e-3
print("B  2d_axial: first half follows the row, second half the column")

# Same construction as JiT's attention RoPE, where the two are comparable.
N, D_J = 8, 32
head = VisionRotaryEmbeddingFast(dim=D_J // 2, pt_seq_len=N)
t = torch.randn(2, 1, N * N, D_J)
via_jit = head(t)[:, 0].transpose(1, 2).reshape(2, D_J, N, N)
via_module = apply_feature_rope(t[:, 0].transpose(1, 2).reshape(2, D_J, N, N), "2d_axial", THETA)
assert max_diff(via_jit, via_module) == 0.0, "2d_axial differs from JiT's construction"
print(f"B  2d_axial == VisionRotaryEmbeddingFast on {N}x{N}, D={D_J} (max|diff| = 0)")

# bf16 autocast: fp32 product, one rounding on the way out.
xb = x64.float().to(torch.bfloat16)
with torch.autocast("cpu", dtype=torch.bfloat16):
    out_b = apply_feature_rope(xb, "2d_axial", THETA)
assert out_b.dtype == torch.bfloat16
assert max_diff(out_b, apply_feature_rope(xb.float(), "2d_axial", THETA).to(torch.bfloat16)) == 0.0
print("B  bf16 input returns bf16, rounded once from the fp32 rotation")

# Cache: shared per key, separate across everything the tables depend on.
tbl = fr._rotation_tables
cpu = torch.device("cpu")
fr._rotation_tables.cache_clear()
key = ("1d", THETA, 16, 3, 5, cpu, torch.float32)
assert tbl(*key)[0] is tbl(*key)[0], "same key must not rebuild the table"
assert tbl(*key)[0].dtype == torch.float32 and tbl(*key)[0].shape == (1, 8, 3, 5)
assert tbl("1d", THETA, 16, 3, 5, cpu, torch.float64)[0].dtype == torch.float64
for other in (("1d", THETA, 16, 5, 3), ("1d", THETA, 32, 3, 5),
              ("2d_axial", THETA, 16, 3, 5), ("1d", 100.0, 16, 3, 5)):
    a, b = tbl(*key)[0], tbl(*other, cpu, torch.float32)[0]
    assert a.shape != b.shape or (a - b).abs().max() > 1e-3, f"table collision: {other}"
apply_feature_rope(x64.float(), "1d", THETA)
misses = fr._rotation_tables.cache_info().misses
apply_feature_rope(x64.float(), "1d", THETA)
assert fr._rotation_tables.cache_info().misses == misses, "second call rebuilt the table"
print("B  cache: reused per key, distinct per shape / width / mode / theta / dtype")

# A table first built in inference mode must not poison a later backward pass.
fr._rotation_tables.cache_clear()
with torch.inference_mode():
    apply_feature_rope(x64.float(), "2d_axial", THETA)
    assert not tbl("2d_axial", THETA, D_R, H_R, W_R, cpu, torch.float32)[0].is_inference()
x_g = x64.float().requires_grad_()
apply_feature_rope(x_g, "2d_axial", THETA).pow(2).sum().backward()
assert x_g.grad is not None and torch.isfinite(x_g.grad).all()
print("B  a table first built under inference_mode still works with autograd")

# The forward path has no NumPy or CPU transfer.
with open(os.path.join(REPO_ROOT, "src", "feature_rope.py"), encoding="utf-8") as f:
    tree = ast.parse(f.read())
imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
assert imported == {"functools", "math", "torch"}, imported
assert not attributes & {"cpu", "numpy", "item", "tolist"}, attributes

# Guards.
def rejected(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f"unclear error: {e}"
        return True
    return False


for bad in ("2d", "axial", "", None, "1D"):
    assert rejected(lambda: validate_feature_rope(bad, THETA, 16), "Unknown feature_rope")
for bad in (0, 0.0, -1.0, float("inf"), float("nan"), "abc", None, True, [1.0]):
    assert rejected(lambda: validate_feature_rope("1d", bad, 16), "finite, positive"), bad
    assert rejected(lambda: validate_feature_rope("2d_axial", bad, 16), "finite, positive"), bad
assert rejected(lambda: validate_feature_rope("1d", THETA, 15), "divisible by 2")
assert rejected(lambda: validate_feature_rope("2d_axial", THETA, 18), "divisible by 4")
assert validate_feature_rope("1d", THETA, 18) == THETA
assert validate_feature_rope("1d", "1e4", 16) == THETA
assert validate_feature_rope("none", float("nan"), 15) is None
for cls in (JiTVMamba, JiTSpatialMamba):
    assert rejected(lambda: cls(**KW, feature_rope="bogus"), "Unknown feature_rope")
    assert rejected(lambda: cls(**KW, feature_rope="1d", feature_rope_theta=0), "finite")
    assert rejected(lambda: cls(**dict(KW, hidden_size=66), feature_rope="2d_axial"), "divisible by 4")
assert rejected(lambda: SS2D(d_model=66, feature_rope="2d_axial"), "divisible by 4")
assert rejected(lambda: StructureAwareSSM(d_model=66, feature_rope="2d_axial"), "divisible by 4")

# The conditioning arms stay mutually exclusive with RoPE on.
for arms in (dict(ssc="bc", state_init="learned"),
             dict(ssc="bc", in_context_len=2, in_context_content="class")):
    try:
        JiTVMamba(**KW, feature_rope="1d", **arms)
    except AssertionError as e:
        assert "conditioning arms" in str(e) or "separate" in str(e), e
    else:
        raise AssertionError("n_arms guard was bypassed")
print("B  guards: bad modes, bad theta, bad widths rejected; arms still exclusive; "
      "no NumPy / CPU transfer in the forward path")


# ═══ C. Placement and coordinates ═══════════════════════════════════════════
worst_independent = 0.0
for kind, label, kw, P in CASES:
    for mode in MODES:
        old = splice(perturb(build(REF_CLS[kind], **kw)), module_rotation(mode))
        new = build(CLS[kind], feature_rope=mode, feature_rope_theta=THETA, **kw)
        new.load_state_dict(old.state_dict(), strict=True)
        x_mix = torch.randn(BATCH, P + GRID * GRID, D)
        with torch.no_grad():
            d_mix = max_diff(mixer_out(kind, old.blocks[-1].mixer, x_mix, GRID, GRID),
                             mixer_out(kind, new.blocks[-1].mixer, x_mix, GRID, GRID))
            out_new = new(x_img, t_in, y_in)
            d_out = max_diff(old(x_img, t_in, y_in), out_new)
        assert d_mix == 0.0 and d_out == 0.0, f"{label} {mode}: {d_mix}, {d_out}"

        # The same mixer comparison through the independent rotation, in float64
        # (the full model cannot run there: its time embedding is always fp32).
        old_d = copy.deepcopy(old.blocks[-1].mixer).double()
        splice(old_d, independent_rotation(mode))
        new_d = copy.deepcopy(new.blocks[-1].mixer).double()
        x_d = x_mix.double()
        with torch.no_grad():
            d_ind = max_diff(mixer_out(kind, old_d, x_d, GRID, GRID),
                             mixer_out(kind, new_d, x_d, GRID, GRID))
        assert d_ind < 1e-3, f"{label} {mode}: independent rotation differs by {d_ind}"
        worst_independent = max(worst_independent, d_ind)

        # A rotation that does nothing must fail this comparison.
        plain = perturb(build(REF_CLS[kind], **kw))
        with torch.no_grad():
            assert max_diff(plain(x_img, t_in, y_in), out_new) > 1e-3, f"vacuous: {label}"
    print(f"C  {kind} {label}: new mixer == pre-edit mixer + rotation after SiLU, "
          f"both modes (bit-exact; independent rotation within tolerance)")
print(f"C  worst mixer difference through the independent float64 rotation: {worst_independent:.2e}")

H_R, W_R, L_R, K = 3, 5, 15, 4
with torch.no_grad():
    for mode in MODES:
        for E in (0, 2):                # plain grid, and prefix tokens
            mix = perturb(SS2D(d_model=D, d_state=4, feature_rope=mode).eval())
            x = torch.randn(2, E + L_R, D)
            SCAN_CALLS.clear()
            CAPTURING[0] = True
            mix(x, H_R, W_R)
            CAPTURING[0] = False
            call = SCAN_CALLS[0]
            u = call["u"].view(2, K, D, E + L_R)

            z = mix.in_proj(x)
            z_tok = z[:, :E].transpose(1, 2)
            z2d = F.silu(mix.conv2d(z[:, E:].view(2, H_R, W_R, D).permute(0, 3, 1, 2)))
            rotated = ref_rotate(z2d, mode)

            # Every cell sits, already rotated, at its explicit position in each path.
            for r in range(H_R):
                for c in range(W_R):
                    steps = [r * W_R + c, L_R - 1 - (r * W_R + c),
                             c * H_R + r, L_R - 1 - (c * H_R + r)]
                    for k, s in enumerate(steps):
                        d = (u[:, k, :, E + s].double() - rotated[:, :, r, c]).abs().max().item()
                        assert d < TOL, f"{mode} E={E} path {k} cell ({r},{c}): {d}"
            for k in range(K):                       # tokens: never rotated
                assert torch.equal(u[:, k, :, :E], z_tok), "a prefix token was rotated"

            # dt, B and C come from the rotated scan input.
            x_dbl = torch.einsum("bkdl,kod->bkol", u, mix.x_proj_weight)
            dt_r, b_ssm, c_ssm = torch.split(x_dbl, [mix.dt_rank, 4, 4], dim=2)
            dt = torch.einsum("bkrl,kdr->bkdl", dt_r, mix.dt_projs_weight)
            assert max_diff(call["B"], b_ssm.reshape(2, K, 4, E + L_R)) < 1e-5
            assert max_diff(call["C"], c_ssm.reshape(2, K, 4, E + L_R)) < 1e-5
            assert max_diff(call["delta"], dt.reshape(2, K * D, E + L_R)) < 1e-5

            # Rotating by scan index instead would be a different model.
            if mode == "1d" and E == 0:
                scanned = cross_scan(z2d)
                for k in (1, 2, 3):
                    per_scan = ref_rotate(scanned[:, k, :, None, :], "1d")[:, :, 0, :]
                    assert (per_scan - u[:, k].double()).abs().max() > 1e-2, "control is vacuous"
print("C  SS2D 3x5: rotated cells at their exact position in all 4 paths; "
      "dt, B, C read the rotated input; prefix tokens not rotated")

with torch.no_grad():
    for mode in MODES:
        for layout, E in (("row", GRID), ("diagonal", 2)):
            mix = perturb(SS2D(d_model=D, d_state=4, feature_rope=mode,
                               in_context_layout=layout).eval())
            x = torch.randn(2, E + GRID * GRID, D)
            SCAN_CALLS.clear()
            CAPTURING[0] = True
            mix(x, GRID, GRID)
            CAPTURING[0] = False
            u = SCAN_CALLS[0]["u"].view(2, K, D, -1)
            z = mix.in_proj(x)
            z_tok = z[:, :E].transpose(1, 2)
            z2d = F.silu(mix.conv2d(z[:, E:].view(2, GRID, GRID, D).permute(0, 3, 1, 2)))
            rotated = ref_rotate(z2d, mode)
            if layout == "row":
                units = u.view(2, K, D, GRID, 1 + GRID)
                tokens = units[..., 0].reshape(2, K, D, GRID)
                grid = units[..., 1:].reshape(2, K, D, -1)
                for k in range(K):
                    assert torch.equal(tokens[:, k], z_tok), "a row token was rotated"
                assert (grid.double() - cross_scan(rotated)).abs().max() < TOL
            else:
                idx, _ = diagonal_scan_index(GRID, GRID, E, torch.device("cpu"))
                joint = torch.cat([z_tok.double(), rotated.reshape(2, D, -1)], dim=-1)
                want = joint[:, :, idx].transpose(1, 2)
                assert (u.double() - want).abs().max() < TOL, "diagonal layout moved an image cell"
print("C  SS2D row / diagonal layouts: tokens not rotated, image cells keep their coordinates")

with torch.no_grad():
    for mode in MODES:
        for gate in (False, True):
            positions = (0, 7)
            mix = perturb(StructureAwareSSM(
                d_model=D, d_state=1, gate=gate, dilations=(1,),
                in_context_positions=positions, feature_rope=mode).eval())
            x = torch.randn(2, len(positions) + L_R, D)
            SCAN_CALLS.clear()
            CAPTURING[0] = True
            mix(x, H_R, W_R)
            mix(x[:, len(positions):], H_R, W_R)   # the same grid, no tokens
            CAPTURING[0] = False
            u_tok, u_plain = SCAN_CALLS[0]["u"], SCAN_CALLS[1]["u"]

            inp = mix.in_proj(x)
            u_in = inp.chunk(2, dim=-1)[0] if gate else inp
            z_tok = u_in[:, :2].transpose(1, 2)
            u2d = F.silu(mix.conv2d(u_in[:, 2:].view(2, H_R, W_R, D).permute(0, 3, 1, 2)))
            rotated = ref_rotate(u2d, mode).reshape(2, D, -1)

            # Scan order: token 0, cells 0..6, token 1, cells 7..14.
            token_steps, grid_steps = [0, 8], [i for i in range(17) if i not in (0, 8)]
            assert torch.equal(u_tok[:, :, token_steps], z_tok), "a Spatial token was rotated"
            assert (u_tok[:, :, grid_steps].double() - rotated).abs().max() < TOL
            assert torch.equal(u_tok[:, :, grid_steps], u_plain), \
                "inserting tokens changed an image coordinate"
print("C  Spatial 3x5 with tokens: rotated in row-major, tokens not rotated, "
      "inserting tokens leaves image cells bit-identical")


# ═══ D. Forward / backward of the active modes ══════════════════════════════
BUILDS = [
    (JiTVMamba, "VMamba d_state 16", dict(VM)),
    (JiTVMamba, "VMamba d_state 16 + prefix", dict(VM, in_context_len=2, in_context_start=1,
                                                   in_context_content="class")),
    (JiTSpatialMamba, "Spatial d_state 1 no gate", dict(SP, gate=False)),
    (JiTSpatialMamba, "Spatial d_state 1 gate", dict(SP, gate=True)),
    (JiTSpatialMamba, "Spatial no gate + tokens",
     dict(SP, gate=False, in_context_len=2, in_context_start=1,
          in_context_positions=[[0, 1], [2, 2]])),
]
for cls, label, kw in BUILDS:
    base = perturb(build(cls, **kw))
    with torch.no_grad():
        out_none = base(x_img, t_in, y_in)
    outs = {}
    for mode in MODES:
        model = build(cls, feature_rope=mode, **kw)
        model.load_state_dict(base.state_dict(), strict=True)
        model.train()
        model.zero_grad()
        rng = torch.get_rng_state()
        out = model(x_img, t_in, y_in)
        assert torch.equal(rng, torch.get_rng_state()), "RoPE consumed RNG in forward"
        assert torch.isfinite(out).all() and out.shape == x_img.shape, f"{label} {mode}"
        out.pow(2).mean().backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, f"non-finite gradients: {bad[:3]}"
        mixer = model.blocks[-1].mixer
        for name in ("x_proj_weight", "dt_projs_weight"):
            assert getattr(mixer, name).grad.abs().max() > 0, f"{label} {mode}: no grad in {name}"
        assert mixer.in_proj.weight.grad.abs().max() > 0
        assert (out.detach() - out_none).abs().max() > 1e-3, f"{label} {mode}: RoPE had no effect"
        outs[mode] = out.detach()

        SCAN_DTYPES.clear()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf = model(x_img, t_in, y_in)
        active_dtypes = set(SCAN_DTYPES)
        SCAN_DTYPES.clear()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            base(x_img, t_in, y_in)
        assert active_dtypes == set(SCAN_DTYPES), f"{label} {mode}: scan dtypes changed"
        assert torch.isfinite(bf.float()).all()
        with torch.inference_mode():
            assert torch.isfinite(model.eval()(x_img, t_in, y_in)).all()
    assert (outs["1d"] - outs["2d_axial"]).abs().max() > 1e-3, f"{label}: the modes coincide"
    print(f"D  {label}: both modes finite forward + backward, RoPE changes the output, "
          f"bf16 scan dtypes unchanged")


# ═══ E. Parameters, checkpoints, builders, configs ══════════════════════════
def load_build_model(rel):
    path = os.path.join(REPO_ROOT, rel)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_model")
    ns = {}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns["build_model"]


bm_train = load_build_model("scripts/run_experiment.py")
bm_eval = load_build_model("scripts/evaluate.py")
bm_prof = load_build_model("scripts/profile_model.py")


def seeded(builder, cfg):
    torch.manual_seed(0)
    return builder(cfg).eval()


def flatten(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        out.update(flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


ALLOWED_DIFF = {"experiment.name", "checkpoint.output_dir",
                "model.feature_rope", "model.feature_rope_theta"}
CFG_DIR = os.path.join(REPO_ROOT, "configs", "tiny_imagenet")
SOURCES = {
    "jit-s2-vmamba-baseline": "jit-s2-vmamba",
    "jit-s2-spatial-nogate": "jit-s2-spatial-nogate",
}
identifiers = {}
for fname in sorted(os.listdir(CFG_DIR)):
    if fname.endswith(".yaml"):
        with open(os.path.join(CFG_DIR, fname), encoding="utf-8") as f:
            c = yaml.safe_load(f)
        identifiers[fname] = (c["experiment"]["name"], c["checkpoint"]["output_dir"])
assert len({v[0] for v in identifiers.values()}) == len(identifiers), "duplicate experiment.name"
assert len({v[1] for v in identifiers.values()}) == len(identifiers), "duplicate output_dir"

for source, stem in SOURCES.items():
    base_cfg = load_config(os.path.join(CFG_DIR, f"{source}.yaml"))
    assert "feature_rope" not in base_cfg["model"], "a baseline config was modified"
    base = seeded(bm_train, base_cfg)
    assert base.blocks[0].mixer.feature_rope == "none", "missing key must mean 'none'"
    base_params = n_params(base)
    for mode, slug in (("1d", "1d"), ("2d_axial", "2d-axial")):
        name = f"{stem}-feature-rope-{slug}.yaml"
        cfg = load_config(os.path.join(CFG_DIR, name))
        a, b = flatten(base_cfg), flatten(cfg)
        changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
        assert changed == ALLOWED_DIFF, f"{name} changes {changed - ALLOWED_DIFF}"
        assert b["model.feature_rope"] == mode and b["model.feature_rope_theta"] == 10000.0
        assert b["model.d_state"] == (16 if "vmamba" in source else 1)

        train, evaluate = seeded(bm_train, cfg), seeded(bm_eval, cfg)
        for m in (train, evaluate):
            assert all(bl.mixer.feature_rope == mode and bl.mixer.feature_rope_theta == 10000.0
                       for bl in m.blocks), f"{name}: flag not threaded to every mixer"
        st, se = train.state_dict(), evaluate.state_dict()
        assert list(st) == list(se) and all(torch.equal(st[k], se[k]) for k in st)
        assert n_params(train) == base_params, f"{name}: parameter count changed"
        sb = base.state_dict()
        assert list(st) == list(sb), f"{name}: state_dict keys changed"
        assert all(torch.equal(st[k], sb[k]) for k in st), f"{name}: init changed"
        assert not any("rope" in k or "cos" in k or "sin" in k for k in st)
        train.load_state_dict(base.state_dict(), strict=True)
        base.load_state_dict(train.state_dict(), strict=True)

        if cfg["experiment"]["model"] == "vmamba":
            prof = seeded(bm_prof, cfg)
            assert all(bl.mixer.feature_rope == mode for bl in prof.blocks)

        # The real 64x64, depth-12 model: one forward + backward pass.
        perturb(train)
        train.train()
        xr = torch.randn(1, 3, 64, 64)
        out = train(xr, torch.rand(1), torch.zeros(1, dtype=torch.long))
        out.pow(2).mean().backward()
        assert torch.isfinite(out).all() and all(
            p.grad is None or torch.isfinite(p.grad).all() for p in train.parameters())
        print(f"E  {name}: train == eval, {n_params(train):,} params (== base), strict load both "
              f"ways, only RoPE keys / identifiers differ, full-size fwd+bwd finite")

# A theta passed under a different value must reach the mixers through both builders.
for builder in (bm_train, bm_eval):
    for source in ("jit-s2-vmamba-baseline", "jit-s2-spatial-nogate"):
        cfg = load_config(os.path.join(CFG_DIR, f"{source}.yaml"))
        cfg["model"].update(depth=1, feature_rope="2d_axial", feature_rope_theta=500.0)
        m = builder(cfg)
        assert m.blocks[0].mixer.feature_rope == "2d_axial"
        assert m.blocks[0].mixer.feature_rope_theta == 500.0
print("E  run_experiment / evaluate forward feature_rope_theta for both architectures")

print("ALL CHECKS PASSED")
