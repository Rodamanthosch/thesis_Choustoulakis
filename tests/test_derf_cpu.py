"""
tests/test_derf_cpu.py
======================
CPU-only verification of `norm: derf` — Dynamic erf (Chen et al., "Stronger
Normalization-Free Transformers", CVPR 2026) in place of the pre-mixer,
pre-FFN and final RMSNorms, for vmamba, vmamba_sasf, spatial_mamba, vmamba2.

Runs WITHOUT a GPU and WITHOUT mamba_ssm: both kernels are stubbed with
reference recurrences.

Checks:
  A. norm="rmsnorm" (the default) is BIT-IDENTICAL to HEAD for all four models
     (same init at the same seed, and the same output after a random
     perturbation of every weight, so adaLN-Zero's zero output can't hide a
     difference), and src/primitives.py only ADDS lines vs HEAD.
  B. Derf equals a verbatim copy of the official Dynamic_erf
     (zlab-princeton/Derf, DiT/dynamic_erf.py), max|diff| == 0 in fp32, and in
     bf16 it is the fp32 result cast back (as RMSNorm).
  C. norm="derf" swaps exactly norm1 + norm2 of every block and norm_final
     (2 * depth + 1), leaves every other norm (SS2D's out_norm, Spatial's
     LN(y)) alone, keeps EVERY baseline weight bit-identical at the same seed
     and only adds {bias, alpha, shift} per swapped norm (+(2*depth+1)*(D+2)
     params); alpha starts at derf_alpha_init, shift and bias at 0.
  D. gradients reach alpha, shift, weight and bias of every Derf.
  E. add_weight_decay puts every Derf parameter in the no-decay group.
  F. scripts/run_experiment.py::build_model and scripts/evaluate.py::build_model
     build the same Derf model from configs/tiny_imagenet/jit-s2-vmamba-derf.yaml
     (and for the other three archs from a small cfg).

Run from the repo root:
    python tests/test_derf_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
import ast
import difflib
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


# ── stub mamba_ssm: Mamba-1 selective scan + Mamba-2 SSD as recurrences ──────
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, g, n, l) grouped."""
    b, d, l = u.shape
    u, delta, A = u.float(), delta.float(), A.float()
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    rep = d // B.shape[1]
    Bf = B.float().repeat_interleave(rep, dim=1)     # (b, d, n, l)
    Cf = C.float().repeat_interleave(rep, dim=1)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for i in range(l):
        h = torch.exp(delta[:, :, i, None] * A[None]) * h \
            + delta[:, :, i, None] * Bf[:, :, :, i] * u[:, :, i, None]
        ys.append((h * Cf[:, :, :, i]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D[None, :, None].float() * u
    return y


def ssd_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
            initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False,
            dt_limit=(0.0, float("inf")), return_final_states=False,
            return_varlen_states=False):
    """x: (b, l, h, p)  dt: (b, l, h)  A: (h,)  B, C: (b, l, g, n)  D: (h, p) | (h,)."""
    b, l, h, p = x.shape
    xw, dtw, Aw = x.float(), dt.float(), A.float()
    if dt_bias is not None:
        dtw = dtw + dt_bias.float()
    if dt_softplus:
        dtw = F.softplus(dtw)
    rep = h // B.shape[2]
    Bh = B.float().repeat_interleave(rep, dim=2)     # (b, l, h, n)
    Ch = C.float().repeat_interleave(rep, dim=2)
    st = xw.new_zeros(b, h, p, B.shape[-1])
    ys = []
    for t in range(l):
        st = torch.exp(dtw[:, t] * Aw)[:, :, None, None] * st \
            + (dtw[:, t, :, None] * xw[:, t])[..., None] * Bh[:, t, :, None, :]
        ys.append((st * Ch[:, t, :, None, :]).sum(-1))
    y = torch.stack(ys, dim=1)                       # (b, l, h, p)
    if D is not None:
        Dw = D.float()
        y = y + xw * (Dw[None, None] if Dw.dim() == 2 else Dw[None, None, :, None])
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(x.dtype)


if "mamba_ssm" not in sys.modules:
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
        sys.modules[name] = types.ModuleType(name)
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = selective_scan_ref
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = iface
    ssd_mod = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
    ssd_mod.mamba_chunk_scan_combined = ssd_ref
    sys.modules["mamba_ssm.ops.triton.ssd_combined"] = ssd_mod

from src.primitives import RMSNorm, Derf                     # noqa: E402  (after the stubs)
from src.models.vmamba import JiTVMamba                      # noqa: E402
from src.models.vmamba_sasf import JiTVMambaSASF             # noqa: E402
from src.models.spatial_mamba import JiTSpatialMamba         # noqa: E402
from src.models.vmamba2 import JiTVMamba2                    # noqa: E402
from src.utils import add_weight_decay                       # noqa: E402

KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32)
D, DEPTH = KW["hidden_size"], KW["depth"]
N_SWAP = 2 * DEPTH + 1
B = 3
torch.manual_seed(0)
x_img = torch.randn(B, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(B)
y_in = torch.randint(0, KW["num_classes"], (B,))

# (file, class name, current class, extra kwargs)
MODELS = [
    ("src/models/vmamba.py", "JiTVMamba", JiTVMamba, dict(d_state=4)),
    ("src/models/vmamba_sasf.py", "JiTVMambaSASF", JiTVMambaSASF,
     dict(sasf_dilations=(1, 2, 3))),
    ("src/models/spatial_mamba.py", "JiTSpatialMamba", JiTSpatialMamba,
     dict(gate=False, lpu="zero", sasf_dilations=(1, 2, 3))),
    ("src/models/vmamba2.py", "JiTVMamba2", JiTVMamba2, dict(d_state=4)),
]
# extra vmamba arms the rmsnorm path must leave untouched
VMAMBA_ARMS = [
    ("in-context class K=2", dict(in_context_len=2, in_context_content="class")),
    ("ssc bc + adaln mlp", dict(ssc="bc", adaln_cond="mlp")),
    ("ssc bc + adaln none", dict(ssc="bc", adaln_cond="none")),
    ("ffn convglu", dict(ffn="convglu")),
]


def build(cls, **kw):
    torch.manual_seed(0)
    return cls(**KW, **kw).eval()


def perturb(model, seed=1):
    """Add seeded noise to every parameter (in state_dict order), so the
    zero-init adaLN gates and output layer no longer zero the output."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.add_(0.1 * torch.randn(p.shape, generator=g))
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


# ═══ A. norm="rmsnorm" is bit-identical to HEAD ══════════════════════════════
prim_head = head_source("src/primitives.py")
if prim_head is not None:
    cur = open(os.path.join(REPO_ROOT, "src", "primitives.py"), encoding="utf-8").read()
    removed = [l for l in difflib.ndiff(prim_head.splitlines(), cur.splitlines())
               if l.startswith("- ")]
    assert not removed, f"src/primitives.py changed existing lines: {removed[:3]}"

cases = [(rel, name, cls, kw, "") for rel, name, cls, kw in MODELS]
cases += [(MODELS[0][0], MODELS[0][1], MODELS[0][2], kw, label) for label, kw in VMAMBA_ARMS]
n_checked = 0
for rel, name, cls, kw, label in cases:
    src = head_source(rel)
    if src is None:
        print(f"A  SKIPPED for {rel}")
        continue
    head_mod = types.ModuleType(f"head_{name}")
    exec(compile(src, f"<HEAD:{rel}>", "exec"), head_mod.__dict__)
    old = build(getattr(head_mod, name), **kw)
    new = build(cls, **kw)                                  # norm defaults to rmsnorm
    so, sn = old.state_dict(), new.state_dict()
    tag = f"{name} {label}".strip()
    assert list(so) == list(sn), f"state_dict keys changed for {tag}"
    assert all(torch.equal(so[k], sn[k]) for k in so), f"init changed for {tag}"
    perturb(old)
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.no_grad():
        out_old = old(x_img, t_in, y_in)
        d = (out_old - new(x_img, t_in, y_in)).abs().max().item()
    assert out_old.abs().max() > 0, f"perturbed output still zero for {tag}"
    assert d == 0.0, f"norm='rmsnorm' not bit-identical to HEAD for {tag}: {d}"
    n_checked += 1
print(f"A  norm='rmsnorm' bit-identical to HEAD for {n_checked} model/arm builds "
      f"(init + perturbed output, max|diff| = 0); primitives.py only adds lines")


# ═══ B. Derf == the official Dynamic_erf ══════════════════════════════════════
class Dynamic_erf(nn.Module):
    """Verbatim from zlab-princeton/Derf DiT/dynamic_erf.py."""
    def __init__(self, normalized_shape, elementwise_affine, alpha_init_value=0.5, shift_init_value=0.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)

    def forward(self, x):
        x = self.alpha * x + self.shift
        if self.elementwise_affine:
            return torch.erf(x) * self.weight + self.bias
        else:
            return torch.erf(x)


ours, ref = Derf(D), Dynamic_erf(D, True)
assert [n for n, _ in ours.named_parameters()] == [n for n, _ in ref.named_parameters()]
assert all(torch.equal(a, b) for a, b in zip(ours.state_dict().values(),
                                             ref.state_dict().values()))
torch.manual_seed(2)
with torch.no_grad():
    ref.alpha.fill_(0.73)
    ref.shift.fill_(-0.21)
    ref.weight.normal_()
    ref.bias.normal_()
ours.load_state_dict(ref.state_dict())
xb = 3 * torch.randn(4, 9, D)
with torch.no_grad():
    d32 = (ours(xb) - ref(xb)).abs().max().item()
    x16 = xb.bfloat16()
    y16 = ours(x16)
    d16 = (y16.float() - ref(x16.float()).bfloat16().float()).abs().max().item()
assert d32 == 0.0, d32
assert y16.dtype == torch.bfloat16 and d16 == 0.0, d16
print(f"B  Derf == official Dynamic_erf: fp32 max|diff| = {d32}, "
      f"bf16 == fp32 result cast back (max|diff| = {d16})")


# ═══ C. norm="derf": what is swapped, what is kept ═══════════════════════════
def count(model, typ):
    return sum(isinstance(m, typ) for m in model.modules())


for rel, name, cls, kw in MODELS:
    base = build(cls, **kw)
    derf = build(cls, norm="derf", **kw)
    assert derf.norm_type == "derf" and base.norm_type == "rmsnorm"
    assert count(base, RMSNorm) == N_SWAP and count(base, Derf) == 0, name
    assert count(derf, RMSNorm) == 0 and count(derf, Derf) == N_SWAP, name
    assert count(derf, nn.LayerNorm) == count(base, nn.LayerNorm) > 0, name
    swapped = [m for blk in derf.blocks for m in (blk.norm1, blk.norm2)]
    swapped.append(derf.final_layer.norm_final)
    assert all(isinstance(m, Derf) for m in swapped), name
    sb, sd = base.state_dict(), derf.state_dict()
    assert all(torch.equal(sb[k], sd[k]) for k in sb), f"{name}: a baseline weight moved"
    new_keys = set(sd) - set(sb)
    prefixes = [f"blocks.{i}.{n}" for i in range(DEPTH) for n in ("norm1", "norm2")]
    prefixes.append("final_layer.norm_final")
    assert new_keys == {f"{p}.{s}" for p in prefixes for s in ("bias", "alpha", "shift")}, name
    assert n_params(derf) - n_params(base) == N_SWAP * (D + 2), name
    for m in swapped:
        assert m.alpha.item() == 0.5 and m.shift.item() == 0.0
        assert torch.equal(m.weight, torch.ones(D)) and torch.equal(m.bias, torch.zeros(D))
    derf1 = build(cls, norm="derf", derf_alpha_init=1.0, **kw)
    assert all(m.alpha.item() == 1.0 for m in derf1.modules() if isinstance(m, Derf))
print(f"C  norm='derf' swaps exactly norm1/norm2/norm_final ({N_SWAP} per model), "
      f"keeps out_norm / LN(y) and every baseline weight, adds "
      f"{N_SWAP} x (D + 2) = {N_SWAP * (D + 2)} params, alpha init honoured "
      f"(4 models)")

try:
    build(JiTVMamba, norm="layernorm")
except AssertionError:
    pass
else:
    raise AssertionError("unknown norm name was accepted")


# ═══ D. gradients reach every Derf parameter ═════════════════════════════════
for rel, name, cls, kw in MODELS:
    m = perturb(build(cls, norm="derf", **kw)).train()
    m(x_img, t_in, y_in).pow(2).mean().backward()
    for mod in (mod for mod in m.modules() if isinstance(mod, Derf)):
        for pn, p in mod.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"{name}: no grad on {pn}"
print("D  gradients reach alpha, shift, weight, bias of every Derf (4 models)")


# ═══ E. Derf parameters get no weight decay ══════════════════════════════════
m = build(JiTVMamba, norm="derf")
derf_ids = {id(p) for mod in m.modules() if isinstance(mod, Derf) for p in mod.parameters()}
no_decay, decay = add_weight_decay(m, weight_decay=0.05)
assert derf_ids <= {id(p) for p in no_decay["params"]}
assert not derf_ids & {id(p) for p in decay["params"]}
print(f"E  add_weight_decay: all {len(derf_ids)} Derf parameters in the no-decay group")


# ═══ F. build_model wiring (train == eval) ═══════════════════════════════════
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


cfg = load_cfg("jit-s2-vmamba-derf.yaml")
base_cfg = load_cfg("jit-s2-vmamba-baseline.yaml")
assert {k: v for k, v in cfg["model"].items() if k not in ("norm", "derf_alpha_init")} \
    == base_cfg["model"], "the Derf config differs from the baseline beyond the norm"
assert cfg["training"] == base_cfg["training"]
mt = same_build(cfg)
assert mt.norm_type == "derf" and count(mt, Derf) == 2 * 12 + 1
assert all(mod.alpha.item() == 0.5 for mod in mt.modules() if isinstance(mod, Derf))
n_derf, n_base = n_params(mt), n_params(same_build(base_cfg))
assert n_derf == n_base + 25 * (384 + 2), (n_derf, n_base)
print(f"F  jit-s2-vmamba-derf.yaml: train == eval build_model, {n_derf:,} params "
      f"(baseline {n_base:,} + 25 x 386)")

small = dict(input_size=16, patch_size=4, in_channels=3, hidden_size=64, depth=2,
             num_heads=1, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0,
             num_classes=7, bottleneck_dim=32, norm="derf", derf_alpha_init=1.0)
for arch, extra in [("vmamba_sasf", dict(d_state=1, sasf_dilations=[1, 2, 3])),
                    ("spatial_mamba", dict(gate=False, sasf_dilations=[1, 2, 3])),
                    ("vmamba2", dict(d_state=4))]:
    m = same_build({"experiment": {"model": arch}, "model": {**small, **extra}})
    assert count(m, Derf) == N_SWAP and count(m, RMSNorm) == 0, arch
    assert all(mod.alpha.item() == 1.0 for mod in m.modules() if isinstance(mod, Derf)), arch
print("F  vmamba_sasf / spatial_mamba / vmamba2: build_model passes norm and "
      "derf_alpha_init through (train == eval)")

print("ALL CHECKS PASSED")
