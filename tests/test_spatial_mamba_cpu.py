"""
tests/test_spatial_mamba_cpu.py
===============================
CPU-only verification of JiT-S2-Spatial-Mamba (src/models/spatial_mamba.py):
Spatial-Mamba's Structure-aware SSM mixer in the JiT scaffold.

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, kept
in float64 when fed float64 so the fidelity checks are exact).

Checks:
  A. existing models untouched: src/models/vmamba.py and src/primitives.py are
     identical to HEAD.
  B. state extraction: the scan called with C = 1 and D = None returns the SSM
     states x_t of an independent explicit recursion (float64).
  C. paper fidelity: the mixer equals a naive per-token transcription of
     Eq. (2)-(3) -- explicit replicate-clamped neighbour sums, y = C h + D u,
     LayerNorm, optional SiLU(z) gate, out_proj -- for both gate settings and
     several dilation sets (float64).
  D. official fidelity: StateFusion equals a verbatim transcription of the
     official StateFusion training branch (EdwardChasel/Spatial-Mamba), with
     the weights copied across (float64).
  E. reduction: with SASF set to the identity (centre tap, alpha = (1, 0, 0)),
     the mixer equals a plain unidirectional Mamba y = selective_scan(C, D).
  F. param counts at D=384: mixer 477,699 (gate) / 330,243 (no gate), LPU
     7,680 per block; Tiny-ImageNet models 31.46M / 29.69M with lpu=zero.
  G. smoke + init: forward + backward for gate x ffn x lpu, grads reach the
     SASF kernels, alpha and the LPU; every block is the identity at init for
     lpu none/zero; dt bias keeps its official range after initialize_weights;
     CPU bf16 autocast runs.
  H. guards: d_state != 1, unknown lpu / ffn, and a token count != H*W fail.
  I. wiring: the three new Tiny-ImageNet configs build IDENTICAL models through
     scripts/run_experiment.py::build_model and scripts/evaluate.py::build_model,
     with the expected param counts (needs pyyaml; skipped otherwise).
  J. drift guard: with each block's mixer swapped for SS2D(d_state=1), the
     JiTSpatialMamba scaffold (lpu=none) loads JiTVMamba(d_state=1)'s state_dict
     strictly and is BIT-IDENTICAL to it -- the duplicated block/model code
     cannot silently diverge from JiTVMamba.

Run from the repo root:
    PYTHONPATH=. python tests/test_spatial_mamba_cpu.py
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
    import torch.nn as nn
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
    JiTSpatialMamba, JiTSpatialBlock, StructureAwareSSM, StateFusion,
)
from src.models.vmamba import JiTVMamba, SS2D               # noqa: E402

torch.manual_seed(0)
KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32)
H = W = KW["input_size"] // KW["patch_size"]       # 4
HW = H * W                                         # 16
Bsz = 3

x_img = torch.randn(Bsz, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(Bsz)
y_in = torch.randint(0, KW["num_classes"], (Bsz,))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def perturb(module, scale=0.05):
    """Move every parameter off its (often degenerate) init, deterministically."""
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))
    return module


# ═══ A. existing models untouched ════════════════════════════════════════════
try:
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--", "src/models/vmamba.py", "src/primitives.py"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8")
    assert diff == "", "src/models/vmamba.py or src/primitives.py changed vs HEAD"
    print("A  src/models/vmamba.py and src/primitives.py identical to HEAD")
except (OSError, subprocess.CalledProcessError) as e:       # pragma: no cover
    print(f"A  SKIPPED (git not usable here: {e})")


# ═══ B. state extraction through C = 1, D = None ═════════════════════════════
torch.manual_seed(1)
d, L = 6, 11
u = torch.randn(2, d, L, dtype=torch.float64)
dt = torch.randn(2, d, L, dtype=torch.float64)
A = -torch.rand(d, 1, dtype=torch.float64) - 0.2
Bm = torch.randn(2, 1, 1, L, dtype=torch.float64)
bias = torch.randn(d, dtype=torch.float64)
got = selective_scan_ref(u, dt, A, Bm, torch.ones_like(Bm), None,
                         delta_bias=bias, delta_softplus=True)
# independent explicit recursion  x_t = exp(dt A) x_{t-1} + dt B_t u_t
delta = F.softplus(dt + bias[None, :, None])
xs, x_prev = [], torch.zeros(2, d, dtype=torch.float64)
for t in range(L):
    x_prev = torch.exp(delta[:, :, t] * A[:, 0]) * x_prev \
        + delta[:, :, t] * Bm[:, 0, 0, t, None] * u[:, :, t]
    xs.append(x_prev)
want = torch.stack(xs, dim=-1)
err_b = (got - want).abs().max().item()
assert err_b < 1e-12, err_b
print(f"B  scan with C=1, D=None returns the states x_t (max|diff| = {err_b:.1e})")


# ═══ C. Eq. (2)-(3) fidelity: naive per-token transcription ══════════════════
def naive_mixer(mx, x, Hh, Ww):
    """Spatial-Mamba's mixer written token by token, straight from the paper."""
    Bb, Ll, _ = x.shape
    di = mx.d_inner
    xz = x @ mx.in_proj.weight.T
    u, z = (xz[..., :di], xz[..., di:]) if mx.gate else (xz, None)
    u = mx.act(mx.conv2d(u.reshape(Bb, Hh, Ww, di).permute(0, 3, 1, 2)))
    u = u.reshape(Bb, di, Ll)                                   # row-major
    dbl = torch.einsum("od,bdl->bol", mx.x_proj_weight, u)
    dt_r = dbl[:, :mx.dt_rank]
    Bt, Ct = dbl[:, mx.dt_rank], dbl[:, mx.dt_rank + 1]          # (B, L) each
    # A and the dt bias go to the kernel in fp32 (as in SS2D); mirror that cast
    delta = F.softplus(torch.einsum("dr,brl->bdl", mx.dt_projs_weight, dt_r)
                       + mx.dt_projs_bias.float().to(x.dtype)[None, :, None])
    Av = -torch.exp(mx.A_logs.float()).to(x.dtype)[:, 0]
    # state transition (Eq. 2, left)
    st, xprev = [], torch.zeros(Bb, di, dtype=x.dtype)
    for t in range(Ll):
        xprev = torch.exp(delta[:, :, t] * Av) * xprev \
            + delta[:, :, t] * Bt[:, t, None] * u[:, :, t]
        st.append(xprev)
    st = torch.stack(st, dim=-1).reshape(Bb, di, Hh, Ww)
    # SASF (Eq. 3): replicate padding == clamping the neighbour index
    sf = mx.state_fusion
    h = torch.zeros_like(st)
    for r in range(Hh):
        for c in range(Ww):
            acc = 0
            for a, k, dil in zip(sf.alpha, sf.kernels, sf.dilations):
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        rr = min(max(r + i * dil, 0), Hh - 1)
                        cc = min(max(c + j * dil, 0), Ww - 1)
                        acc = acc + a * k[:, 0, i + 1, j + 1] * st[:, :, rr, cc]
            h[:, :, r, c] = acc
    h = h.reshape(Bb, di, Ll)
    # observation (Eq. 2, right)
    y = h * Ct[:, None, :] + u * mx.Ds[None, :, None]
    y = F.layer_norm(y.transpose(1, 2), (di,), mx.out_norm.weight, mx.out_norm.bias,
                     mx.out_norm.eps)
    if z is not None:
        y = y * F.silu(z)
    return y @ mx.out_proj.weight.T


worst_c = 0.0
for gate in (True, False):
    for dil in ((1, 3, 5), (1, 2, 3), (1,), (2, 4)):
        torch.manual_seed(2)
        mx = perturb(StructureAwareSSM(32, expand=1, gate=gate, dilations=dil)).double()
        x = torch.randn(2, HW, 32, dtype=torch.float64)
        with torch.no_grad():
            ref = naive_mixer(mx, x, H, W)
            dc = ((mx(x, H, W) - ref).abs().max() / ref.abs().max()).item()
        assert dc < 1e-12, (gate, dil, dc)
        worst_c = max(worst_c, dc)
torch.manual_seed(2)
mx = perturb(StructureAwareSSM(32, expand=2, gate=True, dilations=(1, 2))).double()
x = torch.randn(2, 6 * 5, 32, dtype=torch.float64)          # non-square grid
with torch.no_grad():
    ref = naive_mixer(mx, x, 6, 5)
    dc = ((mx(x, 6, 5) - ref).abs().max() / ref.abs().max()).item()
assert dc < 1e-12, dc
worst_c = max(worst_c, dc)
print(f"C  mixer == naive Eq. (2)-(3) transcription, gate on/off x 4 dilation "
      f"sets, + expand 2 on a 6x5 grid (max relative diff = {worst_c:.1e})")


# ═══ D. official StateFusion (training branch), verbatim ═════════════════════
# --- EdwardChasel/Spatial-Mamba, classification/models/spatialmamba.py ---
class OfficialStateFusion(nn.Module):
    def __init__(self, dim):
        super(OfficialStateFusion, self).__init__()
        self.dim = dim
        self.kernel_3   = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):
        if self.training:
            h1 = F.conv2d(self.padding(h, (1,1,1,1)), self.kernel_3, padding=0, dilation=1, groups=self.dim)
            h2 = F.conv2d(self.padding(h, (3,3,3,3)), self.kernel_3_1, padding=0, dilation=3, groups=self.dim)
            h3 = F.conv2d(self.padding(h, (5,5,5,5)), self.kernel_3_2, padding=0, dilation=5, groups=self.dim)
            out = self.alpha[0]*h1 + self.alpha[1]*h2 + self.alpha[2]*h3
            return out
        raise NotImplementedError("eval branch (11x11 re-param) is not ported")


worst_d = 0.0
for (hh, ww) in ((8, 8), (16, 16), (7, 7), (4, 4)):
    torch.manual_seed(3)
    ours = perturb(StateFusion(24, dilations=(1, 3, 5)), 0.3).double()
    off = OfficialStateFusion(24).double().train()
    with torch.no_grad():
        off.kernel_3.copy_(ours.kernels[0])
        off.kernel_3_1.copy_(ours.kernels[1])
        off.kernel_3_2.copy_(ours.kernels[2])
        off.alpha.copy_(ours.alpha)
    s = torch.randn(2, 24, hh, ww, dtype=torch.float64)
    with torch.no_grad():
        dd = (ours.eval()(s) - off(s)).abs().max().item()     # ours: same in eval
    assert dd < 1e-12, ((hh, ww), dd)
    worst_d = max(worst_d, dd)
fresh = StateFusion(24)
assert all(torch.equal(k, torch.ones_like(k)) for k in fresh.kernels) \
    and torch.equal(fresh.alpha, torch.ones(3))
print(f"D  StateFusion == official training branch on 8x8/16x16/7x7/4x4 grids "
      f"(max|diff| = {worst_d:.1e}); ones-init kernels and alpha as official")


# ═══ E. reduction: identity SASF -> plain unidirectional Mamba ════════════════
def plain_mamba(mx, x, Hh, Ww):
    Bb, Ll, _ = x.shape
    di = mx.d_inner
    xz = mx.in_proj(x)
    u, z = xz.chunk(2, dim=-1) if mx.gate else (xz, None)
    u = mx.act(mx.conv2d(u.reshape(Bb, Hh, Ww, di).permute(0, 3, 1, 2)))
    u = u.reshape(Bb, di, Ll)
    dbl = torch.einsum("bdl,od->bol", u, mx.x_proj_weight)
    dt_r, Bs, Cs = torch.split(dbl, [mx.dt_rank, 1, 1], dim=1)
    dts = torch.einsum("brl,dr->bdl", dt_r, mx.dt_projs_weight)
    y = selective_scan_ref(u, dts, -torch.exp(mx.A_logs.float()), Bs.unsqueeze(1),
                           Cs.unsqueeze(1), mx.Ds, delta_bias=mx.dt_projs_bias.float(),
                           delta_softplus=True)
    y = mx.out_norm(y.transpose(1, 2))
    if z is not None:
        y = y * F.silu(z)
    return mx.out_proj(y)


for gate in (True, False):
    torch.manual_seed(4)
    mx = perturb(StructureAwareSSM(32, gate=gate, dilations=(1, 3, 5))).double()
    with torch.no_grad():
        for k in mx.state_fusion.kernels:
            k.zero_()
        mx.state_fusion.kernels[0][:, 0, 1, 1] = 1.0
        mx.state_fusion.alpha.copy_(torch.tensor([1.0, 0.0, 0.0]))
        x = torch.randn(2, HW, 32, dtype=torch.float64)
        de = (mx(x, H, W) - plain_mamba(mx, x, H, W)).abs().max().item()
    assert de < 1e-12, (gate, de)
print(f"E  identity SASF reduces the mixer to plain unidirectional Mamba "
      f"(gate on/off, max|diff| = {de:.1e})")


# ═══ F. param counts ═════════════════════════════════════════════════════════
p_gate = n_params(StructureAwareSSM(384, gate=True))
p_nog = n_params(StructureAwareSSM(384, gate=False))
assert (p_gate, p_nog) == (477_699, 330_243), (p_gate, p_nog)
blk_none = n_params(JiTSpatialBlock(384, lpu="none"))
blk_zero = n_params(JiTSpatialBlock(384, lpu="zero"))
assert blk_zero - blk_none == 7_680, blk_zero - blk_none
TINY = dict(input_size=64, patch_size=8, hidden_size=384, depth=12,
            num_classes=200, bottleneck_dim=128, sasf_dilations=(1, 2, 3))
counts = {}
for gate in (True, False):
    for lpu in ("none", "zero"):
        counts[(gate, lpu)] = n_params(JiTSpatialMamba(gate=gate, lpu=lpu, **TINY))
assert counts == {(True, "none"): 31_363_428, (True, "zero"): 31_455_588,
                  (False, "none"): 29_593_956, (False, "zero"): 29_686_116}, counts
p_vm1 = n_params(JiTVMamba(input_size=64, patch_size=8, hidden_size=384, depth=12,
                           num_classes=200, bottleneck_dim=128, d_state=1, expand=1))
print(f"F  mixer/block at D=384: gate {p_gate:,} | no gate {p_nog:,} | LPU +7,680; "
      f"Tiny-IN: gate+LPU {counts[(True, 'zero')] / 1e6:.2f}M, "
      f"no-gate+LPU {counts[(False, 'zero')] / 1e6:.2f}M, "
      f"VMamba d_state=1 {p_vm1 / 1e6:.2f}M")


# ═══ G. smoke, grads, identity at init, dt range, bf16 ═══════════════════════
for gate in (True, False):
    for ffn in ("swiglu", "convglu", "glumbconv"):
        for lpu in ("none", "zero", "paper"):
            torch.manual_seed(5)
            m = JiTSpatialMamba(gate=gate, lpu=lpu, ffn=ffn,
                                sasf_dilations=(1, 2, 3), **KW)
            # identity at init (adaLN-Zero gates; LPU zero) -- before perturbing
            if lpu != "paper":
                xb = torch.randn(Bsz, HW, KW["hidden_size"])
                cb = torch.randn(Bsz, KW["hidden_size"])
                with torch.no_grad():
                    for blk in m.blocks:
                        assert torch.equal(blk(xb, cb, H, W), xb), (gate, ffn, lpu)
            perturb(m, 0.02).train()
            out = m(x_img, t_in, y_in)
            assert out.shape == x_img.shape and torch.isfinite(out).all()
            out.square().mean().backward()
            mx = m.blocks[-1].mixer
            grads = [mx.state_fusion.alpha.grad, mx.state_fusion.kernels[1].grad,
                     mx.dt_projs_bias.grad, mx.A_logs.grad]
            if lpu != "none":
                grads += [m.blocks[-1].cpe1.weight.grad, m.blocks[-1].cpe2.weight.grad]
            for g in grads:
                assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0, \
                    (gate, ffn, lpu)
torch.manual_seed(6)
m = JiTSpatialMamba(lpu="zero", **TINY)
dts = torch.cat([F.softplus(b.mixer.dt_projs_bias.detach()) for b in m.blocks])
assert dts.min().item() >= 1e-4 and dts.max().item() <= 0.1 + 1e-6, (dts.min(), dts.max())
assert all(torch.equal(b.cpe1.weight, torch.zeros_like(b.cpe1.weight)) for b in m.blocks)
m = JiTSpatialMamba(lpu="zero", sasf_dilations=(1, 2, 3), **KW)
perturb(m, 0.02)
with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
    assert torch.isfinite(m(x_img, t_in, y_in).float()).all()
print(f"G  fwd+bwd OK for gate x {{swiglu,convglu,glumbconv}} x lpu {{none,zero,paper}}; "
      f"grads reach SASF/alpha/dt/A/LPU; blocks identity at init; "
      f"softplus(dt_bias) in [{dts.min().item():.1e}, {dts.max().item():.1e}]; bf16 OK")


# ═══ H. guards ═══════════════════════════════════════════════════════════════
def must_fail(fn, what):
    try:
        fn()
    except AssertionError:
        return
    raise SystemExit(f"guard missing: {what}")


must_fail(lambda: JiTSpatialMamba(d_state=16, **KW), "d_state != 1")
must_fail(lambda: JiTSpatialMamba(lpu="cpe", **KW), "unknown lpu")
must_fail(lambda: JiTSpatialMamba(ffn="mixffn", **KW), "unknown ffn")
must_fail(lambda: StructureAwareSSM(32)(torch.randn(1, HW + 2, 32), H, W),
          "token count != H*W")
print("H  d_state != 1, unknown lpu / ffn, and a prefix (L != H*W) are rejected")


# ═══ J. drift guard: scaffold == JiTVMamba's ═════════════════════════════════
torch.manual_seed(7)
vm = perturb(JiTVMamba(d_state=1, expand=1, **KW), 0.02).eval()
sp = JiTSpatialMamba(lpu="none", **KW)
for blk in sp.blocks:
    blk.mixer = SS2D(d_model=KW["hidden_size"], d_state=1, d_conv=3, expand=1, K=4)
sp.load_state_dict(vm.state_dict(), strict=True)
sp.eval()
with torch.no_grad():
    dj = (sp(x_img, t_in, y_in) - vm(x_img, t_in, y_in)).abs().max().item()
assert dj == 0.0, dj
for ffn in ("convglu", "glumbconv"):
    torch.manual_seed(7)
    vm = perturb(JiTVMamba(d_state=1, ffn=ffn, **KW), 0.02).eval()
    sp = JiTSpatialMamba(lpu="none", ffn=ffn, **KW)
    for blk in sp.blocks:
        blk.mixer = SS2D(d_model=KW["hidden_size"], d_state=1, d_conv=3, expand=1, K=4)
    sp.load_state_dict(vm.state_dict(), strict=True)
    with torch.no_grad():
        dj = max(dj, (sp.eval()(x_img, t_in, y_in) - vm(x_img, t_in, y_in)).abs().max().item())
    assert dj == 0.0, (ffn, dj)
print("J  scaffold with SS2D swapped in loads JiTVMamba(d_state=1) strictly and is "
      "bit-identical to it (swiglu, convglu, glumbconv)")


# ═══ I. both build_model copies agree on the new configs ═════════════════════
try:
    import yaml                                             # noqa: E402
except ImportError as e:                                    # pragma: no cover
    print(f"I  SKIPPED (pyyaml not importable here: {e})")
    print("ALL CHECKS PASSED")
    sys.exit(0)


def load_build_model(rel):
    """Exec ONLY build_model out of a script (its module imports need
    torchvision/tqdm, which the CPU test env may not have)."""
    path = os.path.join(REPO_ROOT, rel)
    src = open(path, encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_model")
    ns = {}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns["build_model"]


bm_train = load_build_model("scripts/run_experiment.py")
bm_eval = load_build_model("scripts/evaluate.py")
EXPECT = {
    "jit-s2-vmamba-dstate1.yaml":  (JiTVMamba, 30_202_176),
    "jit-s2-spatial-gate.yaml":    (JiTSpatialMamba, 31_455_588),
    "jit-s2-spatial-nogate.yaml":  (JiTSpatialMamba, 29_686_116),
}
for name, (cls, n_expect) in EXPECT.items():
    cfg_path = os.path.join(REPO_ROOT, "configs", "tiny_imagenet", name)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    torch.manual_seed(0)
    mt = bm_train(cfg)
    torch.manual_seed(0)
    me = bm_eval(cfg)
    assert type(mt) is cls and type(me) is cls, (name, type(mt), type(me))
    st, se = mt.state_dict(), me.state_dict()
    assert set(st) == set(se) and all(torch.equal(st[k], se[k]) for k in st), name
    assert n_params(mt) == n_expect, (name, n_params(mt))
    if cls is JiTSpatialMamba:
        assert mt.sasf_dilations == (1, 2, 3) and mt.lpu == "zero", name
        assert mt.gate == (name == "jit-s2-spatial-gate.yaml"), name
    print(f"I  {name}: train == eval build_model, {cls.__name__}, "
          f"{n_params(mt) / 1e6:.2f}M params")

print("ALL CHECKS PASSED")
