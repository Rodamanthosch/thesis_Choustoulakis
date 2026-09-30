"""
tests/test_ffn_cpu.py
=====================
CPU-only verification of the `ffn` flag: ConvGLU (TransNeXt) and GLUMBConv /
Mix-FFN (SANA) as drop-in replacements for the token-wise SwiGLU FFN.

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub).

Checks:
  A. ffn="swiglu" (the default) is BIT-IDENTICAL to HEAD for every existing arm,
     proven by state_dict transplant from a module exec'd out of `git show HEAD:`.
  B. repo fidelity: our modules equal verbatim transcriptions of TransNeXt's
     ConvolutionalGLU + DWConv (act_layer=nn.SiLU) and SANA's GLUMBConv +
     ConvLayer (real Conv2d-1x1 NCHW path, as wired in the Sana blocks), with
     weights copied across -- float64, max|diff| at machine epsilon.
  C. reduction: ConvGLU with an identity depthwise conv IS SwiGLUFFN.
  D. prefix handling: with P leading non-grid tokens, the grid output equals the
     FFN run on the grid alone, each prefix token equals the FFN run on it as an
     isolated 1x1 image (an independent path through the real conv), and the
     Jacobian between prefix and grid is exactly zero in both directions.
  E. param counts: +10,240 (ConvGLU) and +20,096 (GLUMBConv) per block at D=384;
     ConvGLU's depthwise init follows TransNeXt (std sqrt(2/9), zero bias).
  F. model smoke: forward + backward for each ffn x {baseline, ssc bc, adaln mlp,
     in-context prefix, in-context row}, grads reach the depthwise conv; CPU
     bf16 autocast runs with a prefix.
  G. guards: unknown ffn name fails.
  H. wiring: the two new Tiny-ImageNet configs build IDENTICAL models through
     scripts/run_experiment.py::build_model and scripts/evaluate.py::build_model.

Run from the repo root:
    PYTHONPATH=. python tests/test_ffn_cpu.py
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
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:                                    # pragma: no cover
    print(f"SKIP (torch not importable here): {e}")
    sys.exit(0)


# ── stub mamba_ssm with a faithful reference selective scan ──────────────────
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, g, n, l) grouped."""
    b, d, l = u.shape
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(delta.dtype)
    if delta_softplus:
        delta = F.softplus(delta.float())
    delta = delta.float()
    u = u.float()
    A = A.float()
    g = B.shape[1]
    rep = d // g
    Bf = B.float().repeat_interleave(rep, dim=1)     # (b, d, n, l)
    Cf = C.float().repeat_interleave(rep, dim=1)
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
        y = y + D[None, :, None].float() * u
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

from src.models.vmamba import JiTVMamba                    # noqa: E402  (after the stub)
from src.primitives import SwiGLUFFN, ConvGLUFFN, GLUMBConvFFN   # noqa: E402

torch.manual_seed(0)
KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32, d_state=16, K=4, expand=1)
H = W = KW["input_size"] // KW["patch_size"]       # 4
HW = H * W                                         # 16
B = 3

x_img = torch.randn(B, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(B)
y_in = torch.randint(0, KW["num_classes"], (B,))


def build(**kw):
    torch.manual_seed(0)
    return JiTVMamba(**kw, **KW).eval()


# ═══ A. ffn="swiglu" is bit-identical to HEAD for every existing arm ═════════
LEGACY_ARMS = [
    ("baseline",            dict()),
    ("class K=4",           dict(in_context_len=4, in_context_content="class")),
    ("time_class K=2",      dict(in_context_len=2, in_context_content="time_class")),
    ("class row",           dict(in_context_len=H, in_context_content="class",
                                 in_context_layout="row")),
    ("state_init learned",  dict(state_init="learned")),
    ("ssc static",          dict(ssc="static")),
    ("ssc bc",              dict(ssc="bc")),
    ("ssc abc",             dict(ssc="abc")),
    ("ssc bc adaln mlp",    dict(ssc="bc", adaln_cond="mlp", ssc_z_mlp=True)),
    ("ssc bc adaln none",   dict(ssc="bc", adaln_cond="none")),
]

try:
    head_src = subprocess.run(
        ["git", "show", "HEAD:src/models/vmamba.py"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8")
except Exception as e:                                      # pragma: no cover
    head_src = None
    print(f"A  SKIPPED (cannot read HEAD:src/models/vmamba.py: {e})")

if head_src is not None:
    head_mod = types.ModuleType("vmamba_head")
    exec(compile(head_src, "<HEAD:src/models/vmamba.py>", "exec"), head_mod.__dict__)
    worst = 0.0
    for name, kw in LEGACY_ARMS:
        torch.manual_seed(0)
        old = head_mod.JiTVMamba(**kw, **KW).eval()
        for new in (build(**kw), build(ffn="swiglu", **kw)):
            assert set(old.state_dict()) == set(new.state_dict()), \
                f"state_dict keys changed for {name}"
            # same RNG stream at construction -> identical weights WITHOUT a transplant
            for k, v in old.state_dict().items():
                assert torch.equal(v, new.state_dict()[k]), f"init differs: {name} {k}"
            new.load_state_dict(old.state_dict(), strict=True)
            with torch.no_grad():
                d = (old(x_img, t_in, y_in) - new(x_img, t_in, y_in)).abs().max().item()
            assert d == 0.0, f"ffn='swiglu' not bit-identical to HEAD for {name}: {d}"
            worst = max(worst, d)
    print(f"A  ffn='swiglu' bit-identical to HEAD (init AND output) for "
          f"{len(LEGACY_ARMS)} arms (max|diff| = {worst})")


# ═══ B. repo fidelity (verbatim transcriptions) ══════════════════════════════
# --- TransNeXt, classification/transnext.py L42-74 (verbatim) ---
class TNX_DWConv(nn.Module):
    def __init__(self, dim=768):
        super(TNX_DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class TNX_ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = TNX_DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# --- SANA, diffusion/model/nets/basic_modules.py ConvLayer + GLUMBConv ---
# Verbatim for the path the Sana blocks use (conv_type="2d", norm=None,
# act in {"silu", None}, HW given, no chunking); build_act/val2tuple/
# get_same_padding inlined.
def _sana_build_act(name):
    return {"silu": nn.SiLU(), None: None}[name]


class SANA_ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, dilation=1, groups=1,
                 padding=None, use_bias=False, dropout=0.0, norm=None, act="relu"):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2          # get_same_padding(kernel_size)
            padding *= dilation
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride), padding=padding,
                              dilation=(dilation, dilation), groups=groups, bias=use_bias)
        self.norm = None
        self.act = _sana_build_act(act)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class SANA_GLUMBConv(nn.Module):
    def __init__(self, in_features, hidden_features, out_feature=None, kernel_size=3,
                 stride=1, padding=None, use_bias=False, norm=(None, None, None),
                 act=("silu", "silu", None), dilation=1):
        out_feature = out_feature or in_features
        super().__init__()
        self.glu_act = _sana_build_act(act[1])
        self.inverted_conv = SANA_ConvLayer(in_features, hidden_features * 2, 1,
                                            use_bias=use_bias[0], norm=norm[0], act=act[0])
        self.depth_conv = SANA_ConvLayer(hidden_features * 2, hidden_features * 2, kernel_size,
                                         stride=stride, groups=hidden_features * 2,
                                         padding=padding, use_bias=use_bias[1], norm=norm[1],
                                         act=None, dilation=dilation)
        self.point_conv = SANA_ConvLayer(hidden_features, out_feature, 1,
                                         use_bias=use_bias[2], norm=norm[2], act=act[2])

    def _apply_spatial(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        a, g = torch.chunk(x, 2, dim=1)
        g = self.glu_act(g)
        return self.point_conv(a * g)

    def forward(self, x, HW=None):
        B, N, C = x.shape
        H, W = HW
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self._apply_spatial(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


D_FFN, HID = 48, 4 * 48            # hidden_dim as JiTBlock passes it (mlp_ratio*D)
Hg, Wg = 5, 7                      # non-square on purpose: catches H/W swaps
xg = torch.randn(2, Hg * Wg, D_FFN, dtype=torch.float64)

torch.manual_seed(1)
ours_c = ConvGLUFFN(D_FFN, HID).double().eval()
ref_c = TNX_ConvolutionalGLU(D_FFN, hidden_features=HID, act_layer=nn.SiLU).double().eval()
assert ref_c.fc1.weight.shape == ours_c.fc1.weight.shape    # same 2/3 width rule
with torch.no_grad():
    ref_c.fc1.load_state_dict(ours_c.fc1.state_dict())
    ref_c.dwconv.dwconv.load_state_dict(ours_c.dwconv.state_dict())
    ref_c.fc2.load_state_dict(ours_c.fc2.state_dict())
    for p in list(ours_c.dwconv.parameters()):                # non-trivial bias
        p.add_(0.1 * torch.randn_like(p))
    ref_c.dwconv.dwconv.load_state_dict(ours_c.dwconv.state_dict())
    d_c = (ours_c(xg, Hg, Wg) - ref_c(xg, Hg, Wg)).abs().max().item()
assert d_c < 1e-12, f"ConvGLU != TransNeXt ConvolutionalGLU: {d_c}"

torch.manual_seed(2)
ours_g = GLUMBConvFFN(D_FFN, HID).double().eval()
h_g = ours_g.point_conv.in_features
ref_g = SANA_GLUMBConv(D_FFN, h_g, use_bias=(True, True, False),
                       norm=(None, None, None), act=("silu", "silu", None)).double().eval()
with torch.no_grad():
    ref_g.inverted_conv.conv.weight.copy_(ours_g.inverted_conv.weight[:, :, None, None])
    ref_g.inverted_conv.conv.bias.copy_(ours_g.inverted_conv.bias)
    ref_g.depth_conv.conv.load_state_dict(ours_g.depth_conv.state_dict())
    ref_g.point_conv.conv.weight.copy_(ours_g.point_conv.weight[:, :, None, None])
    assert ref_g.point_conv.conv.bias is None and ours_g.point_conv.bias is None
    d_g = (ours_g(xg, Hg, Wg) - ref_g(xg, (Hg, Wg))).abs().max().item()
assert d_g < 1e-12, f"GLUMBConv != SANA GLUMBConv: {d_g}"
print(f"B  repo fidelity (float64, {Hg}x{Wg} grid): ConvGLU vs TransNeXt "
      f"max|diff| = {d_c:.1e}, GLUMBConv vs SANA max|diff| = {d_g:.1e}")


# ═══ C. ConvGLU with an identity depthwise conv IS SwiGLU ═════════════════════
torch.manual_seed(3)
swi = SwiGLUFFN(D_FFN, HID).double().eval()
cg = ConvGLUFFN(D_FFN, HID).double().eval()
with torch.no_grad():
    cg.fc1.load_state_dict(swi.w12.state_dict())
    cg.fc2.load_state_dict(swi.w3.state_dict())
    cg.dwconv.weight.zero_()
    cg.dwconv.weight[:, 0, 1, 1] = 1.0
    cg.dwconv.bias.zero_()
    d = (cg(xg, Hg, Wg) - swi(xg)).abs().max().item()
assert d < 1e-12, f"identity-DW ConvGLU != SwiGLU: {d}"
print(f"C  ConvGLU with identity DWConv == SwiGLUFFN (max|diff| = {d:.1e})")


# ═══ D. prefix tokens: centre tap only, no mixing with the grid ══════════════
P = 3
xp = torch.randn(2, P + Hg * Wg, D_FFN, dtype=torch.float64)
for name, ffn in (("convglu", ours_c), ("glumbconv", ours_g)):
    with torch.no_grad():
        out = ffn(xp, Hg, Wg)
        d_grid = (out[:, P:] - ffn(xp[:, P:], Hg, Wg)).abs().max().item()
        # each prefix token as its own zero-padded 1x1 image, through the real conv
        iso = torch.cat([ffn(xp[:, p:p + 1], 1, 1) for p in range(P)], dim=1)
        d_pre = (out[:, :P] - iso).abs().max().item()
    assert d_grid < 1e-12 and d_pre < 1e-12, (name, d_grid, d_pre)
    J = torch.autograd.functional.jacobian(lambda z: ffn(z, Hg, Wg), xp[:1])
    J = J[0, :, :, 0]                                 # (N_out, D, N_in, D)
    cross = max(J[:P, :, P:].abs().max().item(), J[P:, :, :P].abs().max().item())
    assert cross == 0.0, f"{name}: prefix <-> grid leak {cross}"
    print(f"D  {name}: grid == grid-alone ({d_grid:.1e}), prefix == isolated 1x1 "
          f"({d_pre:.1e}), prefix<->grid Jacobian exactly 0")


# ═══ E. param counts and init ════════════════════════════════════════════════
def n_params(m):
    return sum(p.numel() for p in m.parameters())


D384 = 384
p_swi = n_params(SwiGLUFFN(D384, 4 * D384))
p_cg = n_params(ConvGLUFFN(D384, 4 * D384))
p_gl = n_params(GLUMBConvFFN(D384, 4 * D384))
assert (p_swi, p_cg - p_swi, p_gl - p_swi) == (1_182_080, 10_240, 20_096), \
    (p_swi, p_cg - p_swi, p_gl - p_swi)
print(f"E  per-block FFN params at D=384: swiglu {p_swi:,} | convglu {p_cg:,} "
      f"(+{p_cg - p_swi:,}) | glumbconv {p_gl:,} (+{p_gl - p_swi:,})")

torch.manual_seed(0)
big = JiTVMamba(input_size=64, patch_size=8, hidden_size=384, depth=12,
                num_classes=200, bottleneck_dim=128, ffn="convglu")
w = torch.cat([b.mlp.dwconv.weight.flatten() for b in big.blocks])
bias = torch.cat([b.mlp.dwconv.bias for b in big.blocks])
assert abs(w.std().item() / (2.0 / 9) ** 0.5 - 1) < 0.02, w.std().item()
assert bias.abs().max().item() == 0.0
print(f"   convglu DWConv init: std {w.std().item():.4f} "
      f"(TransNeXt sqrt(2/9) = {(2.0 / 9) ** 0.5:.4f}), bias 0")
del big


# ═══ F. model smoke: forward + backward, grads reach the conv ════════════════
SMOKE_ARMS = [
    ("baseline",         dict()),
    ("ssc bc",           dict(ssc="bc")),
    ("ssc bc adaln mlp", dict(ssc="bc", adaln_cond="mlp", ssc_z_mlp=True)),
    ("class prefix K=4", dict(in_context_len=4, in_context_content="class")),
    ("class row",        dict(in_context_len=H, in_context_content="class",
                              in_context_layout="row")),
]
for ffn in ("convglu", "glumbconv"):
    conv_attr = "dwconv" if ffn == "convglu" else "depth_conv"
    for name, kw in SMOKE_ARMS:
        m = build(ffn=ffn, **kw).train()
        with torch.no_grad():               # leave adaLN-Zero's identity init so
            for p in m.parameters():         # the FFN actually receives gradient
                p.add_(0.02 * torch.randn_like(p))
        out = m(x_img, t_in, y_in)
        assert out.shape == x_img.shape and torch.isfinite(out).all(), (ffn, name)
        out.square().mean().backward()
        g = getattr(m.blocks[-1].mlp, conv_attr).weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0, (ffn, name)
    m = build(ffn=ffn, in_context_len=4, in_context_content="class")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(m(x_img, t_in, y_in).float()).all()
    print(f"F  {ffn}: forward+backward OK for {len(SMOKE_ARMS)} arms "
          f"(grads reach {conv_attr}); bf16 autocast with prefix OK")


# ═══ G. guards ═══════════════════════════════════════════════════════════════
try:
    JiTVMamba(ffn="mixffn", **KW)
    raise SystemExit("guard missing: unknown ffn")
except AssertionError:
    pass
print("G  unknown ffn name rejected")


# ═══ H. both build_model copies agree on the new configs ═════════════════════
try:
    import yaml                                             # noqa: E402
except ImportError as e:                                    # pragma: no cover
    print(f"H  SKIPPED (pyyaml not importable here: {e})")
    print("ALL CHECKS PASSED")
    sys.exit(0)


def load_build_model(rel):
    """Exec ONLY build_model out of a script (its module imports need
    torchvision/tqdm, which the CPU test env does not have)."""
    path = os.path.join(REPO_ROOT, rel)
    src = open(path, encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_model")
    ns = {}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns["build_model"]


bm_train = load_build_model("scripts/run_experiment.py")
bm_eval = load_build_model("scripts/evaluate.py")
for ffn in ("convglu", "glumbconv"):
    cfg_path = os.path.join(REPO_ROOT, "configs", "tiny_imagenet",
                            f"jit-s2-vmamba-ffn-{ffn}.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    assert cfg["model"]["ffn"] == ffn
    torch.manual_seed(0)
    mt = bm_train(cfg)
    torch.manual_seed(0)
    me = bm_eval(cfg)
    assert mt.ffn == me.ffn == ffn
    st, se = mt.state_dict(), me.state_dict()
    assert set(st) == set(se) and all(st[k].shape == se[k].shape for k in st)
    cfg["model"]["ffn"] = "swiglu"
    base = n_params(bm_train(cfg))
    print(f"H  {os.path.basename(cfg_path)}: train == eval build_model, "
          f"{n_params(mt) / 1e6:.2f}M params (swiglu {base / 1e6:.2f}M, "
          f"+{n_params(mt) - base:,})")

print("ALL CHECKS PASSED")
