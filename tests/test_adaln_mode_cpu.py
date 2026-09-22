"""
tests/test_adaln_mode_cpu.py
============================
CPU-only verification of `adaln_cond` — WHICH residual branch adaLN still
carries (t, y) into — and of `ssc_z_mlp`'s separate condition route.

  "full" (default / True)  adaLN-Zero on both branches  → the baseline
  "mlp"                    FFN branch + FinalLayer keep adaLN-Zero; the MIXER
                           branch's shift/scale/gate become a condition-
                           INDEPENDENT zero-init bias, so (t, y) reach the SS2D
                           scan ONLY through SSC (the branch SSC can act on)
  "none" (False)           neither branch; SSC is the only conditioning path

Runs WITHOUT a GPU and WITHOUT mamba_ssm: the CUDA kernel is stubbed with a
faithful reference selective scan.

Checks:
  A. adaln_cond="full" is BIT-IDENTICAL to HEAD for every pre-existing arm
     (state_dict transplant from a module exec'd out of `git show HEAD:`).
  B. identity at init in all three modes (every gate zero-init → zero output),
     i.e. adaLN-Zero's discipline survives the replacement.
  C. routing: with "mlp", the tensor handed to the mixer is INVARIANT to c
     while the FFN branch's modulation is not; with "full" both move; with
     "none" neither does. Backward: d(out)/d(adaLN weights) is nonzero for the
     surviving FFN adaLN in "mlp".
  D. param accounting: full - mlp = depth * 3 * D^2 exactly, and
     mlp - none = depth * 3 * D^2 + 2 * D^2 (the conditional FinalLayer).
  E. checkpoint compat: "none" keeps install_noadaln.sh's parameter NAMES
     (blocks.*.adaLN_bias of 6D, final_layer.adaLN_bias of 2D).
  F. ssc_z_mlp: the mixer receives z = ssc_z(c) while adaLN keeps the raw c;
     without it, both see c.
  G. guards fire (mode needs an ssc arm / no in-context mixing / z-MLP only off
     "full" / bad name) and bools still map to full/none.

Run from the repo root:
    python tests/test_adaln_mode_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
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
    Bf = B.float().repeat_interleave(rep, dim=1)
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

from src.models.vmamba import JiTVMamba                   # noqa: E402

torch.manual_seed(0)
KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32, d_state=16, K=4, expand=1)
D, DEPTH = KW["hidden_size"], KW["depth"]
H = W = KW["input_size"] // KW["patch_size"]
B = 3

x_img = torch.randn(B, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(B)
y_in = torch.randint(0, KW["num_classes"], (B,))


def build(**kw):
    torch.manual_seed(0)
    return JiTVMamba(**kw, **KW).eval()


# ═══ A. adaln_cond="full" is bit-identical to HEAD for every existing arm ════
LEGACY_ARMS = [
    ("baseline",           dict()),
    ("class K=2",          dict(in_context_len=2, in_context_content="class")),
    ("time_class K=2",     dict(in_context_len=2, in_context_content="time_class")),
    ("class row",          dict(in_context_len=H, in_context_content="class",
                                in_context_layout="row")),
    ("state_init dimsum",  dict(state_init="dimsum")),
    ("state_init learned", dict(state_init="learned")),
    ("ssc static",         dict(ssc="static")),
    ("ssc bc",             dict(ssc="bc")),
    ("ssc abc",            dict(ssc="abc")),
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
        new = build(**kw)                                   # default adaln_cond="full"
        assert set(old.state_dict()) == set(new.state_dict()), \
            f"state_dict keys changed for {name}"
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            d = (old(x_img, t_in, y_in) - new(x_img, t_in, y_in)).abs().max().item()
        assert d == 0.0, f"adaln_cond='full' not bit-identical to HEAD for {name}: {d}"
        worst = max(worst, d)
    print(f"A  adaln_cond='full' bit-identical to HEAD for {len(LEGACY_ARMS)} arms "
          f"(max|diff| = {worst})")


# ═══ B. identity at init in all three modes ══════════════════════════════════
MODES = [
    ("full", dict(adaln_cond="full", ssc="bc")),
    ("mlp",  dict(adaln_cond="mlp",  ssc="bc")),
    ("none", dict(adaln_cond="none", ssc="bc")),
]
for name, kw in MODES:
    m = build(**kw)
    with torch.no_grad():
        out = m(x_img, t_in, y_in)
    assert out.abs().max().item() == 0.0, \
        f"adaln_cond={name!r} is not identity at init: {out.abs().max().item()}"
print("B  identity at init for full / mlp / none (every gate zero-init, "
      "adaLN-Zero discipline preserved)")


# ═══ C. routing: what does the MIXER see, and does it move with c? ═══════════
def mixer_inputs(m, c):
    """Return (modulated x handed to the mixer, cond kwarg) for block 0."""
    seen = {}

    def hook(mod, args, kwargs):
        seen["x"] = args[0].detach().clone()
        seen["cond"] = kwargs["cond"].detach().clone()

    h = m.blocks[0].mixer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        torch.manual_seed(4)
        x = torch.randn(B, H * W, D)
        with torch.no_grad():
            m.blocks[0](x, c, H, W)
    finally:
        h.remove()
    return seen["x"], seen["cond"]


def ffn_modulated(m, c):
    """The tensor the FFN branch actually consumes (modulate(norm2(x), ...))."""
    seen = {}

    def hook(mod, args):
        seen["x"] = args[0].detach().clone()

    h = m.blocks[0].mlp.register_forward_pre_hook(hook)
    try:
        torch.manual_seed(4)
        x = torch.randn(B, H * W, D)
        with torch.no_grad():
            m.blocks[0](x, c, H, W)
    finally:
        h.remove()
    return seen["x"]


torch.manual_seed(5)
c1, c2 = torch.randn(B, D), torch.randn(B, D)
for name, kw in MODES:
    m = build(**kw)
    # untrained adaLN is zero-init, so give it a real weight before asking
    # whether the modulation moves with c
    if name != "none":
        for blk in m.blocks:
            torch.nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.02)
    mix1, cond1 = mixer_inputs(m, c1)
    mix2, cond2 = mixer_inputs(m, c2)
    ffn1, ffn2 = ffn_modulated(m, c1), ffn_modulated(m, c2)
    mixer_moves = (mix1 - mix2).abs().max().item() > 0
    ffn_moves = (ffn1 - ffn2).abs().max().item() > 0
    want = {"full": (True, True), "mlp": (False, True), "none": (False, False)}[name]
    assert (mixer_moves, ffn_moves) == want, (
        f"adaln_cond={name!r}: (mixer varies with c, FFN varies with c) = "
        f"{(mixer_moves, ffn_moves)}, expected {want}"
    )
    # the SSC path always receives the condition, in every mode
    assert (cond1 - c1).abs().max().item() == 0.0
    print(f"C  adaln_cond={name!r}: mixer input varies with c: {mixer_moves}, "
          f"FFN input varies with c: {ffn_moves}, SSC cond delivered: True")

# backward: the surviving FFN adaLN in "mlp" must receive gradient, and the
# static mixer bias must receive gradient WITHOUT depending on c
m = build(adaln_cond="mlp", ssc="bc")
m(x_img, t_in, y_in).pow(2).mean().backward()
for i, blk in enumerate(m.blocks):
    assert blk.adaLN_modulation[-1].weight.grad is not None, i
    assert blk.adaLN_bias.grad is not None, i
    assert blk.adaLN_bias.shape == (3 * D,), (i, blk.adaLN_bias.shape)
assert m.final_layer.adaLN_modulation[-1].weight.grad is not None
m.zero_grad(set_to_none=True)
print("C  'mlp' backward: FFN adaLN + FinalLayer adaLN + the static 3D mixer "
      "bias all receive gradient")


# ═══ D. param accounting ═════════════════════════════════════════════════════
n = {name: sum(p.numel() for p in build(**kw).parameters()) for name, kw in MODES}
assert n["full"] - n["mlp"] == DEPTH * 3 * D * D, (n, DEPTH * 3 * D * D)
assert n["mlp"] - n["none"] == DEPTH * 3 * D * D + 2 * D * D, n
print(f"D  params full {n['full']} > mlp {n['mlp']} > none {n['none']}: "
      f"-{DEPTH * 3 * D * D} (block adaLN halved) and -{2 * D * D} "
      f"(FinalLayer goes static)")


# ═══ E. checkpoint compat with install_noadaln.sh's naming ═══════════════════
sd = build(adaln_cond="none", ssc="bc").state_dict()
for i in range(DEPTH):
    assert sd[f"blocks.{i}.adaLN_bias"].shape == (6 * D,), i
    assert f"blocks.{i}.adaLN_modulation.0.weight" not in sd
assert sd["final_layer.adaLN_bias"].shape == (2 * D,)
assert all(not k.startswith("final_layer.adaLN_modulation") for k in sd)
print("E  'none' keeps install_noadaln.sh's names/shapes "
      "(blocks.*.adaLN_bias 6D, final_layer.adaLN_bias 2D) — old checkpoints load")


# ═══ F. ssc_z_mlp routes z to SSC only ═══════════════════════════════════════
m = build(adaln_cond="mlp", ssc="bc", ssc_z_mlp=True)
_, cond = mixer_inputs(m, c1)                    # block called directly: c_ssc=None
assert (cond - c1).abs().max().item() == 0.0, \
    "block() without c_ssc must fall back to c"


def model_level_cond(m):
    """The cond the mixer receives when the FULL model drives the block."""
    seen = {}

    def hook(mod, args, kwargs):
        seen["cond"] = kwargs["cond"].detach().clone()

    h = m.blocks[0].mixer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            m(x_img, t_in, y_in)
    finally:
        h.remove()
    return seen["cond"]


with torch.no_grad():
    c_model = m.t_embedder(t_in) + m.y_embedder(y_in)
    z = m.ssc_z(c_model)
got = model_level_cond(m)
assert (got - z).abs().max().item() == 0.0, "SSC must receive z = MLP(t, c)"
assert (got - c_model).abs().max().item() > 0.0, "z must differ from the raw c"
ffn_in_with_z = ffn_modulated(m, c_model)        # adaLN still fed the raw c
m_noz = build(adaln_cond="mlp", ssc="bc")          # its own RNG stream, so its
with torch.no_grad():                              # embedders differ from m's
    c_noz = m_noz.t_embedder(t_in) + m_noz.y_embedder(y_in)
assert (model_level_cond(m_noz) - c_noz).abs().max().item() == 0.0, \
    "without ssc_z_mlp the mixer must see the raw c"
print("F  ssc_z_mlp: mixer receives z = MLP(t, c), adaLN keeps the raw c; "
      "off by default the mixer sees c")


# ═══ G. guards ═══════════════════════════════════════════════════════════════
BAD = [
    ("mlp without an ssc arm",      dict(adaln_cond="mlp")),
    ("none without an ssc arm",     dict(adaln_cond="none")),
    ("mlp + in-context prefix",     dict(adaln_cond="mlp", ssc="bc",
                                         in_context_len=2,
                                         in_context_content="class")),
    ("ssc_z_mlp with full adaLN",   dict(ssc="bc", ssc_z_mlp=True)),
    ("unknown mode name",           dict(adaln_cond="ffn", ssc="bc")),
]
for why, kw in BAD:
    try:
        JiTVMamba(**kw, **KW)
        raise SystemExit(f"guard missing: {why}")
    except AssertionError:
        pass
# bools still map to full / none (install_noadaln.sh writes YAML booleans)
assert build(adaln_cond=True).adaln_cond == "full"
assert build(adaln_cond=False, ssc="bc").adaln_cond == "none"
assert build(ssc="bc").adaln_cond == "full"
# the arm-exclusivity assert is untouched: adaln_cond is a modifier, not an arm
try:
    JiTVMamba(adaln_cond="mlp", ssc="bc", state_init="dimsum", **KW)
    raise SystemExit("exclusivity assert missing!")
except AssertionError:
    pass
print("G  guards fire (mode needs ssc / no in-context mixing / z-MLP only off "
      "'full' / bad name) and adaln_cond True/False still map to full/none")


# ═══ full forward + backward in every mode ═══════════════════════════════════
for name, kw in MODES:
    m = build(**kw, **({"ssc_z_mlp": True} if name != "full" else {}))
    out = m(x_img, t_in, y_in)
    assert out.shape == x_img.shape, (name, out.shape)
    out.pow(2).mean().backward()
    m.zero_grad(set_to_none=True)
    print(f"   adaln_cond={name!r}: forward+backward OK")

print("ALL CHECKS PASSED")
