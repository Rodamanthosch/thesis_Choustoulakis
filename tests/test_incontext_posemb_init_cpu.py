"""
tests/test_incontext_posemb_init_cpu.py
=======================================
CPU-only verification of the in-context positional-slot init fix:
`incontext_pos_embed` is now N(0, 0.02) (LTH14/JiT model_jit.py:249-250)
instead of zeros, in jit.py, vim.py and vmamba.py.

Runs WITHOUT a GPU and WITHOUT mamba_ssm / causal_conv1d: both kernels are
stubbed with faithful references.

The reference is the fix's parent commit, PRE_FIX (zero init), not HEAD:
once the fix is committed HEAD has the new init too.

Checks:
  A. in_context_len = 0 is BIT-IDENTICAL to PRE_FIX for all three models at
     construction time (same seed -> every state_dict tensor equal), so no
     baseline run changes, not even its random init.
  B. in-context arms: only the init changed -- PRE_FIX weights transplanted
     into the new model give a bit-identical forward, with identical
     state_dict keys.
  C. the new slots look like N(0, 0.02) and are pairwise distinct.
  D. why it matters (attention JiT): with zero slots the K in-context tokens
     leave the last block identical and get identical gradients, so they can
     never separate; with the N(0, 0.02) slots they differ in both.

Run from the repo root:
    python tests/test_incontext_posemb_init_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
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


# ── stub mamba_ssm / causal_conv1d with faithful references ──────────────────
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, n, l) or (b, g, n, l)."""
    b, d, l = u.shape
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(delta.dtype)
    if delta_softplus:
        delta = F.softplus(delta.float())
    delta = delta.float()
    u = u.float()
    A = A.float()
    if B.dim() == 3:
        B, C = B.unsqueeze(1), C.unsqueeze(1)
    rep = d // B.shape[1]
    Bf = B.float().repeat_interleave(rep, dim=1)     # (b, d, n, l)
    Cf = C.float().repeat_interleave(rep, dim=1)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for i in range(l):
        dAi = torch.exp(delta[:, :, i].unsqueeze(-1) * A[None])
        dBu = delta[:, :, i].unsqueeze(-1) * Bf[:, :, :, i] * u[:, :, i].unsqueeze(-1)
        h = dAi * h + dBu
        ys.append((h * Cf[:, :, :, i]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D[None, :, None].float() * u
    if z is not None:
        y = y * F.silu(z.float())
    return y


def causal_conv1d_ref(x, weight, bias=None, activation=None):
    """x: (b, d, l)  weight: (d, k) -> depthwise causal conv, optional SiLU."""
    k = weight.shape[-1]
    out = F.conv1d(F.pad(x, (k - 1, 0)), weight.unsqueeze(1), bias, groups=x.shape[1])
    return F.silu(out) if activation in ("silu", "swish") else out


if "mamba_ssm" not in sys.modules:
    stub = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = selective_scan_ref
    iface.selective_scan_ref = selective_scan_ref
    sys.modules["mamba_ssm"] = stub
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = iface
if "causal_conv1d" not in sys.modules:
    cc = types.ModuleType("causal_conv1d")
    cc.causal_conv1d_fn = causal_conv1d_ref
    sys.modules["causal_conv1d"] = cc

from src.models.jit import JiT                     # noqa: E402  (after the stubs)
from src.models.vim import JiTViM                  # noqa: E402
from src.models.vmamba import JiTVMamba            # noqa: E402

COMMON = dict(input_size=16, patch_size=4, hidden_size=64, num_classes=7,
              bottleneck_dim=32)
MODELS = {
    # name: (path, class name, new class, constructor kwargs)
    "jit":    ("src/models/jit.py", "JiT", JiT,
               dict(COMMON, depth=4, num_heads=4)),
    "vim":    ("src/models/vim.py", "JiTViM", JiTViM,
               dict(COMMON, depth=2, d_state=8, expand=1)),
    "vmamba": ("src/models/vmamba.py", "JiTVMamba", JiTVMamba,
               dict(COMMON, depth=2, d_state=8, K=4, expand=1)),
}
ARMS = {
    "jit":    [dict(in_context_len=32, in_context_start=2),
               dict(in_context_len=4, in_context_start=0)],
    "vim":    [dict(in_context_len=1), dict(in_context_len=4, in_context_start=1)],
    "vmamba": [dict(in_context_len=32, in_context_content="class"),
               dict(in_context_len=2, in_context_content="time_class")],
}

torch.manual_seed(0)
Bsz = 3
x_img = torch.randn(Bsz, 3, COMMON["input_size"], COMMON["input_size"])
t_in = torch.rand(Bsz)
y_in = torch.randint(0, COMMON["num_classes"], (Bsz,))


PRE_FIX = "b59d6f0"     # parent of d64c543, the commit that introduced the N(0, 0.02) init


def pre_fix_class(path, cls_name):
    src = subprocess.run(["git", "show", f"{PRE_FIX}:{path}"], cwd=REPO_ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    mod = types.ModuleType(f"prefix_{cls_name}")
    exec(compile(src, f"<{PRE_FIX}:{path}>", "exec"), mod.__dict__)
    return getattr(mod, cls_name)


def build(cls, **kw):
    torch.manual_seed(0)
    return cls(**kw).eval()


OLD = {name: pre_fix_class(path, cls_name)
       for name, (path, cls_name, _, _) in MODELS.items()}


# ═══ A. in_context_len = 0: init and forward bit-identical to PRE_FIX ═══════════
for name, (_, _, cls, kw) in MODELS.items():
    old, new = build(OLD[name], **kw), build(cls, **kw)
    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn), f"{name}: state_dict keys changed"
    for k in so:
        assert torch.equal(so[k], sn[k]), f"{name}: init of {k} changed at in_context_len=0"
    with torch.no_grad():
        d = (old(x_img, t_in, y_in) - new(x_img, t_in, y_in)).abs().max().item()
    assert d == 0.0, f"{name}: baseline forward changed: {d}"
print(f"A  in_context_len=0 bit-identical to PRE_FIX (init + forward) for "
      f"{', '.join(MODELS)}")


# ═══ B. in-context arms: only the init changed ═══════════════════════════════
n_arms = 0
for name, (_, _, cls, kw) in MODELS.items():
    for arm in ARMS[name]:
        old, new = build(OLD[name], **kw, **arm), build(cls, **kw, **arm)
        assert list(old.state_dict()) == list(new.state_dict()), \
            f"{name} {arm}: state_dict keys changed"
        assert not torch.equal(old.incontext_pos_embed, new.incontext_pos_embed)
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            d = (old(x_img, t_in, y_in) - new(x_img, t_in, y_in)).abs().max().item()
        assert d == 0.0, f"{name} {arm}: forward changed beyond init: {d}"
        n_arms += 1
print(f"B  {n_arms} in-context arms: {PRE_FIX} weights transplanted -> forward "
      f"bit-identical (max|diff| = 0.0)")


# ═══ C. new slots ~ N(0, 0.02), pairwise distinct ════════════════════════════
for name, (_, _, cls, kw) in MODELS.items():
    for arm in ARMS[name]:
        old = build(OLD[name], **kw, **arm)
        assert torch.count_nonzero(old.incontext_pos_embed) == 0   # PRE_FIX: zeros
        pe = build(cls, **kw, **arm).incontext_pos_embed.detach()[0]   # (K, D)
        std, mean = pe.std().item(), pe.mean().item()
        assert 0.015 < std < 0.025, f"{name} {arm}: std {std}"
        assert abs(mean) < 0.005, f"{name} {arm}: mean {mean}"
        if pe.shape[0] > 1:
            gap = torch.cdist(pe, pe) + torch.eye(pe.shape[0]) * 1e9
            assert gap.min().item() > 0.1, f"{name} {arm}: two slots coincide"
big = build(JiT, **MODELS["jit"][3], **ARMS["jit"][0]).incontext_pos_embed
print(f"C  slots ~ N(0, 0.02) and pairwise distinct in all arms "
      f"(jit K=32: std {big.std().item():.4f}, mean {big.mean().item():+.5f})")


# ═══ D. the symmetry argument, on attention JiT ═══════════════════════════════
def slot_spread(model, K):
    """Spread across the K in-context slots of (a) the tokens leaving the last
    block and (b) d loss / d incontext_pos_embed, both relative to their scale."""
    caught = {}
    h = model.blocks[-1].register_forward_hook(
        lambda m, i, o: caught.__setitem__("out", o[:, :K].detach()))
    model.zero_grad()
    model(x_img, t_in, y_in).pow(2).mean().backward()
    h.remove()
    out, g = caught["out"], model.incontext_pos_embed.grad[0]
    spread_out = ((out - out[:, :1]).abs().max() / out.abs().max()).item()
    spread_g = ((g - g[:1]).abs().max() / g.abs().max()).item()
    return spread_out, spread_g


K = ARMS["jit"][0]["in_context_len"]
trained = build(JiT, **MODELS["jit"][3], **ARMS["jit"][0]).train()
# A fresh JiT is identity blocks + a zero head (adaLN-Zero), so every gradient
# into the blocks is 0. Stand in for a mid-training model: give every other
# parameter generic non-zero values, then compare the two slot inits on it.
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    for n, p in trained.named_parameters():
        if p.requires_grad and n != "incontext_pos_embed":
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
zero_init = copy.deepcopy(trained)
with torch.no_grad():
    zero_init.incontext_pos_embed.zero_()

so_old, sg_old = slot_spread(zero_init, K)
so_new, sg_new = slot_spread(trained, K)
assert so_old < 1e-5 and sg_old < 1e-5, (so_old, sg_old)
assert so_new > 1e-2 and sg_new > 1e-2, (so_new, sg_new)
print(f"D  jit K={K}, slot spread (relative)   output     grad")
print(f"     zero slots (old)             {so_old:9.2e} {sg_old:9.2e}   -> K copies of one token")
print(f"     N(0, 0.02) slots (now)       {so_new:9.2e} {sg_new:9.2e}")

print("\nALL CHECKS PASSED")
