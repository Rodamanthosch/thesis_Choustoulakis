"""
tests/test_incontext_row_cpu.py
===============================
CPU-only verification of in_context_layout="row" — the row/column-distributed
placement of the in-context condition tokens.

Runs WITHOUT a GPU and WITHOUT mamba_ssm: the CUDA kernel is stubbed with a
faithful reference selective scan (grouped B/C, delta_bias/softplus semantics),
which also lets us capture the exact tensor handed to the scan.

Checks:
  A. layout="prefix" is BIT-IDENTICAL to HEAD for every pre-existing arm
     (baseline / class K=1,2,4 / time_class K=2,4 / state_init / ssc), proven by
     state_dict transplant from a module exec'd out of `git show HEAD:`.
  B. placement: with layout="row" and U = in_context_len // H, condition token p
     lands at scan index (p // U) * (U + W) + (p % U) and grid token g at
     g + (g // W + 1) * U, in ALL K directions — checked against the tensor the
     scan actually receives, with the indices derived independently of the
     cat+reshape the model uses.
  C. de-interleave round trip: the grid comes back in per-direction scan order
     and the condition tokens in model order (identity-scan transplant).
  D. coverage: max scan distance from an image token back to a condition token
     drops from HW (prefix) to W (row).
  E. the row units READ THE IMAGE — d(condition token out)/d(image tokens) is
     nonzero for every unit after the first, and EXACTLY zero for every prefix
     token. This is the property the head prefix structurally cannot have.
  F. params / state_dict identical to the prefix arm at the same in_context_len.
  G. content: time_class row emits [t,y] per row; class row emits [y] per row.
  H. guards fire (bad multiples, bad time_class length, bad layout name) and the
     arm-exclusivity asserts are untouched.

Run from the repo root:
    python tests/test_incontext_row_cpu.py
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
CAPTURED = []          # every (u, L) handed to the scan, newest last


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, g, n, l) grouped."""
    CAPTURED.append(u.detach().clone())
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


def identity_scan(u, delta, A, B, C, D=None, z=None,
                  delta_bias=None, delta_softplus=False,
                  return_last_state=False):
    """y = u: the scan becomes a pass-through, so the only thing left in the
    output is the interleave/de-interleave permutation."""
    CAPTURED.append(u.detach().clone())
    return u.float()


if "mamba_ssm" not in sys.modules:
    stub = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = selective_scan_ref
    iface.selective_scan_ref = selective_scan_ref
    sys.modules["mamba_ssm"] = stub
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = iface

import src.models.vmamba as vm                    # noqa: E402  (after the stub)
from src.models.vmamba import JiTVMamba, cross_scan, cross_merge   # noqa: E402

torch.manual_seed(0)
KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32, d_state=16, K=4, expand=1)
K, D, DEPTH = 4, 64, 2
H = W = KW["input_size"] // KW["patch_size"]       # 4
HW = H * W                                         # 16
B = 3

x_img = torch.randn(B, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(B)
y_in = torch.randint(0, KW["num_classes"], (B,))


def build(**kw):
    torch.manual_seed(0)
    return JiTVMamba(**kw, **KW).eval()


# ═══ A. layout="prefix" is bit-identical to HEAD for every existing arm ══════
LEGACY_ARMS = [
    ("baseline",          dict()),
    ("class K=1",         dict(in_context_len=1, in_context_content="class")),
    ("class K=2",         dict(in_context_len=2, in_context_content="class")),
    ("class K=4",         dict(in_context_len=4, in_context_content="class")),
    ("time_class K=2",    dict(in_context_len=2, in_context_content="time_class")),
    ("time_class K=4",    dict(in_context_len=4, in_context_content="time_class")),
    ("state_init dimsum", dict(state_init="dimsum")),
    ("state_init learned", dict(state_init="learned")),
    ("ssc static",        dict(ssc="static")),
    ("ssc bc",            dict(ssc="bc")),
    ("ssc abc",           dict(ssc="abc")),
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
        new = build(**kw)                                   # default layout="prefix"
        assert set(old.state_dict()) == set(new.state_dict()), \
            f"state_dict keys changed for {name}"
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            d = (old(x_img, t_in, y_in) - new(x_img, t_in, y_in)).abs().max().item()
        assert d == 0.0, f"layout='prefix' not bit-identical to HEAD for {name}: {d}"
        worst = max(worst, d)
    print(f"A  layout='prefix' bit-identical to HEAD for {len(LEGACY_ARMS)} arms "
          f"(max|diff| = {worst})")


# ═══ shared helpers for the row-layout checks ════════════════════════════════
ROW_ARMS = [
    ("class row  U=1", dict(in_context_len=H, in_context_content="class",
                            in_context_layout="row"), 1),
    ("time_class row U=2", dict(in_context_len=2 * H, in_context_content="time_class",
                                in_context_layout="row"), 2),
]


def scan_input(mixer, z):
    """Run the mixer and return what the scan received, as (B, K, d_inner, L)."""
    CAPTURED.clear()
    mixer(z, H, W, cond=torch.randn(B, D))
    u = CAPTURED[-1]
    return u.view(B, K, mixer.d_inner, u.shape[-1])


def expected_parts(mixer, z, extra_len):
    """Recompute the seed and the per-direction grid EXACTLY as SS2D does,
    but stopping before the placement step."""
    z_all = mixer.in_proj(z)
    z_extra = z_all[:, :extra_len]                            # (B, extra, d_inner)
    z_grid = z_all[:, extra_len:]
    z2d = z_grid.view(B, H, W, mixer.d_inner).permute(0, 3, 1, 2).contiguous()
    z2d = mixer.act(mixer.conv2d(z2d))
    seed = z_extra.transpose(1, 2)                            # (B, d_inner, extra)
    return seed, cross_scan(z2d)                              # (B,K,d_inner,HW)


# ═══ B. placement + D. coverage ══════════════════════════════════════════════
for name, kw, U in ROW_ARMS:
    m = build(**kw)
    mixer = m.blocks[0].mixer
    extra_len = kw["in_context_len"]
    torch.manual_seed(1)
    z = torch.randn(B, extra_len + HW, D)

    with torch.no_grad():
        got = scan_input(mixer, z)
        seed, grid = expected_parts(mixer, z, extra_len)

    L = extra_len + HW
    assert got.shape[-1] == L, (name, got.shape, L)

    # indices derived from the formula, independently of the model's reshape
    cond_idx = [(p // U) * (U + W) + (p % U) for p in range(extra_len)]
    grid_idx = [g + (g // W + 1) * U for g in range(HW)]
    assert sorted(cond_idx + grid_idx) == list(range(L)), name

    for p, i in enumerate(cond_idx):
        for k in range(K):
            assert torch.equal(got[:, k, :, i], seed[:, :, p]), (name, "cond", p, k)
    for g, i in enumerate(grid_idx):
        assert torch.equal(got[:, :, :, i], grid[:, :, :, g]), (name, "grid", g)

    # D. coverage: distance from each image token back to its nearest condition token
    # the nearest condition token is always the LAST one of the row's unit, so
    # the bound is W regardless of U
    worst_row = max(i - max(c for c in cond_idx if c < i) for i in grid_idx)
    assert worst_row == W, (name, worst_row)

    # prefix layout: tokens occupy 0..extra_len-1, so the last grid token is HW away
    worst_prefix = HW
    print(f"B/D {name}: placement exact in all {K} directions; "
          f"max scan distance {worst_row} (prefix: {worst_prefix}), L={L}")


# ═══ C. de-interleave round trip, with the scan replaced by the identity ═════
for name, kw, U in ROW_ARMS:
    m = build(**kw)
    mixer = m.blocks[0].mixer
    extra_len = kw["in_context_len"]
    torch.manual_seed(2)
    z = torch.randn(B, extra_len + HW, D)

    vm.selective_scan_fn = identity_scan
    try:
        with torch.no_grad():
            out = mixer(z, H, W, cond=torch.randn(B, D))
            seed, grid = expected_parts(mixer, z, extra_len)
            # what SS2D should produce with a pass-through scan
            want_grid = cross_merge(grid, H, W)                  # (B, d_inner, HW)
            want_extra = seed[:, None].expand(B, K, mixer.d_inner, extra_len).sum(1)
            want = torch.cat([want_extra, want_grid], dim=-1).transpose(1, 2)
            want = mixer.out_proj(mixer.out_norm(want))
    finally:
        vm.selective_scan_fn = selective_scan_ref

    d = (out - want).abs().max().item()
    assert d < 1e-5, f"{name}: de-interleave round trip off by {d}"
    print(f"C  {name}: interleave -> identity scan -> de-interleave exact "
          f"(max|diff| = {d:.2e})")


# ═══ E. row units read the image; prefix tokens provably do not ══════════════
def image_grad_into_cond(m, extra_len):
    """max |d(condition token outputs) / d(image tokens)| per condition token."""
    mixer = m.blocks[0].mixer
    torch.manual_seed(3)
    z_cond = torch.randn(B, extra_len, D)
    z_grid = torch.randn(B, HW, D, requires_grad=True)
    out = mixer(torch.cat([z_cond, z_grid], dim=1), H, W, cond=torch.randn(B, D))
    per_token = []
    for p in range(extra_len):
        g, = torch.autograd.grad(out[:, p].sum(), z_grid, retain_graph=True,
                                 allow_unused=True)
        per_token.append(0.0 if g is None else g.abs().max().item())
    return per_token


for name, kw, U in ROW_ARMS:
    extra_len = kw["in_context_len"]
    g_row = image_grad_into_cond(build(**kw), extra_len)
    g_pre = image_grad_into_cond(
        build(in_context_len=extra_len, in_context_content="class"), extra_len)
    assert all(v == 0.0 for v in g_pre), \
        f"prefix tokens must not read the image, got {g_pre}"
    assert all(v == 0.0 for v in g_row[:U]), \
        f"the first row unit leads every scan and must not read the image: {g_row[:U]}"
    assert all(v > 0.0 for v in g_row[U:]), \
        f"every later row unit must read the image, got {g_row[U:]}"
    print(f"E  {name}: units 0..{U-1} read nothing (they lead the scan), units "
          f"{U}..{extra_len-1} read the image (min grad {min(g_row[U:]):.2e}); "
          f"prefix arm reads nothing at any position")


# ═══ F. params / state_dict identical to the prefix arm ══════════════════════
for name, kw, U in ROW_ARMS:
    extra_len = kw["in_context_len"]
    row = build(**kw)
    pre = build(in_context_len=extra_len, in_context_content="class")
    assert set(row.state_dict()) == set(pre.state_dict()), name
    assert all(row.state_dict()[k].shape == pre.state_dict()[k].shape
               for k in row.state_dict()), name
    n_row = sum(p.numel() for p in row.parameters())
    n_pre = sum(p.numel() for p in pre.parameters())
    assert n_row == n_pre, (name, n_row, n_pre)
    print(f"F  {name}: state_dict and param count identical to the prefix arm "
          f"({n_row} params)")


# ═══ G. content of the row prefix ════════════════════════════════════════════
m = build(**ROW_ARMS[1][1])                                   # time_class row
with torch.no_grad():
    t_emb, y_emb = m.t_embedder(t_in), m.y_embedder(y_in)
    ctx = m._build_prefix(t_emb, y_emb) - m.incontext_pos_embed
for j in range(H):
    assert torch.allclose(ctx[:, 2 * j], t_emb), j
    assert torch.allclose(ctx[:, 2 * j + 1], y_emb), j
m = build(**ROW_ARMS[0][1])                                   # class row
with torch.no_grad():
    t_emb, y_emb = m.t_embedder(t_in), m.y_embedder(y_in)
    ctx = m._build_prefix(t_emb, y_emb) - m.incontext_pos_embed
assert all(torch.allclose(ctx[:, j], y_emb) for j in range(H))
print(f"G  content: time_class row = [t,y] x {H}, class row = [y] x {H}")


# ═══ full model forward + backward, and the FLOPs-relevant scan length ═══════
for name, kw, U in ROW_ARMS:
    m = build(**kw)
    CAPTURED.clear()
    out = m(x_img, t_in, y_in)
    assert out.shape == x_img.shape, (name, out.shape)
    L = CAPTURED[-1].shape[-1]
    assert L == kw["in_context_len"] + HW, (name, L)
    out.mean().backward()
    assert m.incontext_pos_embed.grad is not None and \
        m.incontext_pos_embed.grad.abs().max() >= 0.0
    m.zero_grad(set_to_none=True)
    print(f"   {name}: forward+backward OK, scan L = {L} (grid {HW} + "
          f"{kw['in_context_len']} condition tokens)")


# ═══ H. guards ═══════════════════════════════════════════════════════════════
BAD = [
    ("len not a multiple of grid_size",
     dict(in_context_len=H + 1, in_context_content="class", in_context_layout="row")),
    ("time_class row with the wrong length",
     dict(in_context_len=H, in_context_content="time_class", in_context_layout="row")),
    ("row layout with no tokens",
     dict(in_context_len=0, in_context_layout="row")),
]
for why, kw in BAD:
    try:
        JiTVMamba(**kw, **KW)
        raise SystemExit(f"guard missing: {why}")
    except AssertionError:
        pass
try:
    JiTVMamba(in_context_len=H, in_context_layout="rows", **KW)
    raise SystemExit("guard missing: unknown layout name")
except ValueError:
    pass
# arm exclusivity is untouched by the new flag
for bad_kw in (dict(state_init="dimsum"), dict(ssc="bc")):
    try:
        JiTVMamba(in_context_len=H, in_context_content="class",
                  in_context_layout="row", **bad_kw, **KW)
        raise SystemExit(f"exclusivity assert missing for {bad_kw}!")
    except AssertionError:
        pass
print("H  guards fire (bad multiple / bad time_class length / len=0 / bad name) "
      "and arm exclusivity still holds")

print("ALL CHECKS PASSED")
