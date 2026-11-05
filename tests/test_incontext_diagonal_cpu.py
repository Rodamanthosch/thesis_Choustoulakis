"""
tests/test_incontext_diagonal_cpu.py
====================================
CPU-only verification of in_context_layout="diagonal" -- in-context class
tokens tied to the grid cells (e*s, e*s) and permuted TOGETHER with the image
in SS2D's four scan paths (before the cell in paths 0/2, after it in the
reversed paths 1/3), as Vim / Mamba-R's flip does for its registers.

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, which
also captures the exact tensor handed to the scan).

Checks:
  A. layout="prefix" and layout="row" are BIT-IDENTICAL to HEAD for every
     pre-existing arm, by state_dict transplant from `git show HEAD:`.
  B. the gather machinery with 0 tokens IS cross_scan / cross_merge, bit for bit.
  C. placement: every token and every grid cell lands where an independent
     derivation from cross_scan's four orders says (token right before its cell
     in paths 0/2, right after it in paths 1/3), for n = H and n = H/2.
  D. identity-scan round trip: gather -> scan -> inverse gather returns
     [K * tokens, cross_merge(grid)].
  E. coverage, with the DWConv set to the identity so dependencies are per
     cell: every diagonal token READS every cell and WRITES to every cell in
     one block; a row token j reads exactly the border band of width j; a
     prefix token reads nothing.
  F. params / state_dict identical to the prefix arm at the same length.
  G. full model: forward + backward, scan length per block (tokens enter at
     in_context_start and persist to the last block).
  H. guards (length must divide the grid, class content only, len > 0, layout
     name) and arm exclusivity.

Run from the repo root:
    PYTHONPATH=. python tests/test_incontext_diagonal_cpu.py
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
CAPTURED = []          # every u handed to the scan, newest last


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


def identity_scan(u, delta, A, B, C, D=None, z=None,
                  delta_bias=None, delta_softplus=False,
                  return_last_state=False):
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
from src.models.vmamba import (                   # noqa: E402
    JiTVMamba, cross_scan, cross_merge, diagonal_cells, diagonal_scan_index)

torch.manual_seed(0)
KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32, d_state=16, K=4, expand=1)
K, D = 4, 64
H = W = KW["input_size"] // KW["patch_size"]       # 4
HW = H * W                                         # 16
B = 3

x_img = torch.randn(B, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(B)
y_in = torch.randint(0, KW["num_classes"], (B,))


def build(**kw):
    torch.manual_seed(0)
    return JiTVMamba(**kw, **KW).eval()


def diag(n):
    return dict(in_context_len=n, in_context_content="class",
                in_context_layout="diagonal")


# ═══ A. prefix and row are bit-identical to HEAD ═════════════════════════════
LEGACY_ARMS = [
    ("baseline",           dict()),
    ("class K=1",          dict(in_context_len=1, in_context_content="class")),
    ("class K=4",          dict(in_context_len=4, in_context_content="class")),
    ("time_class K=2",     dict(in_context_len=2, in_context_content="time_class")),
    ("time_class K=4",     dict(in_context_len=4, in_context_content="time_class")),
    ("class row",          dict(in_context_len=H, in_context_content="class",
                                in_context_layout="row")),
    ("time_class row",     dict(in_context_len=2 * H, in_context_content="time_class",
                                in_context_layout="row")),
    ("state_init dimsum",  dict(state_init="dimsum")),
    ("state_init learned", dict(state_init="learned")),
    ("ssc static",         dict(ssc="static")),
    ("ssc bc",             dict(ssc="bc")),
    ("ssc abc",            dict(ssc="abc")),
    ("ssm_norm bcdt + class K=4", dict(ssm_norm="bcdt", in_context_len=4,
                                       in_context_content="class")),
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
        new = build(**kw)
        assert set(old.state_dict()) == set(new.state_dict()), name
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            d = (old(x_img, t_in, y_in) - new(x_img, t_in, y_in)).abs().max().item()
        assert d == 0.0, f"{name} not bit-identical to HEAD: {d}"
        worst = max(worst, d)
    print(f"A  prefix / row bit-identical to HEAD for {len(LEGACY_ARMS)} arms "
          f"(max|diff| = {worst})")


# ═══ B. 0 tokens: the gather IS cross_scan / cross_merge ═════════════════════
idx0, inv0 = diagonal_scan_index(H, W, 0, torch.device("cpu"))
g = torch.randn(B, 5, H, W)
xs_gather = g.reshape(B, 5, HW)[:, :, idx0].transpose(1, 2)
assert torch.equal(xs_gather, cross_scan(g)), "gather != cross_scan"
ys = torch.randn(B, K, 5, HW)
merged = ys[:, 0][..., inv0[0]]
for k in range(1, K):
    merged = merged + ys[:, k][..., inv0[k]]
assert torch.equal(merged, cross_merge(ys, H, W)), "inverse gather != cross_merge"
print("B  0 tokens: gather == cross_scan and inverse gather == cross_merge, bit for bit")


# ═══ shared helpers ══════════════════════════════════════════════════════════
def scan_input(mixer, z):
    CAPTURED.clear()
    mixer(z, H, W, cond=torch.randn(B, D))
    u = CAPTURED[-1]
    return u.view(B, K, mixer.d_inner, u.shape[-1])


def expected_parts(mixer, z, n):
    """Seed and per-path grid exactly as SS2D computes them, before placement."""
    z_all = mixer.in_proj(z)
    z2d = z_all[:, n:].view(B, H, W, mixer.d_inner).permute(0, 3, 1, 2).contiguous()
    z2d = mixer.act(mixer.conv2d(z2d))
    return z_all[:, :n].transpose(1, 2), cross_scan(z2d)    # (B,d,n), (B,K,d,HW)


def expected_sequence(n):
    """Per path: list of ('t', e) / ('g', s) where s indexes cross_scan's path-k
    sequence. Derived from cross_scan's orders, NOT from diagonal_scan_index."""
    cells = diagonal_cells(H, n)
    pos = [lambda r, c: r * W + c,                 # path 0: row-major
           lambda r, c: HW - 1 - (r * W + c),      # path 1: reversed
           lambda r, c: c * H + r,                 # path 2: column-major
           lambda r, c: HW - 1 - (c * H + r)]      # path 3: reversed
    seqs = []
    for k in range(K):
        before = {pos[k](r, c): e for e, (r, c) in enumerate(cells)}
        seq = []
        for s in range(HW):
            if k in (0, 2) and s in before:
                seq.append(("t", before[s]))       # token just before its cell
            seq.append(("g", s))
            if k in (1, 3) and s in before:
                seq.append(("t", before[s]))       # reversed: just after it
        seqs.append(seq)
    return seqs


# ═══ C. placement ════════════════════════════════════════════════════════════
for n in (H, H // 2, 1):
    mixer = build(**diag(n)).blocks[0].mixer
    torch.manual_seed(1)
    z = torch.randn(B, n + HW, D)
    with torch.no_grad():
        got = scan_input(mixer, z)
        seed, grid = expected_parts(mixer, z, n)
    assert got.shape[-1] == n + HW
    for k, seq in enumerate(expected_sequence(n)):
        assert len(seq) == n + HW
        for i, (kind, j) in enumerate(seq):
            want = seed[:, :, j] if kind == "t" else grid[:, k, :, j]
            assert torch.equal(got[:, k, :, i], want), (n, k, i, kind, j)
    # scan distance from each image token back to the nearest preceding token
    gaps = []
    for seq in expected_sequence(n):
        last, worst, none = None, 0, 0
        for i, (kind, _) in enumerate(seq):
            if kind == "t":
                last = i
            elif last is None:
                none += 1
            else:
                worst = max(worst, i - last)
        gaps.append((worst, none))
    print(f"C  n={n} cells {diagonal_cells(H, n)}: placement exact in all {K} paths; "
          f"(max distance back to a token, #cells with none before) per path = {gaps}")


# ═══ D. identity-scan round trip ═════════════════════════════════════════════
for n in (H, H // 2):
    mixer = build(**diag(n)).blocks[0].mixer
    torch.manual_seed(2)
    z = torch.randn(B, n + HW, D)
    vm.selective_scan_fn = identity_scan
    try:
        with torch.no_grad():
            out = mixer(z, H, W, cond=torch.randn(B, D))
            seed, grid = expected_parts(mixer, z, n)
            want = torch.cat([seed * K, cross_merge(grid, H, W)], -1).transpose(1, 2)
            want = mixer.out_proj(mixer.out_norm(want))
    finally:
        vm.selective_scan_fn = selective_scan_ref
    d = (out - want).abs().max().item()
    assert d < 1e-5, f"n={n}: round trip off by {d}"
    print(f"D  n={n}: gather -> identity scan -> inverse gather exact "
          f"(max|diff| = {d:.2e})")


# ═══ E. who reads / writes whom (DWConv set to the identity) ═════════════════
def per_cell_mixer(**kw):
    mixer = build(**kw).blocks[0].mixer
    with torch.no_grad():
        mixer.conv2d.weight.zero_()
        mixer.conv2d.weight[:, :, mixer.d_conv // 2, mixer.d_conv // 2] = 1.0
        mixer.conv2d.bias.zero_()
    return mixer


def reads(mixer, n):
    """(n, HW) bool: token e's output depends on image cell g."""
    torch.manual_seed(3)
    z_tok = torch.randn(B, n, D)
    z_grid = torch.randn(B, HW, D, requires_grad=True)
    out = mixer(torch.cat([z_tok, z_grid], 1), H, W, cond=torch.randn(B, D))
    rows = []
    for e in range(n):
        gr, = torch.autograd.grad(out[:, e].pow(2).sum(), z_grid, retain_graph=True)
        rows.append(gr.abs().amax(dim=(0, 2)) > 0)
    return torch.stack(rows)


def writes(mixer, n):
    """(n, HW) bool: image cell g's output depends on token e."""
    torch.manual_seed(4)
    z_tok = torch.randn(B, n, D, requires_grad=True)
    z_grid = torch.randn(B, HW, D)
    out = mixer(torch.cat([z_tok, z_grid], 1), H, W, cond=torch.randn(B, D))
    cols = []
    for gcell in range(HW):
        gr, = torch.autograd.grad(out[:, n + gcell].pow(2).sum(), z_tok, retain_graph=True)
        cols.append(gr.abs().amax(dim=(0, 2)) > 0)
    return torch.stack(cols, dim=1)


for n in (H, H // 2):
    m = per_cell_mixer(**diag(n))
    assert reads(m, n).all(), f"diagonal n={n}: some token misses some cell"
    assert writes(m, n).all(), f"diagonal n={n}: some cell misses some token"
    print(f"E  diagonal n={n}: every token reads all {HW} cells and writes to all "
          f"{HW} cells in one block")

m_row = per_cell_mixer(in_context_len=H, in_context_content="class",
                       in_context_layout="row")
r_row, w_row = reads(m_row, H), writes(m_row, H)
for j in range(H):
    # token j leads row j (path 0), row H-1-j (1), column j (2), column W-1-j (3)
    band = torch.tensor([(r < j) or (r > H - 1 - j) or (c < j) or (c > W - 1 - j)
                         for r in range(H) for c in range(W)])
    reach = torch.tensor([(r >= j) or (r <= H - 1 - j) or (c >= j) or (c <= W - 1 - j)
                          for r in range(H) for c in range(W)])
    assert torch.equal(r_row[j], band), (j, r_row[j], band)
    assert torch.equal(w_row[j], reach), (j, w_row[j], reach)
r_pre = reads(per_cell_mixer(in_context_len=H, in_context_content="class"), H)
assert not r_pre.any(), "prefix tokens must read nothing"
print(f"E  row: token j reads exactly the border band of width j and writes to the "
      f"cells after it in each path (cells read {r_row.sum(1).tolist()}, written "
      f"{w_row.sum(1).tolist()}); prefix: reads none")


# ═══ F. params / state_dict identical to the prefix arm ══════════════════════
for n in (H, H // 2):
    a, b = build(**diag(n)), build(in_context_len=n, in_context_content="class")
    assert {k: v.shape for k, v in a.state_dict().items()} == \
           {k: v.shape for k, v in b.state_dict().items()}, n
    a.load_state_dict(b.state_dict(), strict=True)
    print(f"F  n={n}: state_dict identical to the class prefix arm "
          f"({sum(p.numel() for p in a.parameters())} params)")


# ═══ G. full model: forward, backward, scan length per block ═════════════════
for n, start in ((H, 1), (H // 2, 0)):
    m = build(**diag(n), in_context_start=start).train()
    CAPTURED.clear()
    out = m(x_img, t_in, y_in)
    assert out.shape == x_img.shape
    Ls = [u.shape[-1] for u in CAPTURED]
    want = [HW] * start + [HW + n] * (KW["depth"] - start)
    assert Ls == want, (Ls, want)
    out.pow(2).mean().backward()
    assert m.incontext_pos_embed.grad is not None
    print(f"G  n={n}, start={start}: forward+backward OK, scan L per block {Ls}")


# ═══ H. guards ═══════════════════════════════════════════════════════════════
BAD = [
    ("length does not divide the grid", diag(3)),
    ("time_class content", dict(in_context_len=H, in_context_content="time_class",
                                in_context_layout="diagonal")),
    ("no tokens", dict(in_context_len=0, in_context_layout="diagonal")),
    ("with state_init", dict(diag(H), state_init="dimsum")),
    ("with ssc", dict(diag(H), ssc="bc")),
]
for why, kw in BAD:
    try:
        JiTVMamba(**kw, **KW)
        raise SystemExit(f"guard missing: {why}")
    except AssertionError:
        pass
try:
    JiTVMamba(in_context_len=H, in_context_layout="diag", **KW)
    raise SystemExit("guard missing: unknown layout name")
except ValueError:
    pass
print("H  guards fire (length / content / len=0 / exclusivity / layout name)")

print("ALL CHECKS PASSED")
