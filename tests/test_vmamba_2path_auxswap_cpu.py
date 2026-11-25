"""
tests/test_vmamba_2path_auxswap_cpu.py
======================================
CPU-only verification of the 2-path SS2D (model.scan_paths [0, 2]) and the
SF-Mamba auxiliary tokens on it (model.aux_tokens swap | noswap) in
src/models/vmamba.py (Yoshimura et al., ICML 2026, arXiv 2603.16423; official
sony/SF-Mamba HybridStage.forward).

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, kept
in float64 when fed float64).

Checks:
  A. defaults (scan_paths [0,1,2,3], aux_tokens none) are BIT-IDENTICAL to
     commit fe955ad (pinned; SCAN_PATHS_REFERENCE=<rev> overrides): init and
     perturbed output of JiTVMamba under every arm / modifier, JiTVMambaSASF
     and JiTVMambaGlobal built on the reference vmamba.py.
  B. cross_scan_paths / cross_merge_paths: (0, 2) == cross_scan's paths 0 and 2
     and == cross_merge with paths 1 / 3 zeroed, bit for bit; the round trip is
     2x; with the four paths both are cross_scan / cross_merge exactly.
  C. the [0, 2] mixer without tokens == a naive float64 transcription (cell
     orders written out, per-path recurrence with that path's x_proj / dt_proj /
     A / D, outputs scattered back to their cells and summed, LN, out_proj), on
     a rectangular grid.
  D. the same with the two aux tokens: head scanned first and tail last in both
     paths, token input in_proj -> centre tap + bias -> SiLU, token outputs
     summed over the paths, rows returned as [head, tail, grid]; runs under CPU
     bf16 autocast.
  E. placement / causality on the mixer and on a whole JiTBlock (float64):
     the tail's input changes no grid output and not the head (exactly); the
     head's input reaches every grid cell; every grid cell reaches the tail;
     the grid does not reach the head.
  F. swap / noswap == hand-written loops over the blocks (mean init at
     aux_tokens_start, swaps after blocks start .. depth-2, strip); the tokens
     enter as the grid mean; swap != noswap; start = depth-1 -> no swap, the
     two modes are equal exactly.
  G. params: the 2-path model has 2 x per-path x depth fewer (toy size);
     aux_tokens adds 0, and the 2-path / swap / noswap models load into each
     other strictly.
  H. guards: K != len(scan_paths), unsupported scan_paths, aux tokens with the
     four paths, with every arm / modifier, a bad aux_tokens_start or mode, and
     the 2-path model with any arm / modifier; SS2D-level guards.
  I. wiring (needs pyyaml): both build_model copies build IDENTICAL models from
     jit-s2-vmamba-2path(-auxswap).yaml, 30,128,448 params = baseline
     31,031,616 - 903,168; profile_model builds the 2-path config.

Run from the repo root:
    PYTHONPATH=. python tests/test_vmamba_2path_auxswap_cpu.py
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
    import torch.nn.functional as F
except ImportError as e:                                    # pragma: no cover
    print(f"SKIP (torch not importable here): {e}")
    sys.exit(0)


# ── stub mamba_ssm: Mamba-1 selective scan as a recurrence (keeps float64) ───
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, g, n, l) grouped."""
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

from src.models.vmamba import (                              # noqa: E402  (after the stub)
    JiTVMamba, JiTBlock, SS2D, cross_scan, cross_merge, cross_scan_paths,
    cross_merge_paths)
from src.models.vmamba_sasf import JiTVMambaSASF             # noqa: E402
from src.models.vmamba_global import JiTVMambaGlobal         # noqa: E402

REFERENCE = os.environ.get("SCAN_PATHS_REFERENCE", "fe955ad")
KW = dict(input_size=16, patch_size=4, hidden_size=32, depth=4, num_classes=7,
          bottleneck_dim=16)
D, DEPTH, N = KW["hidden_size"], KW["depth"], 4
GRID = KW["input_size"] // KW["patch_size"]                 # 4
VM = dict(d_state=N)
TWO = dict(VM, K=2, scan_paths=(0, 2))
B = 3
torch.manual_seed(0)
x_img = torch.randn(B, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(B)
y_in = torch.randint(0, KW["num_classes"], (B,))


def build(cls, **kw):
    torch.manual_seed(0)
    return cls(**dict(KW, **kw)).eval()


def perturb(model, seed=1, scale=0.1):
    """Seeded noise on every parameter, so adaLN-Zero's zero output can't hide
    a difference."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def to64(m):
    """Whole model in float64 (the fp32 sin-cos table cast before the MLP)."""
    m.double()
    te = m.t_embedder
    te.forward = lambda t: te.mlp(te.timestep_embedding(t, te.frequency_embedding_size).double())
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fails(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return True
    return False


# ═══ A. the defaults are bit-identical to the reference commit ═══════════════
def ref_module(rel, name, overrides=None):
    try:
        src = subprocess.run(["git", "show", f"{REFERENCE}:{rel}"], cwd=REPO_ROOT,
                             capture_output=True, check=True).stdout.decode("utf-8")
    except Exception as e:                                  # pragma: no cover
        print(f"   (cannot read {REFERENCE}:{rel}: {e})")
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
    ("in-context prefix time_class from block 1", dict(VM, in_context_len=2, in_context_start=1)),
    ("in-context row", dict(VM, in_context_len=4, in_context_content="class",
                            in_context_layout="row")),
    ("in-context diagonal", dict(VM, in_context_len=2, in_context_content="class",
                                 in_context_layout="diagonal", in_context_start=1)),
    ("ssc abc", dict(VM, ssc="abc")),
    ("ssc bc, adaln_cond none, z mlp", dict(VM, ssc="bc", adaln_cond="none", ssc_z_mlp=True)),
    ("ssc static, adaln_cond mlp", dict(VM, ssc="static", adaln_cond="mlp")),
    ("state-init learned", dict(VM, state_init="learned")),
    ("state-init dimsum", dict(VM, state_init="dimsum")),
    ("ssm_norm bcdt", dict(VM, ssm_norm="bcdt")),
    ("feature_rope 2d_axial", dict(VM, feature_rope="2d_axial")),
    ("a_tie + bc_rope", dict(VM, a_tie="pairs", bc_rope="2d_axial")),
    ("glumbconv + derf", dict(VM, ffn="glumbconv", norm="derf")),
    ("convglu", dict(VM, ffn="convglu")),
    ("d_state 1", dict(d_state=1)),
]
n_checked = 0
if ref_vm is None:
    print("A  SKIPPED (no reference commit in this clone)")
else:
    for label, kw in CASES:
        same_as_ref(ref_vm.JiTVMamba, JiTVMamba, kw, f"JiTVMamba {label}")
        n_checked += 1
    over = {"src.models.vmamba": ref_vm}
    ref_sasf = ref_module("src/models/vmamba_sasf.py", "ref_vmamba_sasf", over)
    same_as_ref(ref_sasf.JiTVMambaSASF, JiTVMambaSASF,
                dict(d_state=1, sasf_dilations=(1, 2)), "JiTVMambaSASF")
    ref_glob = ref_module("src/models/vmamba_global.py", "ref_vmamba_global", over)
    for mode in ("linear", "softmax"):
        same_as_ref(ref_glob.JiTVMambaGlobal, JiTVMambaGlobal,
                    dict(VM, global_len=2, global_start=1, global_kernel_xe=mode),
                    f"JiTVMambaGlobal {mode}")
    n_checked += 3
    print(f"A  scan_paths / aux_tokens defaults bit-identical to {REFERENCE} for "
          f"{n_checked} model/arm builds (init + perturbed output, max|diff| = 0)")


# ═══ B. the path helpers ═════════════════════════════════════════════════════
torch.manual_seed(2)
xg = torch.randn(2, 3, 4, 5, dtype=torch.float64)          # rectangular H=4, W=5
full = cross_scan(xg)
assert torch.equal(cross_scan_paths(xg, (0, 1, 2, 3)), full)
assert torch.equal(cross_scan_paths(xg, (0, 2)), full[:, [0, 2]])
ys2 = torch.randn(2, 2, 3, 20, dtype=torch.float64)
ys4 = torch.zeros(2, 4, 3, 20, dtype=torch.float64)
ys4[:, [0, 2]] = ys2
assert torch.equal(cross_merge_paths(ys2, 4, 5, (0, 2)), cross_merge(ys4, 4, 5))
ys4r = torch.randn(2, 4, 3, 20, dtype=torch.float64)
assert torch.equal(cross_merge_paths(ys4r, 4, 5, (0, 1, 2, 3)), cross_merge(ys4r, 4, 5))
assert torch.equal(cross_merge_paths(cross_scan_paths(xg, (0, 2)), 4, 5, (0, 2)),
                   2 * xg.reshape(2, 3, 20))
print("B  cross_scan_paths / cross_merge_paths: (0, 2) == paths 0 and 2 of cross_scan, "
      "== cross_merge with 1 / 3 zeroed, round trip 2x; four paths == the originals (bit for bit)")


# ═══ C / D. the mixer == a naive float64 transcription ═══════════════════════
def cell_order(p, H, W):
    """Cells visited by forward path p, written out (0: row-major, 2: column-major)."""
    if p == 0:
        return [(r, c) for r in range(H) for c in range(W)]
    assert p == 2
    return [(r, c) for c in range(W) for r in range(H)]


def naive_mixer(mx, x, H, W):
    """SS2D with scan_paths (0, 2), optionally with [head, tail] leading rows."""
    Bn, L_in, _ = x.shape
    P = L_in - H * W
    d = mx.d_inner
    R, Nn = mx.dt_rank, mx.d_state
    z = x @ mx.in_proj.weight.T                               # (B, L_in, d)
    g = z[:, P:].reshape(Bn, H, W, d).permute(0, 3, 1, 2)
    g = F.silu(F.conv2d(g, mx.conv2d.weight, mx.conv2d.bias,
                        padding=mx.d_conv // 2, groups=d))     # (B, d, H, W)
    kc = mx.d_conv // 2
    tok = [F.silu(z[:, e] * mx.conv2d.weight[:, 0, kc, kc] + mx.conv2d.bias) for e in range(P)]
    out_grid = torch.zeros(Bn, d, H, W, dtype=x.dtype)
    out_tok = [torch.zeros(Bn, d, dtype=x.dtype) for _ in range(P)]
    for k, p in enumerate(mx.scan_paths):
        cells = cell_order(p, H, W)
        seq = ([tok[0]] if P else []) + [g[:, :, r, c] for r, c in cells] + ([tok[1]] if P else [])
        h = torch.zeros(Bn, d, Nn, dtype=x.dtype)
        # SS2D always hands A, D and the delta bias to the scan in fp32 (by design)
        A = (-torch.exp(mx.A_logs[k].float())).to(x.dtype)    # (d, N)
        Dk = mx.Ds[k].float().to(x.dtype)
        bias = mx.dt_projs_bias[k].float().to(x.dtype)
        for i, u in enumerate(seq):
            v = u @ mx.x_proj_weight[k].T                     # (B, R + 2N)
            dt = F.softplus(v[:, :R] @ mx.dt_projs_weight[k].T + bias)
            Bv, Cv = v[:, R:R + Nn], v[:, R + Nn:]
            h = torch.exp(dt[:, :, None] * A) * h + dt[:, :, None] * Bv[:, None, :] * u[:, :, None]
            y = (h * Cv[:, None, :]).sum(-1) + Dk * u
            if P and i == 0:
                out_tok[0] = out_tok[0] + y
            elif P and i == len(seq) - 1:
                out_tok[1] = out_tok[1] + y
            else:
                r, c = cells[i - (1 if P else 0)]
                out_grid[:, :, r, c] = out_grid[:, :, r, c] + y
    rows = out_tok + [out_grid[:, :, r, c] for r in range(H) for c in range(W)]
    out = torch.stack(rows, dim=1)                            # (B, L_in, d)
    return mx.out_proj(mx.out_norm(out))


Hm, Wm = 3, 5                                               # rectangular grid
worst = 0.0
for aux in (False, True):
    torch.manual_seed(3)
    mx = perturb(SS2D(D, d_state=N, K=2, scan_paths=(0, 2), aux_tokens=aux).double(), 4)
    for P in ((0, 2) if aux else (0,)):
        xm = torch.randn(B, P + Hm * Wm, D, dtype=torch.float64)
        with torch.no_grad():
            dd = (mx(xm, Hm, Wm) - naive_mixer(mx, xm, Hm, Wm)).abs().max().item()
        assert dd < 1e-12, (aux, P, dd)
        worst = max(worst, dd)
    if aux:                                                 # CPU bf16 autocast runs, finite
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            yb = mx.float()(torch.randn(B, 2 + Hm * Wm, D), Hm, Wm)
        assert torch.isfinite(yb.float()).all() and yb.shape == (B, 2 + Hm * Wm, D)
print(f"C  [0, 2] mixer without tokens == naive float64 transcription (3x5 grid)")
print(f"D  [0, 2] mixer with [head, tail] == naive transcription (head first / tail last in "
      f"both paths, centre tap + SiLU, outputs summed; max|diff| = {worst:.1e}); also with "
      f"the aux flag but no tokens yet; bf16 autocast runs")


# ═══ E. placement and causality ══════════════════════════════════════════════
def check_causal(fn, label, Lg):
    """fn(x) -> (B, 2 + Lg, D). Row 0 head, row 1 tail, rows 2.. grid."""
    torch.manual_seed(5)
    x0 = torch.randn(2, 2 + Lg, D, dtype=torch.float64)
    with torch.no_grad():
        y0 = fn(x0)
        x1 = x0.clone()
        x1[:, 1] += torch.randn(2, D, dtype=torch.float64)    # tail input
        y1 = fn(x1)
        assert torch.equal(y0[:, 0], y1[:, 0]) and torch.equal(y0[:, 2:], y1[:, 2:]), label
        assert (y0[:, 1] - y1[:, 1]).abs().max() > 0, label
        x2 = x0.clone()
        x2[:, 0] += torch.randn(2, D, dtype=torch.float64)    # head input
        diff = (fn(x2) - y0).abs().amax(dim=(0, 2))
        assert (diff[2:] > 0).all(), (label, "head does not reach every cell")
        for j in range(Lg):                                   # every cell reaches the tail
            x3 = x0.clone()
            x3[:, 2 + j] += torch.randn(2, D, dtype=torch.float64)
            y3 = fn(x3)
            assert (y3[:, 1] - y0[:, 1]).abs().max() > 0, (label, j)
            assert torch.equal(y3[:, 0], y0[:, 0]), (label, "grid reaches the head", j)


torch.manual_seed(6)
mx_e = perturb(SS2D(D, d_state=N, K=2, scan_paths=(0, 2), aux_tokens=True).double(), 7)
check_causal(lambda x: mx_e(x, Hm, Wm), "mixer", Hm * Wm)
torch.manual_seed(6)
blk = perturb(JiTBlock(D, d_state=N, K=2, scan_paths=(0, 2), aux_tokens=True).double(), 8)
c_blk = torch.randn(2, D, dtype=torch.float64)
check_causal(lambda x: blk(x, c_blk, Hm, Wm), "JiTBlock", Hm * Wm)
print("E  causality (mixer and whole JiTBlock, 3x5, float64): the tail's input changes no "
      "grid row and not the head (exactly); the head reaches every cell; every cell "
      "reaches the tail and none reaches the head")


# ═══ F. swap / noswap == hand-written block loops ════════════════════════════
def hand_forward(m, x, t, y, mode, start, seen=None):
    t_emb, y_emb = m.t_embedder(t), m.y_embedder(y)
    c = t_emb + y_emb
    h = m.x_embedder(x) + m.pos_embed
    n_swaps = 0
    for i, blk_i in enumerate(m.blocks):
        if i == start:
            mean = h.mean(dim=1, keepdim=True)
            h = torch.cat([mean, mean, h], dim=1)
            if seen is not None:
                seen.append(h.clone())
        h = blk_i(h, c, GRID, GRID)
        if mode == "swap" and start <= i < len(m.blocks) - 1:
            h = torch.cat([h[:, 1:2], h[:, 0:1], h[:, 2:]], dim=1)
            n_swaps += 1
    h = h[:, 2:]
    return m.unpatchify(m.final_layer(h, c)), n_swaps


START = 1
outs = {}
for mode in ("swap", "noswap"):
    m = to64(perturb(build(JiTVMamba, **TWO, aux_tokens=mode, aux_tokens_start=START)))
    captured = []
    hook = m.blocks[START].register_forward_pre_hook(lambda mod, a: captured.append(a[0].clone()))
    with torch.no_grad():
        got = m(x_img.double(), t_in.double(), y_in)
        seen = []
        ref, n_sw = hand_forward(m, x_img.double(), t_in.double(), y_in, mode, START, seen)
    hook.remove()
    assert torch.equal(got, ref), mode
    assert n_sw == (DEPTH - 1 - START if mode == "swap" else 0)
    xin = captured[0]
    assert torch.equal(xin[:, 0], xin[:, 1]) and torch.equal(xin, seen[0])
    assert (xin[:, 0] - xin[:, 2:].mean(1)).abs().max() < 1e-14
    outs[mode] = got
assert (outs["swap"] - outs["noswap"]).abs().max() > 1e-8
last = {}
for mode in ("swap", "noswap"):
    m = to64(perturb(build(JiTVMamba, **TWO, aux_tokens=mode, aux_tokens_start=DEPTH - 1)))
    with torch.no_grad():
        last[mode] = m(x_img.double(), t_in.double(), y_in)
assert torch.equal(last["swap"], last["noswap"])
print(f"F  swap / noswap == hand-written block loops exactly (tokens = grid mean at block "
      f"{START}, {DEPTH - 1 - START} swaps, strip); swap != noswap; start = depth-1 -> the two "
      f"modes equal exactly")


# ═══ G. parameter counts and strict loading ══════════════════════════════════
m4, m2 = build(JiTVMamba, **VM), build(JiTVMamba, **TWO)
R = m4.blocks[0].mixer.dt_rank
d_in = m4.blocks[0].mixer.d_inner
per_path = (R + 2 * N) * d_in + d_in * R + d_in + d_in * N + d_in
assert n_params(m4) - n_params(m2) == 2 * per_path * DEPTH, (n_params(m4), n_params(m2))
mx2 = m2.blocks[0].mixer
assert tuple(mx2.x_proj_weight.shape[:1]) == tuple(mx2.dt_projs_weight.shape[:1]) == \
    tuple(mx2.A_logs.shape[:1]) == tuple(mx2.Ds.shape[:1]) == (2,)
ms = build(JiTVMamba, **TWO, aux_tokens="swap", aux_tokens_start=1)
mn = build(JiTVMamba, **TWO, aux_tokens="noswap", aux_tokens_start=2)
assert n_params(ms) == n_params(mn) == n_params(m2)
for a in (m2, ms, mn):
    for b_ in (m2, ms, mn):
        b_.load_state_dict(a.state_dict(), strict=True)
assert all(torch.equal(v, ms.state_dict()[k]) for k, v in m2.state_dict().items())  # same init
print(f"G  params: 2-path = 4-path - 2 x {per_path:,} x {DEPTH} blocks (toy); aux tokens add 0 "
      f"and draw no random numbers; 2-path / swap / noswap load into each other strictly")


# ═══ H. guards ═══════════════════════════════════════════════════════════════
A2 = dict(TWO, aux_tokens="swap", aux_tokens_start=1)
assert fails(lambda: build(JiTVMamba, **VM, scan_paths=(0, 2)))        # K 4 != 2
assert fails(lambda: build(JiTVMamba, **VM, K=2))                      # K 2, four paths
assert fails(lambda: build(JiTVMamba, **VM, K=2, scan_paths=(0, 1)))
assert fails(lambda: build(JiTVMamba, **VM, K=3, scan_paths=(0, 1, 2)))
assert fails(lambda: build(JiTVMamba, **VM, aux_tokens="swap"))        # four paths
assert fails(lambda: build(JiTVMamba, **dict(A2, aux_tokens="bogus")))
assert fails(lambda: build(JiTVMamba, **dict(A2, aux_tokens_start=DEPTH)))
assert fails(lambda: build(JiTVMamba, **dict(A2, aux_tokens_start=-1)))
BAD = [dict(in_context_len=2, in_context_content="class"), dict(state_init="learned"),
       dict(state_init="dimsum"), dict(ssc="bc"), dict(ssc="static", adaln_cond="mlp"),
       dict(ssm_norm="dt"), dict(feature_rope="1d"), dict(a_tie="pairs"),
       dict(a_tie="pairs", bc_rope="2d_axial"), dict(ffn="convglu"), dict(ffn="glumbconv"),
       dict(norm="derf")]
for bad in BAD:
    assert fails(lambda: build(JiTVMamba, **dict(A2, **bad))), ("aux", bad)
    assert fails(lambda: build(JiTVMamba, **dict(TWO, **bad))), ("2-path", bad)
    build(JiTVMamba, **dict(VM, **bad))                                # still fine on 4 paths
assert fails(lambda: SS2D(D, d_state=N, aux_tokens=True))              # aux with four paths
assert fails(lambda: SS2D(D, d_state=N, K=4, scan_paths=(0, 2)))
mx_g = SS2D(D, d_state=N, K=2, scan_paths=(0, 2), aux_tokens=True)
assert fails(lambda: mx_g(torch.randn(1, 3 + 16, D), 4, 4))           # 3 extra rows
print(f"H  guards: K != len(scan_paths), unsupported paths, aux with four paths / bad mode / "
      f"bad start, {len(BAD)} arms or modifiers with aux and with the 2-path model; SS2D level")


# ═══ I. both build_model copies agree on the configs ═════════════════════════
try:
    import yaml                                             # noqa: E402
except ImportError as e:                                    # pragma: no cover
    print(f"I  SKIPPED (pyyaml not importable here: {e})")
    print("ALL CHECKS PASSED")
    sys.exit(0)


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
bm_prof = load_build_model("scripts/profile_model.py")
cfg_dir = os.path.join(REPO_ROOT, "configs", "tiny_imagenet")


def cfg_of(name):
    return yaml.safe_load(open(os.path.join(cfg_dir, name), encoding="utf-8"))


base_cfg = cfg_of("jit-s2-vmamba-baseline.yaml")
torch.manual_seed(0)
n_base = n_params(bm_eval(base_cfg))
assert n_base == 31_031_616, n_base
built = {}
for name, mode in (("jit-s2-vmamba-2path.yaml", "none"), ("jit-s2-vmamba-2path-auxswap.yaml", "swap")):
    cfg = cfg_of(name)
    diff = {k for k in set(cfg["model"]) | set(base_cfg["model"])
            if cfg["model"].get(k) != base_cfg["model"].get(k)}
    assert diff == ({"K", "scan_paths"} | ({"aux_tokens", "aux_tokens_start"} if mode != "none" else set())), diff
    torch.manual_seed(0)
    mt = bm_train(cfg)
    torch.manual_seed(0)
    me = bm_eval(cfg)
    st, se = mt.state_dict(), me.state_dict()
    assert set(st) == set(se) and all(torch.equal(st[k], se[k]) for k in st), name
    assert mt.scan_paths == me.scan_paths == (0, 2)
    assert mt.aux_tokens == me.aux_tokens == mode
    assert mt.aux_tokens_start == me.aux_tokens_start == (4 if mode != "none" else 0)
    assert n_params(mt) == 30_128_448 == n_base - 903_168, n_params(mt)
    built[name] = mt
torch.manual_seed(0)
mp = bm_prof(cfg_of("jit-s2-vmamba-2path.yaml"))
assert n_params(mp) == 30_128_448 and mp.scan_paths == (0, 2)
with torch.no_grad():                                       # one real-size forward
    mreal = built["jit-s2-vmamba-2path-auxswap.yaml"].eval()
    out = mreal(torch.randn(1, 3, 64, 64), torch.rand(1), torch.tensor([3]))
assert out.shape == (1, 3, 64, 64) and torch.isfinite(out).all()
print(f"I  configs: train == eval build_model, 30,128,448 params = baseline {n_base:,} - 903,168 "
      f"(2-path and 2-path-auxswap); profile_model builds the 2-path config; a 64x64 forward runs")

print("ALL CHECKS PASSED")
