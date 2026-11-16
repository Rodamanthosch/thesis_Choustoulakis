"""
tests/test_spatial_auxswap_cpu.py
=================================
CPU-only verification of SF-Mamba's auxiliary patch swapping in
JiT-S2-Spatial-Mamba (src/models/spatial_mamba.py: aux_tokens /
aux_tokens_start; Yoshimura et al., ICML 2026, arXiv 2603.16423).

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, kept
in float64 when fed float64 so the fidelity checks are exact). Only the plain
no-gate setup the two Tiny-IN runs use is exercised (d_state 1, lpu zero,
swiglu, rmsnorm); combinations with feature RoPE / ssm_norm / Derf / conv FFNs
are not tested.

Checks:
  A. aux_tokens "none" (the default) is BIT-IDENTICAL to the last commit
     before the feature (pinned af4de45; AUX_TOKENS_REFERENCE=<rev> overrides):
     the model exec'd out of `git show <rev>:src/models/spatial_mamba.py` and
     the current one, from the same seed, have identical state_dicts and outputs -- for
     the no-gate setup AND with in-context class tokens (so the _scan_order /
     token_conv edits leave the class-token arm untouched).
  B. _scan_order with positions (0, HW): scan = [head, cells 0..HW-1, tail].
  C. mixer with the two tokens == naive float64 transcription: head, row-major
     grid, tail; tokens get the DWConv centre tap + bias then SiLU; explicit
     recurrence; SASF over the GRID states only (replicate-clamped); tokens'
     states unfused; y = C h + D u; LayerNorm; out_proj.
  D. causality in one mixer: the tail token's input changes no grid output and
     not the head; the head token's input changes every grid output (SASF
     identity and the real SASF).
  E. model: swap and noswap == hand-written loops over model.blocks (mean of
     the grid at aux_tokens_start; head <-> tail after every block but the
     last for swap only; strip after the last), exact; per block, the head
     input of block i+1 IS the tail output of block i under swap and the head
     output under noswap; swap != noswap from the same weights; +0 params;
     strict load_state_dict both ways between none / swap / noswap; fwd + bwd
     finite; grad_checkpoint bit-identical; CPU bf16 autocast runs.
  F. guards: unknown mode, start outside the blocks, with class tokens.
  G. wiring: both Tiny-IN configs build IDENTICAL models through
     scripts/run_experiment.py::build_model and scripts/evaluate.py::build_model,
     29,686,116 params (= jit-s2-spatial-nogate), tokens at (0, 64) from block 4
     (needs pyyaml).

Run from the repo root:
    PYTHONPATH=. python tests/test_spatial_auxswap_cpu.py
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
    JiTSpatialMamba, StructureAwareSSM,
)

# The plain no-gate setup of the Tiny-IN runs, at toy size.
KW = dict(input_size=16, patch_size=4, hidden_size=32, depth=4, num_classes=7,
          bottleneck_dim=16, sasf_dilations=(1, 2, 3), d_state=1, gate=False,
          lpu="zero", ffn="swiglu")
H = W = KW["input_size"] // KW["patch_size"]       # 4
HW = H * W
Bsz = 2
START = 1                                           # swaps after blocks 1, 2
torch.manual_seed(0)
x_img = torch.randn(Bsz, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(Bsz)
y_in = torch.randint(0, KW["num_classes"], (Bsz,))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def perturb(module, scale=0.05):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))
    return module


# ═══ A. aux_tokens "none" == the pre-feature commit, bit for bit ═════════════
REF = os.environ.get("AUX_TOKENS_REFERENCE", "af4de45")
try:
    head_src = subprocess.run(
        ["git", "show", f"{REF}:src/models/spatial_mamba.py"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8")
except Exception as e:                                      # pragma: no cover
    head_src = None
    print(f"A  SKIPPED (cannot read {REF}:src/models/spatial_mamba.py: {e})")

if head_src is not None:
    head_mod = types.ModuleType("spatial_mamba_head")
    exec(compile(head_src, f"<{REF}:src/models/spatial_mamba.py>", "exec"), head_mod.__dict__)
    n_cfg = 0
    for extra in ({}, dict(in_context_len=3, in_context_start=1,
                           in_context_positions=[[0, 1], [2, 1], [3, 3]])):
        kw = dict(KW, **extra)
        torch.manual_seed(1)
        old = head_mod.JiTSpatialMamba(**kw).eval()
        torch.manual_seed(1)
        new = JiTSpatialMamba(**kw).eval()
        so, sn = old.state_dict(), new.state_dict()
        assert set(so) == set(sn), (kw, set(so) ^ set(sn))
        assert all(torch.equal(so[k], sn[k]) for k in so), kw
        perturb(old, 0.02)
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            d = (old(x_img, t_in, y_in) - new(x_img, t_in, y_in)).abs().max().item()
        assert d == 0.0, (kw, d)
        n_cfg += 1
    print(f"A  aux_tokens none: bit-identical to {REF} (init AND output), "
          f"no-gate and no-gate + class tokens ({n_cfg} configs)")


# ═══ B. scan order: [head, grid, tail] ════════════════════════════════════════
mx = StructureAwareSSM(12, in_context_positions=(0, HW), token_conv=True)
order, inv = mx._scan_order(HW, "cpu")
assert order.tolist() == [0] + [2 + g for g in range(HW)] + [1], order.tolist()
assert torch.equal(order[inv], torch.arange(HW + 2))
print(f"B  positions (0, {HW}): scan order = [head, cells 0..{HW - 1}, tail]")


# ═══ C. mixer with the two tokens == naive transcription (float64) ════════════
def naive_mixer(mx, x, Hh, Ww):
    """Head token, the row-major grid, tail token; d_state 1, no gate."""
    Bb, Lt, _ = x.shape
    n_cells = Hh * Ww
    assert Lt == 2 + n_cells and mx.d_state == 1 and not mx.gate
    di, R = mx.d_inner, mx.dt_rank
    u_all = x @ mx.in_proj.weight.T                          # (B, L, di)
    ug = mx.act(mx.conv2d(u_all[:, 2:].reshape(Bb, Hh, Ww, di).permute(0, 3, 1, 2)))
    ug = ug.reshape(Bb, di, n_cells)
    wc, bc = mx.conv2d.weight[:, 0, 1, 1], mx.conv2d.bias    # centre tap
    tok = [F.silu(u_all[:, e] * wc + bc) for e in (0, 1)]    # (B, di) each
    seq = [("e", 0)] + [("g", g) for g in range(n_cells)] + [("e", 1)]
    useq = torch.stack([tok[i] if k == "e" else ug[:, :, i] for k, i in seq], dim=-1)
    dbl = torch.einsum("od,bdl->bol", mx.x_proj_weight, useq)
    dt_r, Bt, Ct = dbl[:, :R], dbl[:, R], dbl[:, R + 1]      # B_t, C_t: (B, L)
    delta = F.softplus(torch.einsum("dr,brl->bdl", mx.dt_projs_weight, dt_r)
                       + mx.dt_projs_bias.float().to(x.dtype)[None, :, None])
    Av = -torch.exp(mx.A_logs.float()).to(x.dtype)[:, 0]    # (di,), fp32 as the model
    hst = x.new_zeros(Bb, di)
    st = {}
    for l, key in enumerate(seq):                            # x_t = e^{dA} x_{t-1} + d B u
        hst = torch.exp(delta[:, :, l] * Av) * hst + delta[:, :, l] * Bt[:, None, l] * useq[:, :, l]
        st[key] = (hst, Ct[:, l], useq[:, :, l])
    sf = mx.state_fusion
    cl = lambda v, n: min(max(v, 0), n - 1)                  # noqa: E731
    ys = []
    for e in (0, 1):                                         # tokens: unfused state
        hs, Cv, uv = st[("e", e)]
        ys.append(hs * Cv[:, None] + mx.Ds * uv)
    for i in range(Hh):
        for j in range(Ww):
            hf = x.new_zeros(Bb, di)
            for k, dd in enumerate(sf.dilations):
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        nb = st[("g", cl(i + a * dd, Hh) * Ww + cl(j + b * dd, Ww))][0]
                        hf = hf + sf.alpha[k] * sf.kernels[k][:, 0, a + 1, b + 1] * nb
            _, Cv, uv = st[("g", i * Ww + j)]
            ys.append(hf * Cv[:, None] + mx.Ds * uv)
    y = mx.out_norm(torch.stack(ys, dim=1))                  # (B, 2 + HW, di)
    return y @ mx.out_proj.weight.T


worst = 0.0
for seed in (3, 4):
    torch.manual_seed(seed)
    mx = StructureAwareSSM(12, d_state=1, gate=False, dilations=(1, 2),
                           in_context_positions=(0, HW), token_conv=True)
    perturb(mx, 0.2).double().eval()
    xin = torch.randn(2, 2 + HW, 12, dtype=torch.float64)
    with torch.no_grad():
        got, want = mx(xin, H, W), naive_mixer(mx, xin, H, W)
    err = (got - want).abs().max().item()
    assert err < 1e-10, (seed, err)
    worst = max(worst, err)
print(f"C  mixer with head + tail tokens == naive transcription (max|diff| = {worst:.1e})")


# ═══ D. causality: tail reads everything, head is read by everything ══════════
def token_effect(mx, xin, e):
    x2 = xin.clone()
    x2[:, e] += 1.0
    with torch.no_grad():
        return (mx(xin, H, W) - mx(x2, H, W)).abs().amax(dim=-1)[0]   # (2 + HW,)


for sasf in ("identity", "real"):
    torch.manual_seed(5)
    mx = perturb(StructureAwareSSM(12, d_state=1, gate=False, dilations=(1, 2),
                                   in_context_positions=(0, HW), token_conv=True),
                 0.2).double().eval()
    if sasf == "identity":
        with torch.no_grad():
            for k in mx.state_fusion.kernels:
                k.zero_()
                k[:, 0, 1, 1] = 1.0
            mx.state_fusion.alpha.copy_(torch.tensor([1.0, 0.0]))
    xin = torch.randn(1, 2 + HW, 12, dtype=torch.float64)
    d_tail, d_head = token_effect(mx, xin, 1), token_effect(mx, xin, 0)
    assert d_tail[1] > 0 and (d_tail[0] == 0).item() and (d_tail[2:] == 0).all(), sasf
    assert d_head[0] > 0 and (d_head[1] > 0).item() and (d_head[2:] > 0).all(), sasf
print("D  tail input changes no grid output (nor the head); head input changes every "
      "grid output and the tail (SASF identity and real)")


# ═══ E. model == hand-written loop; swap vs noswap ════════════════════════════
def hand_forward(m, swap):
    """The forward written out: tokens = mean of the grid at START, head <-> tail
    after every block but the last (swap only), strip after the last."""
    c = m.t_embedder(t_in) + m.y_embedder(y_in)
    x = m.x_embedder(x_img) + m.pos_embed
    ins, outs = [], []
    for i, blk in enumerate(m.blocks):
        if i == START:
            outs.append(x)                       # the grid the tokens average
            mean = x.mean(dim=1, keepdim=True)
            x = torch.cat([mean, mean, x], dim=1)
        ins.append(x)
        x = blk(x, c, H, W)
        outs.append(x)
        if swap and START <= i < len(m.blocks) - 1:
            x = torch.cat([x[:, 1:2], x[:, 0:1], x[:, 2:]], dim=1)
    x = x[:, 2:]
    pre = outs.pop(START)
    return m.unpatchify(m.final_layer(x, c)), ins, outs, pre


base = JiTSpatialMamba(**KW)
models = {}
for mode in ("none", "swap", "noswap"):
    torch.manual_seed(6)
    models[mode] = perturb(JiTSpatialMamba(**KW, aux_tokens=mode, aux_tokens_start=START),
                           0.05).eval()
for mode in ("swap", "noswap"):
    m = models[mode]
    assert n_params(m) == n_params(base)
    assert all(b.mixer.in_context_positions == (0, HW) and b.mixer.token_conv for b in m.blocks)
    with torch.no_grad():
        want, ins, outs, pre = hand_forward(m, swap=(mode == "swap"))
        got = m(x_img, t_in, y_in)
    assert torch.equal(got, want), mode
    assert torch.equal(ins[START][:, 2:], pre)                              # grid kept
    assert torch.equal(ins[START][:, 0], pre.mean(dim=1))                   # mean init
    assert torch.equal(ins[START][:, 0], ins[START][:, 1])
    for i in range(START, len(m.blocks) - 1):
        nxt, prv = ins[i + 1], outs[i]
        head_src, tail_src = (prv[:, 1], prv[:, 0]) if mode == "swap" else (prv[:, 0], prv[:, 1])
        assert torch.equal(nxt[:, 0], head_src) and torch.equal(nxt[:, 1], tail_src), (mode, i)
        assert torch.equal(nxt[:, 2:], prv[:, 2:])
with torch.no_grad():
    o_swap, o_noswap = models["swap"](x_img, t_in, y_in), models["noswap"](x_img, t_in, y_in)
    o_none = models["none"](x_img, t_in, y_in)
assert (o_swap - o_noswap).abs().max() > 0 and (o_swap - o_none).abs().max() > 0

for a in models:                                            # same keys, strict both ways
    for b in models:
        models[b].load_state_dict(models[a].state_dict(), strict=True)

grads = {}
for ck in (False, True):
    torch.manual_seed(7)
    m = perturb(JiTSpatialMamba(**KW, aux_tokens="swap", aux_tokens_start=START,
                                grad_checkpoint=ck), 0.05).train()
    out = m(x_img, t_in, y_in)
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    g = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(v).all() for v in g.values())
    assert g["blocks.0.mixer.state_fusion.alpha"].abs().sum() > 0
    grads[ck] = (out.detach(), g)
assert torch.equal(grads[False][0], grads[True][0])
assert set(grads[False][1]) == set(grads[True][1])
assert all(torch.equal(grads[False][1][n], grads[True][1][n]) for n in grads[False][1])
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = m(x_img, t_in, y_in)
assert torch.isfinite(out.float()).all()
print(f"E  swap / noswap == hand-written loops (exact); block i+1 head = block i tail "
      f"(swap) / head (noswap); mean init; +0 params; strict load across modes; "
      f"fwd+bwd, grad_checkpoint bit-identical, bf16 runs")


# ═══ F. guards ════════════════════════════════════════════════════════════════
def fails(**kw):
    try:
        JiTSpatialMamba(**dict(KW, **kw))
    except (AssertionError, ValueError):
        return True
    return False


assert fails(aux_tokens="learnable")
assert fails(aux_tokens="swap", aux_tokens_start=KW["depth"])
assert fails(aux_tokens="swap", aux_tokens_start=-1)
assert fails(aux_tokens="swap", in_context_len=2)
assert fails(aux_tokens="noswap", in_context_len=1)
assert not fails(aux_tokens="none", in_context_len=1)
print("F  guards: unknown mode, start outside the blocks, with class tokens")


# ═══ G. both build_model copies agree on both configs ═════════════════════════
try:
    import yaml                                             # noqa: E402
except ImportError as e:                                    # pragma: no cover
    print(f"G  SKIPPED (pyyaml not importable here: {e})")
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
cfg_dir = os.path.join(REPO_ROOT, "configs", "tiny_imagenet")
base_cfg = yaml.safe_load(open(os.path.join(cfg_dir, "jit-s2-spatial-nogate.yaml"),
                               encoding="utf-8"))
n_base = n_params(bm_train(base_cfg))
assert n_base == 29_686_116, n_base
for mode in ("swap", "noswap"):
    name = f"jit-s2-spatial-nogate-aux{mode}.yaml"
    cfg = yaml.safe_load(open(os.path.join(cfg_dir, name), encoding="utf-8"))
    # Same model as the no-gate config apart from the two aux keys.
    assert {k: v for k, v in cfg["model"].items() if not k.startswith("aux_")} \
        == base_cfg["model"], name
    torch.manual_seed(0)
    mt = bm_train(cfg)
    torch.manual_seed(0)
    me = bm_eval(cfg)
    st, se = mt.state_dict(), me.state_dict()
    assert set(st) == set(se) and all(torch.equal(st[k], se[k]) for k in st), name
    assert (mt.aux_tokens, mt.aux_tokens_start, me.aux_tokens, me.aux_tokens_start) \
        == (mode, 4, mode, 4), name
    assert all(b.mixer.in_context_positions == (0, 64) and b.mixer.token_conv
               for b in mt.blocks)
    assert (mt.d_state, mt.gate, mt.lpu, mt.sasf_dilations, mt.ffn, mt.in_context_len) \
        == (1, False, "zero", (1, 2, 3), "swiglu", 0)
    assert n_params(mt) == n_base, (n_params(mt), n_base)
    print(f"G  {name}: train == eval build_model, aux_tokens {mode} from block 4, "
          f"{n_params(mt):,} params (= no-gate baseline)")

print("ALL CHECKS PASSED")
