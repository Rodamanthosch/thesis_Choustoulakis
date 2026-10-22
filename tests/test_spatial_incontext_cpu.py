"""
tests/test_spatial_incontext_cpu.py
===================================
CPU-only verification of the in-context class tokens of JiT-S2-Spatial-Mamba
(src/models/spatial_mamba.py: in_context_len / in_context_start /
in_context_content / in_context_positions).

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, kept
in float64 when fed float64 so the fidelity checks are exact).

Checks:
  A. tokens OFF (the default) is BIT-IDENTICAL to HEAD: the model exec'd out
     of `git show HEAD:src/models/spatial_mamba.py` and the current one, built
     from the same seed, have identical state_dicts and outputs (gate on/off,
     lpu none/zero/paper, d_state 1 and 3 shared/per-state, swiglu/convglu),
     so every finished Spatial run is untouched.
  B. mixer fidelity WITH tokens: equals a naive transcription -- the scan
     sequence built token by token (token e right before grid cell p_e, tokens
     unconvolved), the explicit N-state recurrence, SASF over the GRID states
     only with replicate-clamped neighbours, the tokens' states unfused,
     y = sum_n C[n] h[c, n] + D u, LayerNorm, optional SiLU(z), out_proj -- for
     gate on/off, d_state 1 and 3 (shared / per-state), positions including
     duplicates, cell 0 and the last cell (float64).
  C. placement is causal: with SASF set to the identity, changing token e's
     input leaves every grid cell scanned before it (and every earlier token)
     EXACTLY unchanged and changes the cell right after it.
  D. model: positions=None == every token at (0, 0); +in_context_len * D params
     exactly; forward + backward with grads into the slots and the label
     embedding (lpu zero/paper, swiglu/convglu, d_state 1/2); the LPU gives a
     token only its centre tap; CPU bf16 autocast runs.
  E. grad_checkpoint with tokens changes nothing (bit-identical output + grads).
  F. guards: wrong count / off-grid positions, positions without tokens,
     content other than class, in_context_start outside the blocks.
  G. the distance figures in the config header, recomputed from the module's
     own scan order and dilations: 3 tokens at (0,3), (3,3), (7,3) on the 8x8
     grid put every image token within max 2 / mean 1.1875 scan steps of a
     class token (one head token: 37 / 12.25).
  H. wiring: the config builds IDENTICAL models through
     scripts/run_experiment.py::build_model and scripts/evaluate.py::build_model,
     with the tokens, positions and param count expected (needs pyyaml).

Run from the repo root:
    PYTHONPATH=. python tests/test_spatial_incontext_cpu.py
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

KW = dict(input_size=16, patch_size=4, hidden_size=32, depth=3, num_classes=7,
          bottleneck_dim=16, sasf_dilations=(1, 2, 3))
H = W = KW["input_size"] // KW["patch_size"]       # 4
HW = H * W
Bsz = 2
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


# ═══ A. tokens off == HEAD, bit for bit ═══════════════════════════════════════
try:
    head_src = subprocess.run(
        ["git", "show", "HEAD:src/models/spatial_mamba.py"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8")
except Exception as e:                                      # pragma: no cover
    head_src = None
    print(f"A  SKIPPED (cannot read HEAD:src/models/spatial_mamba.py: {e})")

if head_src is not None:
    head_mod = types.ModuleType("spatial_mamba_head")
    exec(compile(head_src, "<HEAD:src/models/spatial_mamba.py>", "exec"), head_mod.__dict__)
    n_cfg = 0
    for gate in (True, False):
        for lpu in ("none", "zero", "paper"):
            for ds, sk, ffn in ((1, "shared", "swiglu"), (3, "shared", "convglu"),
                                (3, "per_state", "swiglu")):
                kw = dict(KW, gate=gate, lpu=lpu, d_state=ds, sasf_state_kernels=sk, ffn=ffn)
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
    print(f"A  tokens off: bit-identical to HEAD (init AND output) for {n_cfg} configs")


# ═══ B. mixer with tokens == naive transcription (float64) ═══════════════════
def scan_sequence(pos_flat, n_cells):
    """[('e', e) | ('g', g)] in scan order: token e right before cell pos_flat[e]."""
    seq, e = [], 0
    for g in range(n_cells):
        while e < len(pos_flat) and pos_flat[e] == g:
            seq.append(("e", e))
            e += 1
        seq.append(("g", g))
    assert e == len(pos_flat)
    return seq


def naive_mixer(mx, x, Hh, Ww):
    pos = mx.in_context_positions
    Bb, Lt, _ = x.shape
    P, n_cells = len(pos), Hh * Ww
    assert Lt == P + n_cells
    di, Ns, R = mx.d_inner, mx.d_state, mx.dt_rank
    xz = x @ mx.in_proj.weight.T
    u_all, z = (xz[..., :di], xz[..., di:]) if mx.gate else (xz, None)
    ug = mx.act(mx.conv2d(u_all[:, P:].reshape(Bb, Hh, Ww, di).permute(0, 3, 1, 2)))
    ug = ug.reshape(Bb, di, n_cells)                         # convolved grid
    ue = u_all[:, :P].transpose(1, 2)                        # tokens: no conv
    seq = scan_sequence(pos, n_cells)
    useq = torch.stack([ue[:, :, i] if k == "e" else ug[:, :, i] for k, i in seq], dim=-1)
    dbl = torch.einsum("od,bdl->bol", mx.x_proj_weight, useq)
    dt_r, Bt, Ct = dbl[:, :R], dbl[:, R:R + Ns], dbl[:, R + Ns:]
    delta = F.softplus(torch.einsum("dr,brl->bdl", mx.dt_projs_weight, dt_r)
                       + mx.dt_projs_bias.float().to(x.dtype)[None, :, None])
    Av = -torch.exp(mx.A_logs.float()).to(x.dtype)           # (d, N)
    hst = x.new_zeros(Bb, di, Ns)
    st = {}
    for l, key in enumerate(seq):                            # explicit N-state recurrence
        hst = torch.exp(delta[:, :, l, None] * Av) * hst \
            + delta[:, :, l, None] * Bt[:, None, :, l] * useq[:, :, l, None]
        st[key] = (hst, Ct[:, :, l], useq[:, :, l])          # state, C_t, u_t
    sf = mx.state_fusion
    per_state = sf.dim == di * Ns and Ns > 1
    cl = lambda v, n: min(max(v, 0), n - 1)                  # noqa: E731

    def kern(k, n):                                          # (di, 3, 3) for state n
        w = sf.kernels[k][:, 0]
        return w[n * di:(n + 1) * di] if per_state else w

    ys = []
    for e in range(P):                                       # tokens: unfused state
        hs, Cv, uv = st[("e", e)]
        ys.append((hs * Cv[:, None, :]).sum(-1) + mx.Ds * uv)
    for i in range(Hh):
        for j in range(Ww):
            hs_f = x.new_zeros(Bb, di, Ns)
            for k, dd in enumerate(sf.dilations):
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        nb = st[("g", cl(i + a * dd, Hh) * Ww + cl(j + b * dd, Ww))][0]
                        for n in range(Ns):
                            hs_f[:, :, n] += sf.alpha[k] * kern(k, n)[:, a + 1, b + 1] * nb[:, :, n]
            _, Cv, uv = st[("g", i * Ww + j)]
            ys.append((hs_f * Cv[:, None, :]).sum(-1) + mx.Ds * uv)
    y = mx.out_norm(torch.stack(ys, dim=1))                  # (B, P + HW, di)
    if z is not None:
        y = y * F.silu(z)
    return y @ mx.out_proj.weight.T


n_b, worst_b = 0, 0.0
for gate in (True, False):
    for ds, sk in ((1, "shared"), (3, "shared"), (3, "per_state")):
        for pos in ((0, 5, 5, 15), (0, 0), (6,)):
            torch.manual_seed(3)
            mx = StructureAwareSSM(12, d_state=ds, gate=gate, dilations=(1, 2),
                                   state_kernels=sk, in_context_positions=pos)
            perturb(mx, 0.2).double().eval()
            xin = torch.randn(2, len(pos) + HW, 12, dtype=torch.float64)
            with torch.no_grad():
                got, want = mx(xin, H, W), naive_mixer(mx, xin, H, W)
            err = (got - want).abs().max().item()
            assert err < 1e-10, (gate, ds, sk, pos, err)
            worst_b = max(worst_b, err)
            n_b += 1
print(f"B  mixer with tokens == naive transcription for {n_b} configs "
      f"(max|diff| = {worst_b:.1e})")


# ═══ C. placement is causal (SASF set to the identity) ════════════════════════
pos = (2, 9, 9, 14)
torch.manual_seed(4)
mx = perturb(StructureAwareSSM(12, d_state=1, gate=True, dilations=(1, 2),
                               in_context_positions=pos), 0.2).double().eval()
with torch.no_grad():
    for k in mx.state_fusion.kernels:
        k.zero_()
        k[:, 0, 1, 1] = 1.0
    mx.state_fusion.alpha.copy_(torch.tensor([1.0, 0.0]))
xin = torch.randn(1, len(pos) + HW, 12, dtype=torch.float64)
P = len(pos)
for e, p in enumerate(pos):
    x2 = xin.clone()
    x2[:, e] += 1.0
    with torch.no_grad():
        y1, y2 = mx(xin, H, W), mx(x2, H, W)
    diff = (y1 - y2).abs().amax(dim=-1)[0]                    # per token, layout order
    before = [P + g for g in range(p)] + [f for f in range(P) if pos[f] < p or f < e]
    assert all(diff[i].item() == 0.0 for i in before), (e, p)
    assert diff[P + p].item() > 0 and diff[e].item() > 0, (e, p)
print(f"C  token e changes nothing scanned before its cell, and the cell after it "
      f"(positions {pos})")


# ═══ D. model: defaults, params, smoke, LPU centre tap ════════════════════════
cells = [[0, 1], [2, 1], [3, 3]]
base = JiTSpatialMamba(**KW)
torch.manual_seed(5)
m_none = JiTSpatialMamba(**KW, in_context_len=2, in_context_start=1)
torch.manual_seed(5)
m_00 = JiTSpatialMamba(**KW, in_context_len=2, in_context_start=1,
                       in_context_positions=[[0, 0], [0, 0]])
assert m_none.in_context_positions == m_00.in_context_positions == ((0, 0), (0, 0))
sa, sb = m_none.state_dict(), m_00.state_dict()
assert all(torch.equal(sa[k], sb[k]) for k in sa)
m_unsorted = JiTSpatialMamba(**KW, in_context_len=3, in_context_start=1,
                             in_context_positions=[[3, 3], [0, 1], [2, 1]])
assert m_unsorted.in_context_positions == ((0, 1), (2, 1), (3, 3))
assert m_unsorted.blocks[0].mixer.in_context_positions == (1, 9, 15)
assert n_params(m_unsorted) == n_params(base) + 3 * KW["hidden_size"]

n_d = 0
for lpu in ("zero", "paper"):
    for ffn, ds in (("swiglu", 1), ("convglu", 1), ("swiglu", 2)):
        torch.manual_seed(6)
        m = perturb(JiTSpatialMamba(**dict(KW, lpu=lpu, ffn=ffn, d_state=ds),
                                    in_context_len=3, in_context_start=1,
                                    in_context_positions=cells), 0.02).train()
        out = m(x_img, t_in, y_in)
        assert out.shape == x_img.shape and torch.isfinite(out).all()
        out.square().mean().backward()
        for p in (m.incontext_pos_embed, m.y_embedder.embedding_table.weight,
                  m.blocks[-1].mixer.state_fusion.alpha):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        n_d += 1
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = m(x_img, t_in, y_in)
assert torch.isfinite(out.float()).all()

blk = m.blocks[0]                                            # lpu "paper": random conv
xt = torch.randn(2, 3 + HW, KW["hidden_size"])
lp = blk._grid_conv(blk.cpe1, xt, H, W)
ref_grid = blk.cpe1(xt[:, 3:].transpose(1, 2).reshape(2, -1, H, W)).flatten(2).transpose(1, 2)
assert torch.allclose(lp[:, 3:], ref_grid)
assert torch.allclose(lp[:, :3], xt[:, :3] * blk.cpe1.weight[:, 0, 1, 1] + blk.cpe1.bias)
print(f"D  positions=None == (0,0); +3*D params; fwd+bwd with grads into the slots for "
      f"{n_d} configs; bf16 runs; LPU gives tokens only the centre tap")


# ═══ E. grad_checkpoint with tokens is bit-identical ══════════════════════════
res = []
for ck in (False, True):
    torch.manual_seed(7)
    m = perturb(JiTSpatialMamba(**dict(KW, lpu="zero"), grad_checkpoint=ck,
                                in_context_len=3, in_context_start=1,
                                in_context_positions=cells), 0.02).train()
    out = m(x_img, t_in, y_in)
    out.square().mean().backward()
    res.append((out.detach(), {n: p.grad.clone() for n, p in m.named_parameters()
                               if p.grad is not None}))
assert torch.equal(res[0][0], res[1][0])
assert set(res[0][1]) == set(res[1][1]) and "incontext_pos_embed" in res[0][1]
assert all(torch.equal(res[0][1][n], res[1][1][n]) for n in res[0][1])
print("E  grad_checkpoint with tokens: bit-identical output and gradients")


# ═══ F. guards ════════════════════════════════════════════════════════════════
def fails(**kw):
    try:
        JiTSpatialMamba(**dict(KW, **kw))
    except (AssertionError, ValueError):
        return True
    return False


assert fails(in_context_len=2, in_context_positions=[[0, 1]])                 # count
assert fails(in_context_len=1, in_context_positions=[[0, H]])                 # off-grid
assert fails(in_context_len=1, in_context_positions=[[0, 1, 2]])              # not [r, c]
assert fails(in_context_positions=[[0, 1]])                                   # no tokens
assert fails(in_context_len=2, in_context_content="time_class")               # not ported
assert fails(in_context_len=1, in_context_start=KW["depth"])                  # no such block
mx = StructureAwareSSM(12, in_context_positions=(3,))
try:
    mx(torch.randn(1, HW + 2, 12), H, W)                                       # wrong P
    raise RuntimeError("mixer accepted the wrong number of tokens")
except AssertionError:
    pass
print("F  guards: count, off-grid, shape, no tokens, content, start, mixer token count")


# ═══ G. the config's distance figures, from the module's own scan order ═══════
def distance_stats(mixer, n):
    order, _ = mixer._scan_order(n * n, "cpu")
    P = len(mixer.in_context_positions)
    spos = {int(idx): s for s, idx in enumerate(order.tolist())}      # layout -> scan pos
    tok_s = sorted(spos[e] for e in range(P))
    d = {}
    for g in range(n * n):
        s = spos[P + g]
        prev = [t for t in tok_s if t < s]
        d[divmod(g, n)] = s - prev[-1] if prev else 10 ** 6
    cl = lambda v: min(max(v, 0), n - 1)                     # noqa: E731
    fused = [min(d[(cl(i + a * k), cl(j + b * k))]
                 for k in mixer.state_fusion.dilations for a in (-1, 0, 1) for b in (-1, 0, 1))
             for i in range(n) for j in range(n)]
    return max(fused), sum(fused) / len(fused)


tiny = dict(input_size=64, patch_size=8, hidden_size=16, depth=1, num_classes=3,
            bottleneck_dim=8, sasf_dilations=(1, 2, 3))
m3 = JiTSpatialMamba(**tiny, in_context_len=3, in_context_positions=[[0, 3], [3, 3], [7, 3]])
m1 = JiTSpatialMamba(**tiny, in_context_len=1)
s3, s1 = distance_stats(m3.blocks[0].mixer, 8), distance_stats(m1.blocks[0].mixer, 8)
assert s3 == (2, 76 / 64), s3
assert s1 == (37, 784 / 64), s1
print(f"G  8x8, dilations (1,2,3): tokens at (0,3),(3,3),(7,3) -> max {s3[0]} / mean "
      f"{s3[1]:.4f} scan steps; one head token -> {s1[0]} / {s1[1]:.2f}")


# ═══ H. both build_model copies agree on the config ═══════════════════════════
try:
    import yaml                                             # noqa: E402
except ImportError as e:                                    # pragma: no cover
    print(f"H  SKIPPED (pyyaml not importable here: {e})")
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
name = "jit-s2-spatial-nogate-incontext-class3.yaml"
cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "tiny_imagenet", name),
                          encoding="utf-8"))
base_cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "tiny_imagenet",
                                            "jit-s2-spatial-nogate.yaml"), encoding="utf-8"))
torch.manual_seed(0)
mt = bm_train(cfg)
torch.manual_seed(0)
me = bm_eval(cfg)
st, se = mt.state_dict(), me.state_dict()
assert set(st) == set(se) and all(torch.equal(st[k], se[k]) for k in st), name
assert (mt.in_context_len, mt.in_context_start, mt.in_context_content) == (3, 4, "class")
assert mt.in_context_positions == ((0, 3), (3, 3), (7, 3))
assert all(b.mixer.in_context_positions == (3, 27, 59) for b in mt.blocks)
assert (mt.d_state, mt.gate, mt.lpu, mt.sasf_dilations) == (1, False, "zero", (1, 2, 3))
n_base = n_params(bm_train(base_cfg))
assert n_params(mt) == n_base + 3 * 384, (n_params(mt), n_base)
print(f"H  {name}: train == eval build_model, {n_params(mt):,} params "
      f"(no-gate baseline {n_base:,} + 3 x 384)")

print("ALL CHECKS PASSED")
