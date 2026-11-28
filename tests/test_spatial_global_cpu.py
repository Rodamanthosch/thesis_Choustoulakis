"""
tests/test_spatial_global_cpu.py
================================
CPU-only verification of JiT-S2-Spatial-Global (src/models/spatial_global.py):
Spatial-Mamba (one row-major scan, SASF, d_state 1) with M global condition
tokens that never enter the scan. They write an undecayed memory S that
anchors the image's scan states BEFORE SASF, and read S and the image memory G.

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, kept
in float64 when fed float64 so the fidelity checks are exact). Mixer proofs
run in float64; model-level checks in float32 (TimestepEmbedder is fp32).

Checks:
  A.  global_len 0: every mixer is StructureAwareSSM and the model IS
      JiTSpatialMamba's no-gate baseline (same keys, same weights at the same
      seed, strict load both ways, max|diff| == 0, also perturbed).
  A2. SpatialGlobalSSM fed only the grid == StructureAwareSSM (== 0, gate on/off).
  A3. global_len 3: every shared weight (slots included) equals
      JiTSpatialMamba(in_context_len=3, positions None, same start) at the
      same seed, the RNG state after construction is equal, only alpha_* new.
  B.  the mixer == a naive float64 transcription: states by explicit decay
      sums, + alpha_xe S, SASF as replicate-clamped neighbour sums with alpha,
      y = C h + D u; tokens (a_ee S + a_ex G) C^e + D u^e; LayerNorm, SiLU(z),
      out_proj. Rectangular 3x4 grid, random gains; full / write-only,
      emit_tokens on / off, gate on / off.
  C.  anchored recurrence x'_t = Abar x'_{t-1} + (1 - Abar) aS + v B, x'_0 = aS,
      naive loop == states + aS (the map SASF fuses).
  C2. SASF of a spatially constant map == kappa * map, kappa_c = sum_j alpha_j
      sum(taps_j[c]) (replicate padding), at random kernels.
  D.  permuting the tokens (mixer rows; model slots) leaves the image outputs
      unchanged and permutes the token outputs.
  E.  G == a naive raster loop; S invariant to token order.
  F.  D counted once: Ds -> 0 changes the image rows by exactly D u and the
      token rows by D u^e; with x_proj's B row zeroed the output is D alone.
  G.  bf16 autocast: forward + backward finite (both modes), memories fp32;
      gradients reach the slots, gains and conv centre tap + bias; in
      write-only mode a token-only loss reaches x_proj and NOT the image.
  H.  parameter counts (+3 / +2 x d_inner per block, +d_inner in the last,
      none before global_start), every alpha_* in the no-decay group.
  H2. last block (emit_tokens False) vs True on the same weights: image rows
      identical (== 0), token rows the D term only, no G.
  I.  strict loading: write-only <- full and global <- Spatial prefix fail.
  J.  guards: every forbidden flag and bad value raises.
  K.  wiring: both build_models build IDENTICAL models from the three
      configs, and both reject a spatial_mamba option under spatial_global.
  L.  every block is the identity at init (adaLN-Zero, LPU zero), both modes.
  M.  ffn convglu / glumbconv, norm derf and gate on: forward + backward
      finite; grad_checkpoint: bit-identical output and grads.
  N.  no dead parameters: at the real init every requires_grad parameter has
      .grad not None (what DDP checks); after perturbing the zero-init
      weights every .grad is non-zero (catches the last block's alpha_ee /
      alpha_ex, which would get an all-zero grad through the strip slice).

Run from the repo root:
    PYTHONPATH=. python tests/test_spatial_global_cpu.py
"""
import ast
import copy
import math
import os
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
    Bf = B.to(ct).repeat_interleave(rep, dim=1)
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

from src.models.spatial_mamba import StructureAwareSSM, JiTSpatialMamba   # noqa: E402
from src.models.spatial_global import SpatialGlobalSSM, JiTSpatialGlobal  # noqa: E402

MODES = {"full": True, "writeonly": False}            # name -> readback
DIL = (1, 2)
KW = dict(input_size=16, patch_size=4, hidden_size=32, depth=4, num_classes=7,
          bottleneck_dim=16, gate=False, lpu="zero", sasf_dilations=DIL)
GRID = KW["input_size"] // KW["patch_size"]              # 4 x 4
START = 1                                                # token blocks 1, 2, 3 (3 = last)
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


def perturb_zero_init(model, scale=0.05):
    """Only the adaLN-Zero / LPU-zero / zero-output weights (what makes the
    init gradients 0)."""
    with torch.no_grad():
        mods = [blk.adaLN_modulation[-1] for blk in model.blocks]
        mods += [model.final_layer.adaLN_modulation[-1], model.final_layer.linear]
        mods += [c for blk in model.blocks for c in (getattr(blk, "cpe1", None),
                                                     getattr(blk, "cpe2", None)) if c is not None]
        for mod in mods:
            for p in mod.parameters():
                p.add_(scale * torch.randn_like(p))
    return model


def global_model(M=3, mode="full", seed=0, **extra):
    torch.manual_seed(seed)
    kw = dict(KW)
    kw.update(extra)
    return JiTSpatialGlobal(**kw, global_len=M, global_start=START,
                            global_readback=MODES[mode])


def run(model, x=x_img):
    return model(x, t_in, y_in)


def maxdiff(a, b):
    return (a - b).abs().max().item()


def mixer64(mode="full", emit=True, gate=False, d_model=24, seed=1):
    """A float64 SpatialGlobalSSM with every weight moved off its init and
    random non-unit gains."""
    torch.manual_seed(seed)
    mx = SpatialGlobalSSM(d_model, gate=gate, dilations=DIL, readback=MODES[mode],
                          emit_tokens=emit).double()
    perturb(mx, 0.3)
    with torch.no_grad():
        for name, p in mx.named_parameters():
            if name.startswith("alpha_"):
                p.uniform_(0.5, 1.5)
    return mx


# ═══ A. global_len 0 is the JiTSpatialMamba baseline ════════════════════════
torch.manual_seed(0)
base = JiTSpatialMamba(**KW)
g0 = global_model(M=0)
assert all(type(b.mixer) is StructureAwareSSM for b in g0.blocks)
sd_b, sd_g = base.state_dict(), g0.state_dict()
assert list(sd_b) == list(sd_g)
assert all(torch.equal(sd_b[k], sd_g[k]) for k in sd_b)
assert maxdiff(run(base), run(g0)) == 0
perturb(g0, 0.1)
base.load_state_dict(g0.state_dict(), strict=True)
g0.load_state_dict(base.state_dict(), strict=True)
assert maxdiff(run(base), run(g0)) == 0
print("A  global_len 0: same keys / weights as JiTSpatialMamba, strict both ways, "
      "max|diff| == 0 (also perturbed)")

# ═══ A2. no tokens: SpatialGlobalSSM == StructureAwareSSM ═══════════════════
for gate in (False, True):
    torch.manual_seed(3)
    ss = perturb(StructureAwareSSM(24, d_state=1, gate=gate, dilations=DIL), 0.2)
    for mode in MODES:
        mg = SpatialGlobalSSM(24, gate=gate, dilations=DIL, readback=MODES[mode])
        missing, unexpected = mg.load_state_dict(ss.state_dict(), strict=False)
        assert not unexpected and all(k.startswith("alpha_") for k in missing)
        xg = torch.randn(2, 12, 24)
        assert maxdiff(ss(xg, 3, 4), mg(xg, 3, 4)) == 0, (gate, mode)
print("A2 SpatialGlobalSSM on the grid alone == StructureAwareSSM, max|diff| == 0 (gate on/off)")

# ═══ A3. tokens reuse the in-context scaffolding; same init, same RNG ═══════
torch.manual_seed(0)
pref = JiTSpatialMamba(**KW, in_context_len=3, in_context_start=START)
rng_pref = torch.get_rng_state()
g3 = global_model(M=3)
assert torch.equal(rng_pref, torch.get_rng_state()), "the swap moved the RNG stream"
sd_p, sd_g = pref.state_dict(), g3.state_dict()
assert set(sd_p) <= set(sd_g)
assert all(k.split(".")[-1].startswith("alpha_") for k in set(sd_g) - set(sd_p))
assert all(torch.equal(sd_p[k], sd_g[k]) for k in sd_p)
assert sd_g["incontext_pos_embed"].shape == (1, 3, KW["hidden_size"])
assert [type(b.mixer).__name__ for b in g3.blocks] == \
    ["StructureAwareSSM"] + ["SpatialGlobalSSM"] * 3
print("A3 global_len 3: every shared weight (slots too) == Spatial prefix model at the "
      "same seed; RNG state equal")


# ═══ B. mixer == naive transcription (float64) ══════════════════════════════
def naive_sasf(mx, h, H, W):
    """h: (B, L, Din) raster -> fused, by explicit replicate-clamped sums."""
    Bn, L, Din = h.shape
    g = h.reshape(Bn, H, W, Din)
    out = torch.zeros_like(g)
    sf = mx.state_fusion
    for a, k, d in zip(sf.alpha, sf.kernels, sf.dilations):
        for r in range(H):
            for c in range(W):
                acc = 0
                for i in range(3):
                    for j in range(3):
                        rr = min(max(r + (i - 1) * d, 0), H - 1)
                        cc = min(max(c + (j - 1) * d, 0), W - 1)
                        acc = acc + k[:, 0, i, j] * g[:, rr, cc]
                out[:, r, c] = out[:, r, c] + a * acc
    return out.reshape(Bn, L, Din)


def naive_mixer(mx, x, H, W):
    Bn, L_in, _ = x.shape
    L = H * W
    M = L_in - L
    Din, R, kc = mx.d_inner, mx.dt_rank, mx.d_conv // 2
    xz = x @ mx.in_proj.weight.T
    u, z = (xz[..., :Din], xz[..., Din:]) if mx.gate else (xz, None)
    ue = F.silu(u[:, :M] * mx.conv2d.weight[:, 0, kc, kc] + mx.conv2d.bias)      # (B, M, Din)
    ug = F.silu(F.conv2d(u[:, M:].reshape(Bn, H, W, Din).permute(0, 3, 1, 2),
                         mx.conv2d.weight, mx.conv2d.bias, padding=kc, groups=Din))
    ug = ug.reshape(Bn, Din, L).transpose(1, 2)                                 # (B, L, Din)

    def proj(V):
        p = V @ mx.x_proj_weight.T
        delta = F.softplus(p[..., :R] @ mx.dt_projs_weight.T + mx.dt_projs_bias)
        return delta, p[..., R], p[..., R + 1]
    Dx, Bx, Cx = proj(ug)                     # (B, L, Din), (B, L), (B, L)
    De, Be, Ce = proj(ue)
    vx, ve = Dx * ug, De * ue
    A = -torch.exp(mx.A_logs[:, 0])
    cs = torch.cumsum(Dx, dim=1)
    diff = cs[:, :, None, :] - cs[:, None, :, :]                                # (B, t, s, Din)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))[None, :, :, None]
    decay = torch.where(causal, torch.exp(diff * A), torch.zeros_like(diff))
    states = torch.einsum("btsd,bsd,bs->btd", decay, vx, Bx)
    S = (ve * Be[..., None]).sum(1) / M                                        # (B, Din)
    G = (vx * Bx[..., None]).sum(1) / L
    h = naive_sasf(mx, states + (mx.alpha_xe * S)[:, None, :], H, W)
    y_g = h * Cx[..., None] + mx.Ds * ug
    y_e = mx.Ds * ue
    if mx.emit_tokens:
        mem = mx.alpha_ee * S + (mx.alpha_ex * G if mx.readback else 0)
        y_e = y_e + mem[:, None, :] * Ce[..., None]
    y = torch.cat([y_e, y_g], dim=1)
    y = F.layer_norm(y, (Din,), mx.out_norm.weight, mx.out_norm.bias, mx.out_norm.eps)
    if z is not None:
        y = y * F.silu(z)
    return y @ mx.out_proj.weight.T


Hr, Wr, Mr = 3, 4, 3                                       # rectangular grid
torch.manual_seed(4)
x_mix = torch.randn(2, Mr + Hr * Wr, 24, dtype=torch.float64)
with torch.no_grad():
    for mode in MODES:
        for emit in (True, False):
            for gate in (False, True):
                mx = mixer64(mode, emit=emit, gate=gate)
                d = maxdiff(mx(x_mix, Hr, Wr), naive_mixer(mx, x_mix, Hr, Wr))
                assert d < 1e-10, (mode, emit, gate, d)
print("B  mixer == naive transcription (float64, 3x4 grid, M=3; full / write-only, "
      "emit_tokens on/off, gate on/off)")

# ═══ C. anchored recurrence == states + a S ═════════════════════════════════
with torch.no_grad():
    mx = mixer64("full")
    t = mx.terms(x_mix, Hr, Wr)
    delta = F.softplus(t["dt"] + mx.dt_projs_bias[None, :, None])      # (B, Din, L)
    A = -torch.exp(mx.A_logs[:, 0])
    aS = t["anchor"][..., 0]                                            # (B, Din)
    xp, worst = aS.clone(), 0.0
    for l in range(Hr * Wr):
        Abar = torch.exp(delta[:, :, l] * A)
        xp = Abar * xp + (1 - Abar) * aS + delta[:, :, l] * t["u_g"][:, :, l] * t["B_x"][:, 0, l, None]
        worst = max(worst, maxdiff(xp, t["states"][:, :, l] + aS))
    assert worst < 1e-12, worst
    fused = mx.state_fusion((t["states"] + t["anchor"]).view(2, -1, Hr, Wr)).reshape(2, -1, Hr * Wr)
    assert maxdiff(fused, t["h"]) == 0
print(f"C  anchored recurrence (x'_0 = aS, pulled to aS) == states + aS, max|diff| {worst:.1e}; "
      "SASF fuses exactly that")

# ═══ C2. SASF of a constant map is a per-channel scale ══════════════════════
with torch.no_grad():
    sf = mixer64("full").state_fusion
    const = torch.randn(2, sf.dim, 1, 1, dtype=torch.float64).expand(2, sf.dim, Hr, Wr)
    kappa = sum(a * k.sum(dim=(1, 2, 3)) for a, k in zip(sf.alpha, sf.kernels))
    assert maxdiff(sf(const.contiguous()), kappa[None, :, None, None] * const) < 1e-12
    ones_sf = SpatialGlobalSSM(24, dilations=(1, 2, 3)).state_fusion
    k0 = sum(a * k.sum(dim=(1, 2, 3)) for a, k in zip(ones_sf.alpha, ones_sf.kernels))
    assert torch.all(k0 == 27)
print("C2 SASF of a constant map == kappa * map, kappa = sum_j alpha_j sum(taps_j) "
      "(27 at the ones-init with 3 dilations)")

# ═══ D. token permutation ═══════════════════════════════════════════════════
perm = torch.tensor([2, 0, 1])
with torch.no_grad():
    for mode in MODES:
        for gate in (False, True):
            mx = mixer64(mode, gate=gate)
            o = mx(x_mix, Hr, Wr)
            op = mx(torch.cat([x_mix[:, :Mr][:, perm], x_mix[:, Mr:]], 1), Hr, Wr)
            assert maxdiff(op[:, Mr:], o[:, Mr:]) < 1e-12, mode
            assert maxdiff(op[:, :Mr], o[:, :Mr][:, perm]) < 1e-12, mode
    for mode in MODES:
        gm = perturb(global_model(M=3, mode=mode), 0.05)
        o = run(gm)
        gm.incontext_pos_embed.copy_(gm.incontext_pos_embed[:, perm])
        rel = maxdiff(run(gm), o) / o.abs().max().item()
        assert rel < 1e-5, (mode, rel)
print("D  permuting tokens: image rows unchanged, token rows permuted (mixer, float64, 1e-12); "
      "model output unchanged under slot permutation (float32, rel < 1e-5)")

# ═══ E. S and G do not depend on the order ══════════════════════════════════
with torch.no_grad():
    mx = mixer64("full")
    t = mx.terms(x_mix, Hr, Wr)
    vx = F.softplus(t["dt"] + mx.dt_projs_bias[None, :, None]) * t["u_g"]
    G_loop = torch.zeros_like(t["G"])
    for r in range(Hr):
        for c in range(Wr):
            j = r * Wr + c
            G_loop[..., 0] += vx[..., j] * t["B_x"][:, 0, j, None]
    assert maxdiff(G_loop / (Hr * Wr), t["G"]) < 1e-12
    tp = mx.terms(torch.cat([x_mix[:, :Mr][:, perm], x_mix[:, Mr:]], 1), Hr, Wr)
    assert maxdiff(tp["S"], t["S"]) < 1e-12 and maxdiff(tp["G"], t["G"]) < 1e-12
print("E  G == raster loop; S invariant to token order (1e-12)")

# ═══ F. D is counted exactly once ═══════════════════════════════════════════
with torch.no_grad():
    for mode in MODES:
        mx = mixer64(mode)
        t1 = mx.terms(x_mix, Hr, Wr)
        Ds = mx.Ds.clone()
        mx.Ds.zero_()
        t0 = mx.terms(x_mix, Hr, Wr)
        assert maxdiff(t1["y_g"] - t0["y_g"], Ds[None, :, None] * t1["u_g"]) < 1e-12, mode
        assert maxdiff(t1["y_e"] - t0["y_e"], Ds[None, :, None] * t1["u_e"]) < 1e-12, mode
    mx = mixer64("full")
    mx.x_proj_weight[mx.dt_rank].zero_()                    # B = 0: no write anywhere
    t = mx.terms(x_mix, Hr, Wr)
    assert t["S"].abs().max() == 0 and t["G"].abs().max() == 0 and t["h"].abs().max() == 0
    assert maxdiff(t["y_g"], mx.Ds[None, :, None] * t["u_g"]) < 1e-12
    assert maxdiff(t["y_e"], mx.Ds[None, :, None] * t["u_e"]) < 1e-12
print("F  D enters once: dD changes image rows by D u and token rows by D u^e; "
      "with B = 0 the output is the D term alone")

# ═══ G. bf16 autocast, fp32 memories, gradient flow ═════════════════════════
for mode in MODES:
    gm = perturb(global_model(M=3, mode=mode), 0.05)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = run(gm)
    loss = out.float().square().mean()
    loss.backward()
    assert torch.isfinite(out.float()).all() and torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in gm.parameters())
    mxg = gm.blocks[START].mixer
    xb = torch.randn(2, 3 + GRID * GRID, KW["hidden_size"])
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        tb = mxg.terms(xb, GRID, GRID)
    for key in ("ve", "S", "G", "anchor", "y_e"):
        if tb[key] is not None:
            assert tb[key].dtype == torch.float32, (mode, key, tb[key].dtype)
    gm = perturb_zero_init(global_model(M=3, mode=mode))
    run(gm).square().mean().backward()
    assert gm.incontext_pos_embed.grad.abs().sum() > 0
    for i in range(START, KW["depth"]):
        m_i = gm.blocks[i].mixer
        for name, p in m_i.named_parameters():
            if name.startswith("alpha_"):
                assert p.grad is not None and p.grad.abs().sum() > 0, (mode, i, name)
        assert m_i.conv2d.weight.grad[:, 0, 1, 1].abs().sum() > 0
        assert m_i.conv2d.bias.grad.abs().sum() > 0
mx = mixer64("writeonly")
xw = x_mix.clone().requires_grad_(True)
mx.terms(xw, Hr, Wr)["y_e"].sum().backward()
assert mx.x_proj_weight.grad.abs().sum() > 0
assert xw.grad[:, Mr:].abs().max() == 0 and xw.grad[:, :Mr].abs().sum() > 0
print("G  bf16 autocast fwd+bwd finite (2 modes); ve/S/G/anchor/y_e fp32; grads reach slots, "
      "alpha_*, centre tap + bias; write-only token loss reaches x_proj, never the image")

# ═══ H. parameter counts, no decay ══════════════════════════════════════════
from src.utils import add_weight_decay                      # noqa: E402
d_in = KW["hidden_size"]
torch.manual_seed(0)
nb = n_params(JiTSpatialMamba(**KW))
for mode, n_alpha in (("full", 3), ("writeonly", 2)):
    gm = global_model(M=3, mode=mode)
    for i, blk in enumerate(gm.blocks):
        names = sorted(n for n, _ in blk.mixer.named_parameters() if n.startswith("alpha_"))
        want = ([] if i < START else
                ["alpha_xe"] if i == KW["depth"] - 1 else
                sorted(["alpha_xe", "alpha_ee"] + (["alpha_ex"] if n_alpha == 3 else [])))
        assert names == want, (mode, i, names)
    want_total = nb + 3 * d_in + (KW["depth"] - 1 - START) * n_alpha * d_in + d_in
    assert n_params(gm) == want_total, (mode, n_params(gm), want_total)
    no_decay = {id(p) for p in add_weight_decay(gm, 0.05)[0]["params"]}
    alphas = [p for n, p in gm.named_parameters() if ".alpha_" in n]
    assert alphas and all(id(p) in no_decay for p in alphas), mode
print("H  gains: none before global_start, alpha_xe only in the last block, +3 / +2 x d_inner "
      "elsewhere; totals match; every alpha_* in the no-decay group")

# ═══ H2. last block drops only dead work ════════════════════════════════════
with torch.no_grad():
    for mode in MODES:
        for gate in (False, True):
            mt = mixer64(mode, emit=True, gate=gate)
            mf = SpatialGlobalSSM(24, gate=gate, dilations=DIL, readback=MODES[mode],
                                  emit_tokens=False).double()
            missing, unexpected = mf.load_state_dict(mt.state_dict(), strict=False)
            assert not missing and set(unexpected) <= {"alpha_ee", "alpha_ex"}
            assert maxdiff(mt(x_mix, Hr, Wr)[:, Mr:], mf(x_mix, Hr, Wr)[:, Mr:]) == 0, mode
            tf = mf.terms(x_mix, Hr, Wr)
            assert tf["G"] is None
            assert maxdiff(tf["y_e"], mf.Ds[None, :, None] * tf["u_e"]) == 0, mode
print("H2 emit_tokens off: image rows identical (== 0), token rows = D u^e, no G")

# ═══ I. strict loading ══════════════════════════════════════════════════════
full_sd = global_model(M=3, mode="full").state_dict()
for target, source in ((global_model(M=3, mode="writeonly"), full_sd),
                       (global_model(M=3), pref.state_dict())):
    try:
        target.load_state_dict(source, strict=True)
    except RuntimeError:
        continue
    raise AssertionError("strict load should have failed")
print("I  write-only <- full and global <- Spatial prefix fail strictly")

# ═══ J. guards ══════════════════════════════════════════════════════════════
bad = [dict(in_context_len=2), dict(in_context_positions=[[0, 0]]),
       dict(in_context_content="time_class"), dict(aux_tokens="swap"),
       dict(dwconv_can=True), dict(sasf_can=True), dict(ssm_norm="dt"),
       dict(feature_rope="1d"), dict(sasf_state_kernels="per_state"), dict(d_state=2),
       dict(global_len=-1), dict(global_len=True), dict(global_start=KW["depth"]),
       dict(global_start=-1), dict(global_readback="yes")]
for kw in bad:
    args = dict(KW, global_len=3, global_start=START)
    args.update(kw)
    try:
        JiTSpatialGlobal(**args)
    except (AssertionError, ValueError):
        continue
    raise AssertionError(f"guard missing for {kw}")
print(f"J  guards: {len(bad)} forbidden flags / bad values raise")

# ═══ K. both build_model copies agree on the three configs ══════════════════
try:
    import yaml                                             # noqa: E402
except ImportError as e:                                    # pragma: no cover
    print(f"K  SKIPPED (pyyaml not importable here: {e})")
    yaml = None


def load_build_model(rel):
    path = os.path.join(REPO_ROOT, rel)
    src = open(path, encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_model")
    ns = {}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns["build_model"]


TINY_IN = {"full": 29_700_708, "writeonly": 29_698_020, "full-m32": 29_706_852}
if yaml is not None:
    bm_train = load_build_model("scripts/run_experiment.py")
    bm_eval = load_build_model("scripts/evaluate.py")
    torch.manual_seed(0)
    x64 = torch.randn(1, 3, 64, 64)
    t64, y64 = torch.rand(1), torch.randint(0, 200, (1,))
    for name, want in TINY_IN.items():
        cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "tiny_imagenet",
                                               f"jit-s2-spatial-nogate-global-{name}.yaml"),
                                  encoding="utf-8"))
        assert cfg["experiment"]["model"] == "spatial_global"
        torch.manual_seed(0)
        mt = bm_train(cfg)
        torch.manual_seed(0)
        me = bm_eval(cfg)
        sd_t, sd_e = mt.state_dict(), me.state_dict()
        assert list(sd_t) == list(sd_e) and all(torch.equal(sd_t[k], sd_e[k]) for k in sd_t)
        assert n_params(mt) == want, (name, n_params(mt))
        assert not mt.gate and mt.lpu == "zero" and mt.sasf_dilations == (1, 2, 3)
        perturb_zero_init(mt)
        me.load_state_dict(mt.state_dict(), strict=True)
        with torch.no_grad():
            assert maxdiff(mt(x64, t64, y64), me(x64, t64, y64)) == 0
        bad_cfg = copy.deepcopy(cfg)
        bad_cfg["model"]["aux_tokens"] = "swap"
        for bm in (bm_train, bm_eval):
            try:
                bm(bad_cfg)
            except AssertionError:
                continue
            raise AssertionError("build_model accepted aux_tokens under spatial_global")
        print(f"K  jit-s2-spatial-nogate-global-{name}.yaml: train == eval build_model, "
              f"{n_params(mt):,} params; aux_tokens rejected by both")

# ═══ L. identity at init ════════════════════════════════════════════════════
for mode in MODES:
    gm = global_model(M=3, mode=mode)
    seen = []

    def hook(mod, inp, out):
        seen.append(torch.equal(inp[0], out))
    hs = [b.register_forward_hook(hook) for b in gm.blocks]
    with torch.no_grad():
        run(gm)
    for h in hs:
        h.remove()
    assert seen == [True] * KW["depth"], (mode, seen)
print("L  every block is the identity at init (adaLN-Zero, LPU zero), 2 modes")

# ═══ M. ffn / norm / gate variants, grad checkpointing ══════════════════════
for extra in (dict(ffn="convglu"), dict(ffn="glumbconv"), dict(norm="derf"), dict(gate=True)):
    gm = perturb(global_model(M=3, **extra), 0.05)
    out = run(gm)
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in gm.parameters())
plain = perturb(global_model(M=3), 0.05)
ckpt = global_model(M=3, grad_checkpoint=True)
ckpt.load_state_dict(plain.state_dict(), strict=True)
outs, grads = [], []
for gm in (plain, ckpt):
    gm.train()
    o = run(gm)
    o.square().mean().backward()
    outs.append(o.detach())
    grads.append({n: p.grad.clone() for n, p in gm.named_parameters() if p.grad is not None})
assert torch.equal(outs[0], outs[1])
assert grads[0].keys() == grads[1].keys() and all(torch.equal(grads[0][k], grads[1][k]) for k in grads[0])
print("M  ffn convglu / glumbconv, norm derf, gate on: fwd+bwd finite; grad_checkpoint "
      "bit-identical (output and every grad)")

# ═══ N. no dead parameters (what DDP checks) ════════════════════════════════
cases = [("full", 3, {}), ("writeonly", 3, {}), ("full", 6, {}), ("full", 3, dict(gate=True))]
for mode, M, extra in cases:
    gm = global_model(M=M, mode=mode, seed=5, **extra)
    run(gm).mul(torch.randn(Bsz, 3, KW["input_size"], KW["input_size"])).sum().backward()
    dead = [n for n, p in gm.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, (mode, M, dead)
    gm = perturb_zero_init(global_model(M=M, mode=mode, seed=5, **extra))
    run(gm).mul(torch.randn(Bsz, 3, KW["input_size"], KW["input_size"])).sum().backward()
    zero = [n for n, p in gm.named_parameters() if p.requires_grad and p.grad.abs().sum() == 0]
    assert not zero, (mode, M, extra, zero)
print("N  no dead parameters: grad is not None at init, non-zero after perturbing the "
      "zero-init weights (full / write-only at M=3, full at M=6, gate on)")

print("\nALL spatial_global CPU checks passed.")
