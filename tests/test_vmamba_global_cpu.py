"""
tests/test_vmamba_global_cpu.py
===============================
CPU-only verification of JiT-S2-VMamba-Global (src/models/vmamba_global.py):
SS2D with M global condition tokens that never enter the scan. They talk to
the image through undecayed per-direction memories S_k (tokens -> memory) and
G_k (image -> memory) built from the block's own x_proj / dt_proj.

Runs WITHOUT a GPU and WITHOUT mamba_ssm (reference selective scan stub, kept
in float64 when fed float64 so the fidelity checks are exact). Mixer proofs
run in float64; model-level checks in float32 (TimestepEmbedder is fp32).

Checks:
  A.  global_len 0: every mixer is SS2D and the model IS JiTVMamba's baseline
      (same keys, same weights at the same seed, strict load both ways,
      max|diff| == 0, also with every weight perturbed and transplanted).
  A2. SS2DGlobal fed only the grid == SS2D with the same weights (== 0).
  A3. global_len 3: every shared weight (slots included) equals
      JiTVMamba(in_context_len=3, content class, same start) at the same
      seed, the RNG state after construction is equal (fork_rng), and only
      alpha_* keys are new.
  B.  the mixer == a naive float64 transcription of the four-block matrix
        [y_e]   [ a_ee/M C_e B_e^T   a_ex/L C_e B_x^T ] [v_e]
        [y_x] = [ a_xe/M C_x B_e^T   K_xx             ] [v_x] + D terms
      K_xx[t,s] = sum_n C_tn 1[s<=t] exp(A_dn sum_{r=s+1..t} Delta_r) B_sn,
      per channel, 4 directions put back through coordinate index maps (not
      cross_merge), LayerNorm, out_proj. Rectangular 3x4 grid, random
      non-unit gains; full, write-only and softmax.
  C.  anchored recurrence h_t = Abar h + (1 - Abar)(a S) + v B^T, h_0 = a S,
      naive loop == scan + (a S) C (the pre-merge image paths).
  D.  permuting the tokens (mixer rows; model slots) leaves the image outputs
      unchanged and permutes the token outputs.
  E.  S_k, G_k from path order == from canonical order == a naive raster
      loop; S invariant to token order.
  F.  D counted once: Ds -> 0 changes the image paths by exactly D_k u and
      the token rows by sum_k D_k u^e; with x_proj's B rows zeroed the
      output is the D term alone.
  G.  bf16 autocast: forward + backward finite (3 modes), memories / reads
      fp32; gradients reach the slots, gains and conv centre tap + bias;
      in write-only mode a token-only loss reaches x_proj and NOT the image.
  H.  parameter counts: +3/+2 x d_inner per block, +d_inner in the last
      block, none before global_start; the four Tiny-IN totals.
  H2. last block (emit_tokens False) vs True on the same weights: image rows
      identical (== 0), token rows the D term only.
  I.  strict loading: full <-> softmax load; write-only <- full and global <-
      prefix fail.
  J.  guards: every forbidden flag and bad value raises.
  K.  wiring: both build_models build IDENTICAL models from the four configs,
      and both reject a vmamba.py arm under vmamba_global.
  L.  every block is the identity at init (adaLN-Zero), every mode.
  M.  ffn convglu / glumbconv and norm derf: build, forward + backward
      finite; the conv FFN's token rows are the centre-tap formula.
  N.  no dead parameters (4 configs x modes): at the real init every
      requires_grad parameter has .grad not None (what DDP checks: a gain
      in blocks 0..start-1, or write-only's alpha_ex, would be None); after
      perturbing the zero-init weights every .grad is non-zero (catches the
      last block's alpha_ee / alpha_ex, which sit in the graph through the
      strip slice and so get an all-zero grad, not None).

Run from the repo root:
    PYTHONPATH=. python tests/test_vmamba_global_cpu.py
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

from src.models.vmamba import SS2D, JiTVMamba, cross_merge                # noqa: E402
from src.models.vmamba_sasf import cross_unscan                           # noqa: E402
from src.models.vmamba_global import SS2DGlobal, JiTVMambaGlobal          # noqa: E402

MODES = {                       # name -> (readback, kernel_xe)
    "full": (True, "linear"),
    "writeonly": (False, "linear"),
    "softmax": (True, "softmax"),
}
KW = dict(input_size=16, patch_size=4, hidden_size=32, depth=4, num_classes=7,
          bottleneck_dim=16, d_state=4)
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
    """Only the adaLN-Zero / zero-output weights (what makes init gradients 0)."""
    with torch.no_grad():
        for blk in model.blocks:
            for p in blk.adaLN_modulation[-1].parameters():
                p.add_(scale * torch.randn_like(p))
        for p in model.final_layer.adaLN_modulation[-1].parameters():
            p.add_(scale * torch.randn_like(p))
        for p in model.final_layer.linear.parameters():
            p.add_(scale * torch.randn_like(p))
    return model


def global_model(M=3, mode="full", seed=0, **extra):
    readback, kernel = MODES[mode]
    torch.manual_seed(seed)
    return JiTVMambaGlobal(**KW, global_len=M, global_start=START,
                           global_readback=readback, global_kernel_xe=kernel, **extra)


def run(model, x=x_img):
    return model(x, t_in, y_in)


def maxdiff(a, b):
    return (a - b).abs().max().item()


def mixer64(mode="full", emit=True, d_model=24, d_state=4, seed=1):
    """A float64 SS2DGlobal with every weight moved off its init and
    random non-unit gains."""
    readback, kernel = MODES[mode]
    torch.manual_seed(seed)
    mx = SS2DGlobal(d_model, d_state=d_state, readback=readback,
                    kernel_xe=kernel, emit_tokens=emit).double()
    perturb(mx, 0.3)
    with torch.no_grad():
        for name, p in mx.named_parameters():
            if name.startswith("alpha_"):
                p.uniform_(0.5, 1.5)
    return mx


# ═══ A. global_len 0 is the JiTVMamba baseline ══════════════════════════════
torch.manual_seed(0)
base = JiTVMamba(**KW)
g0 = global_model(M=0)
assert all(type(b.mixer) is SS2D for b in g0.blocks)
sd_b, sd_g = base.state_dict(), g0.state_dict()
assert list(sd_b) == list(sd_g)
assert all(torch.equal(sd_b[k], sd_g[k]) for k in sd_b)
assert maxdiff(run(base), run(g0)) == 0
perturb(g0, 0.1)
base.load_state_dict(g0.state_dict(), strict=True)
g0.load_state_dict(base.state_dict(), strict=True)
assert maxdiff(run(base), run(g0)) == 0
print("A  global_len 0: same keys / weights as JiTVMamba, strict both ways, max|diff| == 0 (also perturbed)")

# ═══ A2. no tokens: SS2DGlobal == SS2D ══════════════════════════════════════
torch.manual_seed(3)
ss = perturb(SS2D(24, d_state=4), 0.2)
for mode in MODES:
    mg = SS2DGlobal(24, d_state=4, readback=MODES[mode][0], kernel_xe=MODES[mode][1])
    mg.load_state_dict(ss.state_dict(), strict=False)
    xg = torch.randn(2, 12, 24)
    assert maxdiff(ss(xg, 3, 4), mg(xg, 3, 4)) == 0, mode
print("A2 SS2DGlobal on the grid alone == SS2D, max|diff| == 0 (3 modes)")

# ═══ A3. tokens reuse the prefix scaffolding; same init, same RNG ═══════════
torch.manual_seed(0)
pref = JiTVMamba(**KW, in_context_len=3, in_context_start=START, in_context_content="class")
rng_pref = torch.get_rng_state()
g3 = global_model(M=3)
rng_glob = torch.get_rng_state()
assert torch.equal(rng_pref, rng_glob), "the swap moved the RNG stream"
sd_p, sd_g = pref.state_dict(), g3.state_dict()
assert set(sd_p) <= set(sd_g)
assert all(k.split(".")[-1].startswith("alpha_") for k in set(sd_g) - set(sd_p))
assert all(torch.equal(sd_p[k], sd_g[k]) for k in sd_p)
assert sd_g["incontext_pos_embed"].shape == (1, 3, KW["hidden_size"])
assert [type(b.mixer).__name__ for b in g3.blocks] == ["SS2D", "SS2DGlobal", "SS2DGlobal", "SS2DGlobal"]
print("A3 global_len 3: every shared weight (slots too) == prefix model at the same seed; RNG state equal")


# ═══ B. mixer == naive four-block transcription (float64) ═══════════════════
def path_orders(H, W):
    p0 = [r * W + c for r in range(H) for c in range(W)]
    p2 = [r * W + c for c in range(W) for r in range(H)]
    return [p0, p0[::-1], p2, p2[::-1]]


def naive_mixer(mx, x, H, W):
    Bn, L_in, _ = x.shape
    L = H * W
    M = L_in - L
    Din, N, R, kc = mx.d_inner, mx.d_state, mx.dt_rank, mx.d_conv // 2
    z = x @ mx.in_proj.weight.T
    ze, zg = z[:, :M], z[:, M:]
    ug = F.silu(F.conv2d(zg.reshape(Bn, H, W, Din).permute(0, 3, 1, 2), mx.conv2d.weight,
                         mx.conv2d.bias, padding=kc, groups=Din))
    ug = ug.reshape(Bn, Din, L).transpose(1, 2)                       # (B, L, Din) raster
    ue = F.silu(ze * mx.conv2d.weight[:, 0, kc, kc] + mx.conv2d.bias)  # (B, M, Din)
    out_x = torch.zeros(Bn, L, Din, dtype=x.dtype)
    out_e = torch.zeros(Bn, M, Din, dtype=x.dtype)
    a_xe = mx.alpha_xe
    for k, order in enumerate(path_orders(H, W)):
        U = ug[:, order]

        def proj(V):
            p = V @ mx.x_proj_weight[k].T
            delta = F.softplus(p[..., :R] @ mx.dt_projs_weight[k].T + mx.dt_projs_bias[k])
            return delta, p[..., R:R + N], p[..., R + N:]
        Dx, Bx, Cx = proj(U)
        De, Be, Ce = proj(ue)
        vx, ve = Dx * U, De * ue
        A = -torch.exp(mx.A_logs[k])                                   # (Din, N)
        T = torch.zeros(Bn, Din, M + L, M + L, dtype=x.dtype)
        if mx.emit_tokens:
            T[:, :, :M, :M] = (mx.alpha_ee[None, :, None, None] / M) * (Ce @ Be.transpose(1, 2))[:, None]
            if mx.readback:
                T[:, :, :M, M:] = (mx.alpha_ex[None, :, None, None] / L) * (Ce @ Bx.transpose(1, 2))[:, None]
        if mx.kernel_xe == "linear":
            T[:, :, M:, :M] = (a_xe[None, :, None, None] / M) * (Cx @ Be.transpose(1, 2))[:, None]
        else:
            P = torch.softmax(Cx @ Be.transpose(1, 2) / math.sqrt(N), dim=-1)
            T[:, :, M:, :M] = a_xe[None, :, None, None] * P[:, None]
        cs = torch.cumsum(Dx, dim=1)
        diff = cs[:, :, None, :] - cs[:, None, :, :]                  # (B, t, s, Din) = sum_{s<r<=t}
        expo = torch.einsum("btsd,dn->bdtsn", diff, A)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))[None, None, :, :, None]
        expo = torch.where(causal, expo, torch.full_like(expo, -math.inf))
        T[:, :, M:, M:] = torch.einsum("btn,bdtsn,bsn->bdts", Cx, torch.exp(expo), Bx)
        vec = torch.cat([ve, vx], dim=1).transpose(1, 2)              # (B, Din, M+L)
        y = torch.einsum("bdij,bdj->bdi", T, vec)
        y = y + mx.Ds[k][None, :, None] * torch.cat([ue, U], dim=1).transpose(1, 2)
        out_e = out_e + y[:, :, :M].transpose(1, 2)
        out_x[:, order] = out_x[:, order] + y[:, :, M:].transpose(1, 2)
    out = torch.cat([out_e, out_x], dim=1)
    out = F.layer_norm(out, (Din,), mx.out_norm.weight, mx.out_norm.bias, mx.out_norm.eps)
    return out @ mx.out_proj.weight.T


Hr, Wr, Mr = 3, 4, 3                                       # rectangular grid
torch.manual_seed(4)
x_mix = torch.randn(2, Mr + Hr * Wr, 24, dtype=torch.float64)
with torch.no_grad():
    for mode in MODES:
        for emit in (True, False):
            mx = mixer64(mode, emit=emit)
            d = maxdiff(mx(x_mix, Hr, Wr), naive_mixer(mx, x_mix, Hr, Wr))
            assert d < 1e-10, (mode, emit, d)
print("B  mixer == naive four-block matrix transcription (float64, 3x4 grid, M=3; "
      "full / write-only / softmax, emit_tokens on and off)")

# ═══ C. anchored recurrence == scan + (a S) C ═══════════════════════════════
with torch.no_grad():
    mx = mixer64("full")
    t = mx.terms(x_mix, Hr, Wr)
    worst = 0.0
    for k in range(4):
        delta = F.softplus(t["dt"][:, k] + mx.dt_projs_bias[k][None, :, None])  # (B, Din, L)
        A = -torch.exp(mx.A_logs[k])
        aS = mx.alpha_xe[None, :, None] * t["S"][:, k]                        # (B, Din, N)
        h = aS.clone()
        for l in range(Hr * Wr):
            Abar = torch.exp(delta[:, :, l, None] * A[None])
            u_l = t["xs"][:, k, :, l]
            h = Abar * h + (1 - Abar) * aS + (delta[:, :, l] * u_l)[..., None] * t["B_x"][:, k, None, :, l]
            y_l = (h * t["C_x"][:, k, None, :, l]).sum(-1) + mx.Ds[k][None] * u_l
            worst = max(worst, maxdiff(y_l, t["ys"][:, k, :, l]))
    assert worst < 1e-12, worst
print(f"C  anchored recurrence (h_0 = a S, pulled to a S) == scan + (a S) C, max|diff| {worst:.1e}")

# ═══ D. token permutation ═══════════════════════════════════════════════════
perm = torch.tensor([2, 0, 1])
with torch.no_grad():
    for mode in MODES:
        mx = mixer64(mode)
        o = mx(x_mix, Hr, Wr)
        op = mx(torch.cat([x_mix[:, :Mr][:, perm], x_mix[:, Mr:]], 1), Hr, Wr)
        assert maxdiff(op[:, Mr:], o[:, Mr:]) < 1e-12, mode
        assert maxdiff(op[:, :Mr], o[:, :Mr][:, perm]) < 1e-12, mode
    for mode in MODES:
        gm = perturb(global_model(M=3, mode=mode), 0.05)
        o = run(gm)
        with torch.no_grad():
            gm.incontext_pos_embed.copy_(gm.incontext_pos_embed[:, perm])
        rel = maxdiff(run(gm), o) / o.abs().max().item()
        assert rel < 1e-5, (mode, rel)
print("D  permuting tokens: image rows unchanged, token rows permuted (mixer, float64, 1e-12); "
      "model output unchanged under slot permutation (float32, rel < 1e-5)")

# ═══ E. S and G do not depend on the scan order ═════════════════════════════
with torch.no_grad():
    mx = mixer64("full")
    t = mx.terms(x_mix, Hr, Wr)
    L = Hr * Wr
    vx = F.softplus(t["dt"] + mx.dt_projs_bias[None, :, :, None]) * t["xs"]
    G_can = torch.einsum("bkdl,bknl->bkdn", cross_unscan(vx, Hr, Wr),
                         cross_unscan(t["B_x"], Hr, Wr)) / L
    assert maxdiff(G_can, t["G"]) < 1e-12
    vxc, Bxc = cross_unscan(vx, Hr, Wr), cross_unscan(t["B_x"], Hr, Wr)
    G_loop = torch.zeros_like(t["G"])
    for r in range(Hr):
        for c in range(Wr):
            j = r * Wr + c
            G_loop += vxc[..., j][..., None] * Bxc[..., j][:, :, None, :]
    assert maxdiff(G_loop / L, t["G"]) < 1e-12
    tp = mx.terms(torch.cat([x_mix[:, :Mr][:, perm], x_mix[:, Mr:]], 1), Hr, Wr)
    assert maxdiff(tp["S"], t["S"]) < 1e-12 and maxdiff(tp["G"], t["G"]) < 1e-12
print("E  G_k from path order == canonical order == raster loop; S_k invariant to token order (1e-12)")

# ═══ F. D is counted exactly once ═══════════════════════════════════════════
with torch.no_grad():
    for mode in MODES:
        mx = mixer64(mode)
        t1 = mx.terms(x_mix, Hr, Wr)
        Ds = mx.Ds.clone()
        mx.Ds.zero_()
        t0 = mx.terms(x_mix, Hr, Wr)
        assert maxdiff(t1["ys"] - t0["ys"], Ds[None, :, :, None] * t1["xs"]) < 1e-12, mode
        dtok = (Ds[None, :, :, None] * t1["u_e"][:, None]).sum(1)
        assert maxdiff(t1["y_e"] - t0["y_e"], dtok) < 1e-12, mode
    mx = mixer64("full")
    R, N = mx.dt_rank, mx.d_state
    mx.x_proj_weight[:, R:R + N].zero_()                    # B = 0: no write anywhere
    t = mx.terms(x_mix, Hr, Wr)
    assert t["S"].abs().max() == 0 and t["G"].abs().max() == 0
    assert maxdiff(t["ys"], mx.Ds[None, :, :, None] * t["xs"]) < 1e-12
    u_can = t["xs"][:, 0]                                    # path 0 is raster order
    assert maxdiff(cross_merge(t["ys"], Hr, Wr), mx.Ds.sum(0)[None, :, None] * u_can) < 1e-12
    assert maxdiff(t["y_e"], (mx.Ds[None, :, :, None] * t["u_e"][:, None]).sum(1)) < 1e-12
print("F  D enters once: dD changes image paths by D_k u and tokens by sum_k D_k u^e; "
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
    for key in ("ve", "S", "G", "read", "y_e"):
        if tb[key] is not None:
            assert tb[key].dtype == torch.float32, (mode, key, tb[key].dtype)
    # gradients through the tokens (fp32, zero-init weights perturbed)
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
# write-only: a token-only loss reaches x_proj and nothing in the image rows
mx = mixer64("writeonly")
xw = x_mix.clone().requires_grad_(True)
mx.terms(xw, Hr, Wr)["y_e"].sum().backward()
assert mx.x_proj_weight.grad.abs().sum() > 0
assert xw.grad[:, Mr:].abs().max() == 0 and xw.grad[:, :Mr].abs().sum() > 0
print("G  bf16 autocast fwd+bwd finite (3 modes); ve/S/G/read/y_e fp32; grads reach slots, "
      "alpha_*, centre tap + bias; write-only token loss reaches x_proj, never the image")

# ═══ H. parameter counts ════════════════════════════════════════════════════
d_in = KW["hidden_size"]
for mode, n_alpha in (("full", 3), ("writeonly", 2), ("softmax", 3)):
    gm = global_model(M=3, mode=mode)
    for i, blk in enumerate(gm.blocks):
        names = sorted(n for n, _ in blk.mixer.named_parameters() if n.startswith("alpha_"))
        want = ([] if i < START else
                ["alpha_xe"] if i == KW["depth"] - 1 else
                sorted(["alpha_xe", "alpha_ee"] + (["alpha_ex"] if n_alpha == 3 else [])))
        assert names == want, (mode, i, names)
    torch.manual_seed(0)
    nb = n_params(JiTVMamba(**KW))
    want_total = nb + 3 * d_in + (KW["depth"] - 1 - START) * n_alpha * d_in + d_in
    assert n_params(gm) == want_total, (mode, n_params(gm), want_total)
    from src.utils import add_weight_decay
    no_decay = {id(p) for p in add_weight_decay(gm, 0.05)[0]["params"]}
    alphas = [p for n, p in gm.named_parameters() if ".alpha_" in n]
    assert alphas and all(id(p) in no_decay for p in alphas), mode
print("H  gains: none before global_start, alpha_xe only in the last block, "
      "+3 / +2 x d_inner elsewhere; totals match; every alpha_* in the no-decay group")

# ═══ H2. last block drops only dead work ════════════════════════════════════
with torch.no_grad():
    for mode in MODES:
        mt = mixer64(mode, emit=True)
        mf = SS2DGlobal(24, d_state=4, readback=MODES[mode][0], kernel_xe=MODES[mode][1],
                        emit_tokens=False).double()
        missing, unexpected = mf.load_state_dict(mt.state_dict(), strict=False)
        assert not missing and set(unexpected) <= {"alpha_ee", "alpha_ex"}
        assert maxdiff(mt(x_mix, Hr, Wr)[:, Mr:], mf(x_mix, Hr, Wr)[:, Mr:]) == 0, mode
        tf = mf.terms(x_mix, Hr, Wr)
        assert tf["G"] is None and (tf["S"] is None) == (mode == "softmax")
        dterm = (mf.Ds[None, :, :, None] * tf["u_e"][:, None]).sum(1)
        assert maxdiff(tf["y_e"], dterm) < 1e-12, mode
print("H2 emit_tokens off: image rows identical (== 0), token rows = sum_k D_k u^e; "
      "no G (and no S under softmax)")

# ═══ I. strict loading ══════════════════════════════════════════════════════
full_sd = global_model(M=3, mode="full").state_dict()
global_model(M=3, mode="softmax").load_state_dict(full_sd, strict=True)
global_model(M=3, mode="full").load_state_dict(global_model(M=3, mode="softmax").state_dict(), strict=True)
for target, source in ((global_model(M=3, mode="writeonly"), full_sd), (global_model(M=3), pref.state_dict())):
    try:
        target.load_state_dict(source, strict=True)
    except RuntimeError:
        continue
    raise AssertionError("strict load should have failed")
print("I  full <-> softmax load strictly (same shapes); write-only <- full and global <- prefix fail")

# ═══ J. guards ══════════════════════════════════════════════════════════════
bad = [dict(in_context_len=2), dict(in_context_layout="row"), dict(in_context_content="time_class"),
       dict(state_init="dimsum"), dict(ssc="bc"), dict(adaln_cond="mlp"), dict(adaln_cond=False),
       dict(ssc_z_mlp=True), dict(ssm_norm="bcdt"), dict(feature_rope="1d"), dict(a_tie="pairs"),
       dict(bc_rope="2d_axial"), dict(global_kernel_xe="cosine"), dict(global_len=-1),
       dict(global_start=KW["depth"]), dict(global_start=-1), dict(global_readback="yes"),
       dict(K=2)]
for kw in bad:
    args = dict(KW, global_len=3, global_start=START)
    args.update(kw)
    try:
        JiTVMambaGlobal(**args)
    except (AssertionError, ValueError):
        continue
    raise AssertionError(f"guard missing for {kw}")
JiTVMambaGlobal(**KW, global_len=3, global_start=START, adaln_cond=True)       # bool form of "full"
print(f"J  guards: {len(bad)} forbidden flags / bad values raise")

# ═══ K. both build_model copies agree on the four configs ═══════════════════
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


TINY_IN = {"full": 31_046_208, "writeonly": 31_043_520, "softmax": 31_046_208,
           "full-m32": 31_052_352}
if yaml is not None:
    bm_train = load_build_model("scripts/run_experiment.py")
    bm_eval = load_build_model("scripts/evaluate.py")
    torch.manual_seed(0)
    x64 = torch.randn(1, 3, 64, 64)
    t64, y64 = torch.rand(1), torch.randint(0, 200, (1,))
    for name, want in TINY_IN.items():
        cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, "configs", "tiny_imagenet",
                                               f"jit-s2-vmamba-global-{name}.yaml"), encoding="utf-8"))
        assert cfg["experiment"]["model"] == "vmamba_global"
        torch.manual_seed(0)
        mt = bm_train(cfg)
        torch.manual_seed(0)
        me = bm_eval(cfg)
        sd_t, sd_e = mt.state_dict(), me.state_dict()
        assert list(sd_t) == list(sd_e) and all(torch.equal(sd_t[k], sd_e[k]) for k in sd_t)
        assert n_params(mt) == want, (name, n_params(mt))
        perturb_zero_init(mt)
        me.load_state_dict(mt.state_dict(), strict=True)
        with torch.no_grad():
            assert maxdiff(mt(x64, t64, y64), me(x64, t64, y64)) == 0
        bad_cfg = copy.deepcopy(cfg)
        bad_cfg["model"]["ssm_norm"] = "bcdt"
        for bm in (bm_train, bm_eval):
            try:
                bm(bad_cfg)
            except AssertionError:
                continue
            raise AssertionError("build_model accepted ssm_norm under vmamba_global")
        print(f"K  jit-s2-vmamba-global-{name}.yaml: train == eval build_model, "
              f"{n_params(mt):,} params; ssm_norm rejected by both")

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
print("L  every block is the identity at init (adaLN-Zero), 3 modes")

# ═══ M. ffn / norm variants are safe with the tokens ════════════════════════
for extra in (dict(ffn="convglu"), dict(ffn="glumbconv"), dict(norm="derf")):
    gm = perturb(global_model(M=3, **extra), 0.05)
    out = run(gm)
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in gm.parameters())
    if "ffn" in extra:
        mlp = gm.blocks[START].mlp
        h = torch.randn(2, 3 + GRID * GRID, KW["hidden_size"])
        with torch.no_grad():
            full_rows = mlp(h, GRID, GRID)[:, :3]
            if extra["ffn"] == "convglu":
                g, v = mlp.fc1(h[:, :3]).chunk(2, dim=-1)
                gc = g * mlp.dwconv.weight[:, 0, 1, 1] + mlp.dwconv.bias
                ref = mlp.fc2(F.silu(gc) * v)
            else:
                s = F.silu(mlp.inverted_conv(h[:, :3]))
                s = s * mlp.depth_conv.weight[:, 0, 1, 1] + mlp.depth_conv.bias
                a, g = s.chunk(2, dim=-1)
                ref = mlp.point_conv(a * F.silu(g))
        assert maxdiff(full_rows, ref) < 1e-5, extra
print("M  ffn convglu / glumbconv and norm derf: fwd+bwd finite; conv FFN token rows = centre tap")

# ═══ N. no dead parameters (what DDP checks) ════════════════════════════════
cases = [("full", 3), ("writeonly", 3), ("softmax", 3), ("full", 6)]
for mode, M in cases:
    gm = global_model(M=M, mode=mode, seed=5)
    run(gm).mul(torch.randn(Bsz, 3, KW["input_size"], KW["input_size"])).sum().backward()
    dead = [n for n, p in gm.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, (mode, M, dead)
    gm = perturb_zero_init(global_model(M=M, mode=mode, seed=5))
    run(gm).mul(torch.randn(Bsz, 3, KW["input_size"], KW["input_size"])).sum().backward()
    zero = [n for n, p in gm.named_parameters() if p.requires_grad and p.grad.abs().sum() == 0]
    assert not zero, (mode, M, zero)
print("N  no dead parameters: grad is not None at init, non-zero after perturbing the "
      "zero-init weights (full / write-only / softmax at M=3, full at M=6)")

print("\nALL vmamba_global CPU checks passed.")
