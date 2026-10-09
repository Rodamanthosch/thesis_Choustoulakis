"""
tests/test_vmamba2_cpu.py
=========================
CPU-only verification of JiT-S2-VMamba2 (src/models/vmamba2.py): SS2D with a
Mamba-2 (SSD) core, VMamba's SS2Dm0 ("m0_noz") with the Mamba-2 module's init.

Runs WITHOUT a GPU and WITHOUT mamba_ssm: both kernels are stubbed by naive
recurrences that stay in float64 when fed float64, so the fidelity checks are
exact. scripts/check_vmamba2_cuda.py proves the real Triton kernel equals the
SSD stub on the GPU, which closes the loop.

Checks:
  A. the SSD stub (the recurrence h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T,
     y_t = h_t C_t + D x_t, grouped B/C, per-channel D, softplus(dt + bias))
     equals an independent quadratic/dual form Y = (L ⊙ C B^T)(dt X) + D X
     (float64).
  B. init: 24 heads x headdim 16 per direction at D=384 (VMamba m0); A in
     [1, 16] and softplus(dt_bias) in [1e-3, 1e-1] (Mamba-2 module init);
     D ones, per channel.
  C. reduction: SS2D-m2 IS the Mamba-1 SS2D (src/models/vmamba.py) with every
     channel's Delta tied to its head's and A tied per head across channels
     and states — the two mixers, run through two independent reference
     kernels, agree to float64 precision; untying one channel breaks it.
  D. params: per block and on the Tiny-IN model, vs the Mamba-1 baseline.
  E. guards: d_inner not divisible by nheads; L != H*W; unknown ffn.
  F. smoke: forward + backward for each ffn, grads reach x_proj / A / dt_bias /
     D; force_fp32 runs; CPU bf16 autocast runs.
  G. drift guard: with each block's mixer swapped for the Mamba-1 SS2D, the
     JiTVMamba2 scaffold loads JiTVMamba's state_dict strictly and is
     BIT-IDENTICAL to it -- the duplicated block/model code cannot silently
     diverge from JiTVMamba.
  H. wiring: the new config builds IDENTICAL models through
     scripts/run_experiment.py::build_model and scripts/evaluate.py::build_model.

Run from the repo root:
    PYTHONPATH=. python tests/test_vmamba2_cpu.py
"""
import ast
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


def _wd(t):
    """Working dtype: float64 stays float64, everything else computes in fp32."""
    return torch.promote_types(t.dtype, torch.float32)


# ── Mamba-1 stub: selective_scan_fn (dtype-preserving) ───────────────────────
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u: (b, d, l)  delta: (b, d, l)  A: (d, n)  B, C: (b, g, n, l) grouped."""
    wd = _wd(u)
    b, d, l = u.shape
    u, delta, A = u.to(wd), delta.to(wd), A.to(wd)
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(wd)
    if delta_softplus:
        delta = F.softplus(delta)
    rep = d // B.shape[1]
    Bf = B.to(wd).repeat_interleave(rep, dim=1)      # (b, d, n, l)
    Cf = C.to(wd).repeat_interleave(rep, dim=1)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for i in range(l):
        h = torch.exp(delta[:, :, i, None] * A[None]) * h \
            + delta[:, :, i, None] * Bf[:, :, :, i] * u[:, :, i, None]
        ys.append((h * Cf[:, :, :, i]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D[None, :, None].to(wd) * u
    return y.to(u.dtype)


# ── Mamba-2 stub: mamba_chunk_scan_combined as a naive recurrence ────────────
def ssd_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
            initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False,
            dt_limit=(0.0, float("inf")), return_final_states=False,
            return_varlen_states=False):
    """x: (b, l, h, p)  dt: (b, l, h)  A: (h,)  B, C: (b, l, g, n)  D: (h, p) | (h,)."""
    assert initial_states is None and not return_final_states
    wd = _wd(x)
    b, l, h, p = x.shape
    xw, dtw, Aw = x.to(wd), dt.to(wd), A.to(wd)
    if dt_bias is not None:
        dtw = dtw + dt_bias.to(wd)
    if dt_softplus:
        dtw = F.softplus(dtw)
    if dt_limit != (0.0, float("inf")):
        dtw = dtw.clamp(*dt_limit)
    rep = h // B.shape[2]
    Bh = B.to(wd).repeat_interleave(rep, dim=2)      # (b, l, h, n)
    Ch = C.to(wd).repeat_interleave(rep, dim=2)
    st = xw.new_zeros(b, h, p, B.shape[-1])
    ys = []
    for t in range(l):
        st = torch.exp(dtw[:, t] * Aw)[:, :, None, None] * st \
            + (dtw[:, t, :, None] * xw[:, t])[..., None] * Bh[:, t, :, None, :]
        ys.append((st * Ch[:, t, :, None, :]).sum(-1))
    y = torch.stack(ys, dim=1)                       # (b, l, h, p)
    if D is not None:
        Dw = D.to(wd)
        y = y + xw * (Dw[None, None] if Dw.dim() == 2 else Dw[None, None, :, None])
    if z is not None:
        y = y * F.silu(z.to(wd))
    return y.to(x.dtype)


if "mamba_ssm" not in sys.modules:
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
        sys.modules[name] = types.ModuleType(name)
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = selective_scan_ref
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = iface
    ssd_mod = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
    ssd_mod.mamba_chunk_scan_combined = ssd_ref
    sys.modules["mamba_ssm.ops.triton.ssd_combined"] = ssd_mod

from src.models.vmamba import JiTVMamba, SS2D                # noqa: E402  (after the stubs)
from src.models.vmamba2 import JiTVMamba2, SS2DMamba2       # noqa: E402

torch.manual_seed(0)
KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32, d_state=16, K=4, expand=1)
H = W = KW["input_size"] // KW["patch_size"]       # 4
HW = H * W
Bsz = 3

x_img = torch.randn(Bsz, 3, KW["input_size"], KW["input_size"])
t_in = torch.rand(Bsz)
y_in = torch.randint(0, KW["num_classes"], (Bsz,))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def perturb(module, scale=0.05):
    """Move every parameter off its (often degenerate) init, deterministically."""
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))
    return module


# ═══ A. SSD stub == independent dual form ════════════════════════════════════
def ssd_dual(x, dt, A, B, C, D, dt_bias):
    """Y = (L ⊙ C B^T)(dt X) + D X per head, L[t,s] = exp(sum_{r=s+1..t} dt_r A)."""
    b, l, h, p = x.shape
    dt = F.softplus(dt + dt_bias)                    # (b, l, h)
    rep = h // B.shape[2]
    Bh = B.repeat_interleave(rep, dim=2)
    Ch = C.repeat_interleave(rep, dim=2)
    cs = torch.cumsum(dt * A, dim=1)                 # (b, l, h)
    seg = cs[:, :, None, :] - cs[:, None, :, :]      # (b, t, s, h)
    causal = torch.tril(torch.ones(l, l, dtype=torch.bool))[None, :, :, None]
    Lm = torch.where(causal, torch.exp(seg.clamp(max=0)), torch.zeros_like(seg))
    scores = torch.einsum("bthn,bshn->btsh", Ch, Bh) * Lm
    y = torch.einsum("btsh,bshp->bthp", scores, dt[..., None] * x)
    return y + x * D[None, None]


torch.manual_seed(1)
b_, l_, h_, p_, g_, n_ = 2, 13, 6, 5, 3, 7
xa = torch.randn(b_, l_, h_, p_, dtype=torch.float64)
dta = torch.randn(b_, l_, h_, dtype=torch.float64)
Aa = -torch.rand(h_, dtype=torch.float64) * 3
Ba = torch.randn(b_, l_, g_, n_, dtype=torch.float64)
Ca = torch.randn(b_, l_, g_, n_, dtype=torch.float64)
Da = torch.randn(h_, p_, dtype=torch.float64)
bias_a = torch.randn(h_, dtype=torch.float64)
d_a = (ssd_ref(xa, dta, Aa, Ba, Ca, 64, D=Da, dt_bias=bias_a, dt_softplus=True)
       - ssd_dual(xa, dta, Aa, Ba, Ca, Da, bias_a)).abs().max().item()
assert d_a < 1e-12, f"SSD stub != dual form: {d_a}"
print(f"A  SSD stub (recurrence) == dual form (L ⊙ CB^T)(dt X) + D X, grouped B/C, "
      f"per-channel D (float64 max|diff| = {d_a:.1e})")


# ═══ B. heads and init ═══════════════════════════════════════════════════════
torch.manual_seed(0)
mix = SS2DMamba2(384, d_state=16, K=4)
assert (mix.nheads, mix.headdim) == (24, 16), (mix.nheads, mix.headdim)
A_vals = mix.A_logs.exp()
dt_vals = F.softplus(mix.dt_bias)
assert A_vals.min() >= 1 and A_vals.max() <= 16
assert dt_vals.min() >= 1e-3 - 1e-9 and dt_vals.max() <= 1e-1 + 1e-9
assert torch.equal(mix.Ds, torch.ones(4, 384))
print(f"B  D=384: {mix.nheads} heads x headdim {mix.headdim} per direction; "
      f"A in [{A_vals.min():.2f}, {A_vals.max():.2f}], softplus(dt_bias) in "
      f"[{dt_vals.min():.4f}, {dt_vals.max():.4f}], D ones (K, d_inner)")


# ═══ C. reduction: SS2D-m2 == Mamba-1 SS2D with Delta, A tied per head ═══════
D_MIX, N = 64, 16
torch.manual_seed(3)
m2 = perturb(SS2DMamba2(D_MIX, d_state=N, K=4), 0.3).double().eval()
R, P = m2.nheads, m2.headdim
m1 = SS2D(D_MIX, d_state=N, K=4).double().eval()
assert m1.dt_rank == R, (m1.dt_rank, R)        # VMamba m0 reuses dt_rank as the head count
with torch.no_grad():
    for name in ("in_proj", "conv2d", "out_norm", "out_proj"):
        getattr(m1, name).load_state_dict(getattr(m2, name).state_dict())
    m1.x_proj_weight.copy_(m2.x_proj_weight)
    sel = torch.zeros(4, D_MIX, R, dtype=torch.float64)
    sel[:, torch.arange(D_MIX), torch.arange(D_MIX) // P] = 1.0   # channel d reads head d // P
    m1.dt_projs_weight.copy_(sel)
    m1.dt_projs_bias.copy_(m2.dt_bias.repeat_interleave(P, dim=1))
    m1.A_logs.copy_(m2.A_logs.repeat_interleave(P, dim=1)[..., None].expand(4, D_MIX, N))
    m1.Ds.copy_(m2.Ds)
xc = torch.randn(2, HW, D_MIX, dtype=torch.float64)
with torch.no_grad():
    d_red = (m1(xc, H, W) - m2(xc, H, W)).abs().max().item()
    m1.dt_projs_weight[1, 5, (5 // P + 1) % R] += 0.5                # untie one channel
    d_neg = (m1(xc, H, W) - m2(xc, H, W)).abs().max().item()
assert d_red < 1e-12, f"SS2D-m2 != tied Mamba-1 SS2D: {d_red}"
assert d_neg > 1e-6, f"negative control did not fire: {d_neg}"
print(f"C  SS2D-m2 == Mamba-1 SS2D with Delta tied per head and A tied per head "
      f"(float64 max|diff| = {d_red:.1e}); untying one channel's Delta -> {d_neg:.1e}")


# ═══ D. params ═══════════════════════════════════════════════════════════════
p1, p2 = n_params(SS2D(384, d_state=16, K=4)), n_params(SS2DMamba2(384, d_state=16, K=4))
TINY = dict(input_size=64, patch_size=8, hidden_size=384, depth=12,
            num_classes=200, bottleneck_dim=128)
torch.manual_seed(0)
t1 = n_params(JiTVMamba(**TINY))
torch.manual_seed(0)
t2 = n_params(JiTVMamba2(**TINY))
assert t2 - t1 == 12 * (p2 - p1), (t1, t2, p1, p2)
print(f"D  mixer per block (D=384, N=16): Mamba-1 {p1:,} | Mamba-2 {p2:,} ({p2 - p1:+,}); "
      f"Tiny-IN model {t1:,} -> {t2:,} ({t2 - t1:+,})")


# ═══ E. guards ═══════════════════════════════════════════════════════════════
for bad, call in (
    ("nheads not dividing d_inner", lambda: SS2DMamba2(64, nheads=5)),
    ("L != H*W", lambda: SS2DMamba2(64)(torch.randn(1, HW + 1, 64), H, W)),
    ("unknown ffn", lambda: JiTVMamba2(ffn="mixffn", **KW)),
):
    try:
        call()
        raise SystemExit(f"guard missing: {bad}")
    except AssertionError:
        pass
print("E  nheads not dividing d_inner, L != H*W and an unknown ffn are rejected")


# ═══ F. smoke ════════════════════════════════════════════════════════════════
for ffn in ("swiglu", "convglu", "glumbconv"):
    torch.manual_seed(0)
    m = perturb(JiTVMamba2(ffn=ffn, **KW), 0.02).train()
    out = m(x_img, t_in, y_in)
    assert out.shape == x_img.shape and torch.isfinite(out).all(), ffn
    out.square().mean().backward()
    mx = m.blocks[-1].mixer
    for name in ("x_proj_weight", "A_logs", "dt_bias", "Ds"):
        g = getattr(mx, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0, (ffn, name)
torch.manual_seed(0)
m = JiTVMamba2(force_fp32=True, **KW).eval()
with torch.no_grad():
    assert torch.isfinite(m(x_img, t_in, y_in)).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(JiTVMamba2(**KW)(x_img, t_in, y_in).float()).all()
print("F  forward+backward OK for swiglu/convglu/glumbconv (grads reach x_proj, "
      "A, dt_bias, D); force_fp32 and CPU bf16 autocast run")


# ═══ G. drift guard: scaffold == JiTVMamba's ═════════════════════════════════
dj = 0.0
for ffn in ("swiglu", "convglu", "glumbconv"):
    torch.manual_seed(7)
    vm = perturb(JiTVMamba(ffn=ffn, **KW), 0.02).eval()
    v2 = JiTVMamba2(ffn=ffn, **KW)
    for blk in v2.blocks:
        blk.mixer = SS2D(d_model=KW["hidden_size"], d_state=16, d_conv=3, expand=1, K=4)
    v2.load_state_dict(vm.state_dict(), strict=True)
    with torch.no_grad():
        dj = max(dj, (v2.eval()(x_img, t_in, y_in) - vm(x_img, t_in, y_in)).abs().max().item())
assert dj == 0.0, f"JiTVMamba2 scaffold drifted from JiTVMamba: {dj}"
print("G  scaffold with SS2D swapped in loads JiTVMamba strictly and is "
      "bit-identical to it (swiglu, convglu, glumbconv)")


# ═══ H. both build_model copies agree on the new config ══════════════════════
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
cfg_path = os.path.join(REPO_ROOT, "configs", "tiny_imagenet", "jit-s2-vmamba2.yaml")
cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
assert cfg["experiment"]["model"] == "vmamba2"
torch.manual_seed(0)
mt = bm_train(cfg)
torch.manual_seed(0)
me = bm_eval(cfg)
assert isinstance(mt, JiTVMamba2) and isinstance(me, JiTVMamba2)
st, se = mt.state_dict(), me.state_dict()
assert set(st) == set(se) and all(torch.equal(st[k], se[k]) for k in st)
assert n_params(mt) == t2
print(f"H  {os.path.basename(cfg_path)}: train == eval build_model, {n_params(mt):,} params")

print("ALL CHECKS PASSED")
