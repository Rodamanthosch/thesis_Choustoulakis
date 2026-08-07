"""
tests/test_ssc_static_cpu.py
============================
CPU-only verification of the ssc="static" bias-only control arm.

Runs WITHOUT a GPU and WITHOUT mamba_ssm: the CUDA kernel is stubbed with a
faithful reference selective scan (grouped B/C, delta_bias/softplus semantics),
so the checks below are pure-Python and exact.

Checks:
  A. all four ssc modes construct + forward + backward
  B. static == bc at init, EXACTLY (bc's condition projections are zero-init)
     -- proven by weight transplant, since independent seeding cannot align
     params (bc's extra Linears shift the init RNG stream)
  C. static != none at init (the ones-init biases are active, as intended)
  D. abc != bc at init (gate s ~= 0.982: near- but not exactly baseline decay)
  E. param accounting: +2*K*N per block for static; +2*D*K*N for bc's
     projections; +(K*D + K) for abc's gate
  F. state_dict compat: none's keys are a strict subset of static's
  G. arm exclusivity asserts still guard static

Run from the repo root:
    python tests/test_ssc_static_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
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


if "mamba_ssm" not in sys.modules:
    stub = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = selective_scan_ref
    iface.selective_scan_ref = selective_scan_ref
    sys.modules["mamba_ssm"] = stub
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = iface

from src.models.vmamba import JiTVMamba  # noqa: E402  (after the stub)

torch.manual_seed(0)
KW = dict(input_size=16, patch_size=8, hidden_size=64, depth=2, num_classes=7,
          bottleneck_dim=32, d_state=16, K=4, expand=1)
K, N, D, DEPTH = 4, 16, 64, 2


def build(ssc):
    torch.manual_seed(0)
    return JiTVMamba(ssc=ssc, **KW).eval()


models = {s: build(s) for s in ["none", "static", "bc", "abc"]}

# ── A. construct + forward + backward for every mode ─────────────────────────
x = torch.randn(3, 3, 16, 16)
t = torch.rand(3)
y = torch.randint(0, 7, (3,))
for s, m in models.items():
    out = m(x, t, y)
    assert out.shape == x.shape, (s, out.shape)
    out.mean().backward()
    m.zero_grad(set_to_none=True)

# ── B/C/D. relations at init, proven by weight transplant from bc ────────────
# (adaLN-Zero + zero-init FinalLayer make ALL variants output 0 at the model
# level at init, so the inequality checks probe the SS2D mixer directly.)
bc = models["bc"]
sd_bc = bc.state_dict()

sd_static = {k: v for k, v in sd_bc.items()
             if "cond_B_proj" not in k and "cond_C_proj" not in k}
models["static"].load_state_dict(sd_static, strict=True)

models["abc"].load_state_dict(sd_bc, strict=False)   # abc keeps its gate init

sd_none = {k: v for k, v in sd_bc.items()
           if not any(tag in k for tag in
                      ("cond_B_proj", "cond_C_proj", "bias_B", "bias_C"))}
models["none"].load_state_dict(sd_none, strict=True)


def mixer_out(m):
    torch.manual_seed(1)
    z = torch.randn(3, 2 * 2, 64)
    c = torch.randn(3, 64)
    return m.blocks[0].mixer(z, 2, 2, cond=c).detach()


outs = {s: mixer_out(m) for s, m in models.items()}
d_sb = (outs["static"] - outs["bc"]).abs().max().item()
d_sn = (outs["static"] - outs["none"]).abs().max().item()
d_ab = (outs["abc"] - outs["bc"]).abs().max().item()
assert d_sb == 0.0, f"static must equal bc at init, got {d_sb}"
assert d_sn > 1e-6, "static must differ from none (ones-init biases)"
assert d_ab > 1e-8, "abc must differ slightly from bc at init (gate ~0.982)"

# ── E. param accounting ──────────────────────────────────────────────────────
p = {s: sum(q.numel() for q in m.parameters()) for s, m in models.items()}
assert p["static"] - p["none"] == 2 * K * N * DEPTH, p
assert p["bc"] - p["static"] == 2 * D * K * N * DEPTH, p
assert p["abc"] - p["bc"] == (K * D + K) * DEPTH, p

# ── F. state_dict compat ─────────────────────────────────────────────────────
k_none = set(models["none"].state_dict())
k_static = set(models["static"].state_dict())
assert k_none < k_static and k_static - k_none == {
    f"blocks.{i}.mixer.bias_{c}" for i in range(DEPTH) for c in "BC"}

# ── G. exclusivity asserts still guard static ────────────────────────────────
for bad_kw in (dict(state_init="dimsum"), dict(in_context_len=2)):
    try:
        JiTVMamba(ssc="static", **bad_kw, **KW)
        raise SystemExit(f"exclusivity assert missing for {bad_kw}!")
    except AssertionError:
        pass

print("A  forward+backward OK for none/static/bc/abc")
print(f"B  static == bc at init:        max|diff| = {d_sb}")
print(f"C  static != none:              max|diff| = {d_sn:.3e}")
print(f"D  abc != bc at init (gate):    max|diff| = {d_ab:.3e}")
print(f"E  params: none={p['none']}, +static={p['static'] - p['none']}, "
      f"+cond_proj={p['bc'] - p['static']}, +gate={p['abc'] - p['bc']}")
print("F  state_dict: none \u2282 static (extra keys: bias_B/bias_C only)")
print("G  arm exclusivity asserts hold")
print("ALL CHECKS PASSED")
