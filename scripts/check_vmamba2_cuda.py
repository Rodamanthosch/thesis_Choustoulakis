"""
scripts/check_vmamba2_cuda.py
=============================
GPU check AND hardware probe for JiT-S2-VMamba2 (model: vmamba2). Run it once
at the start of the first session on a new GPU type (Kaggle T4, H100, ...).

The Mamba-2 core is mamba_ssm's Triton kernel mamba_chunk_scan_combined. Triton
officially targets Ampere+ GPUs, and a T4 (sm 7.5) has no bf16 tensor cores, so
whether the kernel runs there -- and in which dtype -- is an empirical question.
Every probe below is wrapped so the script REPORTS what works instead of
crashing on the first failure.

  0. Environment: GPU, compute capability, torch / triton / mamba_ssm versions.
  1. Kernel vs reference on the Tiny-IN shapes (b 8, L 64, 96 heads x 16,
     4 groups, N 16, per-channel D): forward and backward (dx, ddt, dA, dB, dC,
     dD, ddt_bias) against a float64 dual-form reference, with fp32 inputs and
     with bf16 inputs.
  2. Mixer integration: SS2DMamba2 with the real kernel vs the same mixer in
     float64 with the reference swapped in.
  3. End-to-end: the config through evaluate.build_model, fwd + bwd under bf16
     autocast, with force_fp32 false and true.
  4. Training-step throughput (bf16 autocast, batch 128) vs the Mamba-1
     baseline, for every setting that worked.

Ends with a one-line recommendation for `force_fp32` on this GPU.

Run (on GPU, from the repo root):  python scripts/check_vmamba2_cuda.py
"""
import copy
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml

import src.models.vmamba2 as vm2
from src.models.vmamba2 import SS2DMamba2
from scripts.evaluate import build_model

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
CONFIG = "configs/tiny_imagenet/jit-s2-vmamba2.yaml"
BASELINE = "configs/tiny_imagenet/jit-s2-vmamba-baseline.yaml"
works = {}            # setting -> bool (ran AND matched the reference)


def first_line(e):
    return (str(e).strip().splitlines() or [repr(e)])[0][:160]


# ── 0. environment ───────────────────────────────────────────────────────────
import mamba_ssm                                               # noqa: E402
try:
    import triton                                              # noqa: E402
    triton_ver = triton.__version__
except Exception as e:                                         # pragma: no cover
    triton_ver = f"not importable ({first_line(e)})"
cap = torch.cuda.get_device_capability()
print(f"0. GPU {torch.cuda.get_device_name()}  sm {cap[0]}.{cap[1]}  "
      f"bf16 supported: {torch.cuda.is_bf16_supported()}  | torch {torch.__version__}  "
      f"triton {triton_ver}  mamba_ssm {mamba_ssm.__version__}")


# ── reference: float64 dual form of the SSD (independent of the kernel) ──────
def ssd_dual(x, dt, A, B, C, D, dt_bias):
    """Y = (L ⊙ C B^T)(dt X) + D X per head, L[t,s] = exp(sum_{r=s+1..t} dt_r A)."""
    x, dt, A, B, C, D, dt_bias = (t.double() for t in (x, dt, A, B, C, D, dt_bias))
    b, l, h, p = x.shape
    dt = F.softplus(dt + dt_bias)
    rep = h // B.shape[2]
    Bh, Ch = B.repeat_interleave(rep, dim=2), C.repeat_interleave(rep, dim=2)
    cs = torch.cumsum(dt * A, dim=1)
    seg = cs[:, :, None, :] - cs[:, None, :, :]
    causal = torch.tril(torch.ones(l, l, dtype=torch.bool, device=x.device))[None, :, :, None]
    Lm = torch.where(causal, torch.exp(seg.clamp(max=0)), torch.zeros_like(seg))
    scores = torch.einsum("bthn,bshn->btsh", Ch, Bh) * Lm
    return torch.einsum("btsh,bshp->bthp", scores, dt[..., None] * x) + x * D[None, None]


def ssd_ref_call(x, dt, A, B, C, chunk_size, D=None, dt_bias=None, dt_softplus=False, **kw):
    """Drop-in for mamba_chunk_scan_combined (the subset SS2DMamba2 uses)."""
    assert dt_softplus and D is not None and dt_bias is not None
    return ssd_dual(x, dt, A, B, C, D, dt_bias).to(x.dtype)


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


# ── 1. the kernel itself ─────────────────────────────────────────────────────
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined   # noqa: E402

torch.manual_seed(42)
b, L, h, p, g, n = 8, 64, 96, 16, 4, 16
base = dict(
    x=torch.randn(b, L, h, p, device=dev),
    dt=torch.randn(b, L, h, device=dev) - 2.0,
    A=-torch.empty(h, device=dev).uniform_(1, 16),
    B=torch.randn(b, L, g, n, device=dev),
    C=torch.randn(b, L, g, n, device=dev),
    D=torch.randn(h, p, device=dev),
    dt_bias=torch.randn(h, device=dev) * 0.5,
)
gy = torch.randn(b, L, h, p, device=dev)
names = ("x", "dt", "A", "B", "C", "D", "dt_bias")

leaf = {k: v.double().requires_grad_(True) for k, v in base.items()}
y_ref = ssd_dual(*(leaf[k] for k in names))
g_ref = dict(zip(names, torch.autograd.grad(y_ref, [leaf[k] for k in names], gy.double())))

for tag, dtype, tol in (("fp32", torch.float32, 2e-2), ("bf16", torch.bfloat16, 6e-2)):
    try:
        t = {k: (v.to(dtype) if k in ("x", "dt", "B", "C") else v).clone().requires_grad_(True)
             for k, v in base.items()}
        y = mamba_chunk_scan_combined(t["x"], t["dt"], t["A"], t["B"], t["C"], 64,
                                      D=t["D"], dt_bias=t["dt_bias"], dt_softplus=True)
        grads = torch.autograd.grad(y, [t[k] for k in names], gy.to(y.dtype))
        errs = {"y": rel(y, y_ref)}
        errs.update({f"d{k}": rel(gk, g_ref[k]) for k, gk in zip(names, grads)})
        good = all(e < tol for e in errs.values())
        works[f"kernel {tag}"] = good
        worst = max(errs, key=errs.get)
        print(f"1. kernel {tag:<4} ran: fwd rel err {errs['y']:.1e}, worst grad "
              f"{worst} {errs[worst]:.1e} (tol {tol:g})  {'OK' if good else 'FAIL'}")
    except Exception as e:
        works[f"kernel {tag}"] = False
        print(f"1. kernel {tag:<4} DOES NOT RUN here: {type(e).__name__}: {first_line(e)}")


# ── 2. mixer integration ─────────────────────────────────────────────────────
torch.manual_seed(0)
mix = SS2DMamba2(384, d_state=16, K=4).to(dev).eval()
with torch.no_grad():
    for prm in mix.parameters():
        prm.add_(0.05 * torch.randn_like(prm))
mix64 = copy.deepcopy(mix).double()
xm = torch.randn(4, 64, 384, device=dev)
_kernel = vm2.mamba_chunk_scan_combined
vm2.mamba_chunk_scan_combined = ssd_ref_call
try:
    with torch.no_grad():
        y64 = mix64(xm.double(), 8, 8)
finally:
    vm2.mamba_chunk_scan_combined = _kernel
for tag, force, autocast in (("fp32", False, False), ("bf16 autocast", False, True),
                             ("bf16 autocast + force_fp32", True, True)):
    try:
        mix.force_fp32 = force
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            ym = mix(xm, 8, 8)
        e = rel(ym, y64)
        good = torch.isfinite(ym).all().item() and e < (6e-2 if autocast else 2e-2)
        works[f"mixer {tag}"] = good
        print(f"2. SS2DMamba2 {tag:<27} vs float64 reference: rel err {e:.1e}  "
              f"{'OK' if good else 'FAIL'}")
    except Exception as e:
        works[f"mixer {tag}"] = False
        print(f"2. SS2DMamba2 {tag:<27} DOES NOT RUN here: {type(e).__name__}: {first_line(e)}")
mix.force_fp32 = False


# ── 3. end-to-end through build_model ────────────────────────────────────────
cfg0 = yaml.safe_load(open(CONFIG))
for force in (False, True):
    tag = f"force_fp32={str(force).lower()}"
    try:
        cfg = copy.deepcopy(cfg0)
        cfg["model"]["force_fp32"] = force
        mc = cfg["model"]
        torch.manual_seed(0)
        m = build_model(cfg).to(dev).train()
        with torch.no_grad():                   # off adaLN-Zero's identity init so the
            for prm in m.parameters():           # mixer actually receives gradient
                prm.add_(0.02 * torch.randn_like(prm))
        x = torch.randn(4, 3, mc["input_size"], mc["input_size"], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, torch.rand(4, device=dev),
                    torch.randint(0, mc["num_classes"], (4,), device=dev))
        out.float().square().mean().backward()
        mx = m.blocks[-1].mixer
        good = (out.shape == x.shape and torch.isfinite(out).all().item()
                and all(getattr(mx, k).grad is not None
                        and torch.isfinite(getattr(mx, k).grad).all().item()
                        for k in ("x_proj_weight", "A_logs", "dt_bias", "Ds")))
        works[f"model {tag}"] = good
        n_p = sum(q.numel() for q in m.parameters())
        print(f"3. {os.path.basename(CONFIG)} ({tag}): fwd+bwd bf16 autocast, "
              f"{n_p:,} params  {'OK' if good else 'FAIL'}")
        del m, out
    except Exception as e:
        works[f"model {tag}"] = False
        print(f"3. {os.path.basename(CONFIG)} ({tag}) DOES NOT RUN here: "
              f"{type(e).__name__}: {first_line(e)}")
        traceback.print_exc(limit=2)
    torch.cuda.empty_cache()


# ── 4. throughput vs the Mamba-1 baseline ────────────────────────────────────
def step_rate(cfg, bs=128, warmup=5, iters=20):
    mc = cfg["model"]
    torch.manual_seed(0)
    m = build_model(cfg).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randn(bs, 3, mc["input_size"], mc["input_size"], device=dev)
    t = torch.rand(bs, device=dev)
    y = torch.randint(0, mc["num_classes"], (bs,), device=dev)
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, t, y).float().square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    rate = iters / (time.perf_counter() - t0)
    del m, opt
    torch.cuda.empty_cache()
    return rate


try:
    r_base = step_rate(yaml.safe_load(open(BASELINE)))
    print(f"4. Mamba-1 baseline {r_base:6.2f} it/s (bs 128, bf16, fwd+bwd+AdamW)")
except Exception as e:
    r_base = None
    print(f"4. Mamba-1 baseline did not run: {type(e).__name__}: {first_line(e)}")
for force in (False, True):
    tag = f"force_fp32={str(force).lower()}"
    if not works.get(f"model {tag}"):
        continue
    cfg = copy.deepcopy(cfg0)
    cfg["model"]["force_fp32"] = force
    r = step_rate(cfg)
    ratio = f"  ({r / r_base:.3f}x baseline)" if r_base else ""
    print(f"4. vmamba2 {tag:<17} {r:6.2f} it/s{ratio}")


# ── verdict ──────────────────────────────────────────────────────────────────
if works.get("model force_fp32=false") and works.get("mixer bf16 autocast"):
    verdict = "use force_fp32: false (the bf16 Triton kernel works on this GPU)"
elif works.get("model force_fp32=true") and works.get("mixer bf16 autocast + force_fp32"):
    verdict = "use force_fp32: true (bf16 fails here, fp32 kernel inputs work)"
else:
    verdict = "the Triton SSD kernel is NOT usable on this GPU -- needs a PyTorch backend"
print(f"\nVERDICT: {verdict}")
ok = any(works.get(f"model force_fp32={s}") for s in ("false", "true"))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
