"""
scripts/check_spatial_global_cuda.py
====================================
GPU check for model=spatial_global (src/models/spatial_global.py) -- run once
at the start of the first Kaggle session of each global-token run.

The scan kernel is called exactly as in the no-gate baseline (image tokens
only, L = 64 in EVERY block); what is new is the fp32 memories and the anchor
added to the states before SASF.

  1. Exactness: SpatialGlobalSSM on CUDA (fp32) vs a CPU copy of the same
     weights run through mamba_ssm's selective_scan_ref -- full / write-only,
     gate on and off, and the last-block variant (emit_tokens off).
  2. End to end (--config): evaluate.build_model, forward + backward under
     bf16 autocast; the scan length is 64 in every block (the tokens never
     enter the scan); S / G / ve / anchor / token rows are fp32; the slots and
     every alpha_* get finite, non-zero gradients; the parameter count.
  3. Cost: train-step it/s and peak memory at batch 128, bf16, for the no-gate
     baseline, the 3 in-context class tokens run and the two global configs.
  4. fvcore (src/flops_counter.count_complexity): params and MACs of the same
     list, against the expected deltas over the baseline.

Run (on GPU, from the repo root):
    PYTHONPATH=. python scripts/check_spatial_global_cuda.py --config full
    (--config writeonly; --skip-cost to stop after 2)
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import src.models.spatial_mamba as spatial_mod
import src.models.spatial_global as global_mod
from src.models.spatial_global import SpatialGlobalSSM
from scripts.evaluate import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="full", choices=("full", "writeonly"))
ap.add_argument("--skip-cost", action="store_true")
args = ap.parse_args()

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True

CFG_DIR = "configs/tiny_imagenet"
GLOBAL = {n: f"{CFG_DIR}/jit-s2-spatial-nogate-global-{n}.yaml"
          for n in ("full", "writeonly")}
BASE = f"{CFG_DIR}/jit-s2-spatial-nogate.yaml"
INCONTEXT = f"{CFG_DIR}/jit-s2-spatial-nogate-incontext-class3.yaml"
PARAMS = {"full": 29_697_636, "writeonly": 29_694_948}
# MACs over the baseline: the token rows' linears (no scan, no SASF) + the S
# and G einsums (block 11: S only). The token read is elementwise (uncounted).
EST_DMACS = {"full": 95_797_248, "writeonly": 95_625_216}


def set_scan(fn):
    spatial_mod.selective_scan_fn = fn
    global_mod.selective_scan_fn = fn


def runs(lengths):
    """[64, 64, ...] -> '64 x12' for a one-line report."""
    out, prev, n = [], None, 0
    for L in lengths:
        if L == prev:
            n += 1
        else:
            if prev is not None:
                out.append(f"{prev} x{n}")
            prev, n = L, 1
    out.append(f"{prev} x{n}")
    return ", ".join(out)


# ── 1. mixer: CUDA fp32 vs CPU reference ─────────────────────────────────────
def compare(name, module, x):
    torch.manual_seed(0)
    cpu = module.eval()
    with torch.no_grad():
        for p in cpu.parameters():
            p.add_(0.02 * torch.randn_like(p))
    gpu = copy.deepcopy(cpu).to(dev)
    with torch.no_grad():
        y_gpu = gpu(x.to(dev), 8, 8).cpu()
        set_scan(selective_scan_ref)                     # CPU reference path
        try:
            y_cpu = cpu(x, 8, 8)
        finally:
            set_scan(selective_scan_fn)
    e = ((y_gpu - y_cpu).abs().max() / y_cpu.abs().max()).item()
    g = e < 1e-4
    print(f"1. {name:<40} CUDA fp32 vs CPU ref: max rel |dy| = {e:.2e}  {'OK' if g else 'FAIL'}")
    return g


for readback in (True, False):
    for gate in (False, True):
        for emit in (True, False):
            torch.manual_seed(1)
            ok &= compare(f"readback={readback}, gate={gate}, emit={emit}",
                          SpatialGlobalSSM(384, gate=gate, dilations=(1, 2, 3),
                                           readback=readback, emit_tokens=emit),
                          torch.randn(4, 16 + 64, 384))

# ── 2. end to end under bf16 (--config) ──────────────────────────────────────
cfg = yaml.safe_load(open(GLOBAL[args.config]))
mc = cfg["model"]
res, n_cls = mc["input_size"], mc["num_classes"]
hw = (res // mc["patch_size"]) ** 2
SEEN, DTYPES = [], set()


def recording(u, *a, **kw):
    SEEN.append(u.shape[-1])
    return selective_scan_fn(u, *a, **kw)


terms_orig = SpatialGlobalSSM.terms


def terms_rec(self, x, H, W):
    out = terms_orig(self, x, H, W)
    for key in ("ve", "S", "G", "anchor", "y_e"):
        if out[key] is not None:
            DTYPES.add((key, out[key].dtype))
    return out


torch.manual_seed(0)
m = build_model(cfg).to(dev).train()
with torch.no_grad():                        # leave the zero (identity) init so
    for p in m.parameters():                 # every branch receives gradient
        p.add_(0.02 * torch.randn_like(p))
n_par = sum(p.numel() for p in m.parameters())
xb = torch.randn(2, 3, res, res, device=dev)
set_scan(recording)
SpatialGlobalSSM.terms = terms_rec
try:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(xb, torch.rand(2, device=dev), torch.randint(0, n_cls, (2,), device=dev))
    out.float().pow(2).mean().backward()
finally:
    set_scan(selective_scan_fn)
    SpatialGlobalSSM.terms = terms_orig
grads = {"incontext_pos_embed": m.incontext_pos_embed.grad}
grads.update({n: p.grad for n, p in m.named_parameters() if ".alpha_" in n})
grads_ok = all(g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0
               for g in grads.values())
fp32 = len(DTYPES) > 0 and all(dt == torch.float32 for _, dt in DTYPES)
good = (out.shape == xb.shape and torch.isfinite(out.float()).all().item()
        and SEEN == [hw] * mc["depth"] and fp32 and grads_ok and n_par == PARAMS[args.config])
ok &= good
print(f"2. {args.config} bf16 fwd+bwd: out {tuple(out.shape)}  scan L per block: {runs(SEEN)} "
      f"(want {hw} x{mc['depth']})  memories {sorted({str(d) for _, d in DTYPES})}  "
      f"{len(grads)} slot/gain grads finite & non-zero: {grads_ok}  params {n_par:,}  "
      f"{'OK' if good else 'FAIL'}")
del m, out
torch.cuda.empty_cache()

if args.skip_cost:
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# ── 3. training-step speed and peak memory ───────────────────────────────────
def step_rate(cfg, bs=128, warmup=5, iters=20):
    mc = cfg["model"]
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    m = build_model(cfg).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    xi = torch.randn(bs, 3, mc["input_size"], mc["input_size"], device=dev)
    t = torch.rand(bs, device=dev)
    y = torch.randint(0, mc["num_classes"], (bs,), device=dev)
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(xi, t, y).float().square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    rate = iters / (time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del m, opt
    torch.cuda.empty_cache()
    return rate, peak


RUNS = [("spatial no gate (reference)", BASE), ("+ 3 in-context class tokens", INCONTEXT)]
RUNS += [(f"global {n}", p) for n, p in GLOBAL.items()]
r_ref = None
for name, path in RUNS:
    try:
        r, peak = step_rate(yaml.safe_load(open(path)))
    except torch.cuda.OutOfMemoryError:
        ok = False
        print(f"3. {name:<30} OUT OF MEMORY at batch 128  FAIL")
        torch.cuda.empty_cache()
        continue
    r_ref = r_ref or r
    print(f"3. {name:<30} {r:6.2f} it/s ({r / r_ref:.3f}x reference)  peak {peak:5.2f} GiB")

# ── 4. fvcore params / MACs ──────────────────────────────────────────────────
try:
    from src.flops_counter import count_complexity
except ImportError as e:
    ok = False
    print(f"4. MACs  FAIL (fvcore missing: pip install fvcore) ({e})")
else:
    macs = {}
    print("4. fvcore (MACs, the repo's 'GFLOPs'):")
    for name, path in RUNS:
        c = yaml.safe_load(open(path))
        torch.manual_seed(0)
        r = count_complexity(build_model(c), img_size=c["model"]["input_size"],
                             in_channels=c["model"]["in_channels"],
                             num_classes=c["model"]["num_classes"], device=dev)
        macs[name] = r["macs_total"]
        line = f"   {name:<30} {r['params_total']:>12,} params  {r['macs_total'] / 1e9:.4f} G"
        if name.startswith("global"):
            key = name.split(" ", 1)[1]
            d = r["macs_total"] - macs["spatial no gate (reference)"]
            g = abs(d - EST_DMACS[key]) <= 0.02 * EST_DMACS[key]
            ok &= g
            line += (f"   +{d / 1e6:.2f}M over baseline (expected +{EST_DMACS[key] / 1e6:.2f}M)  "
                     f"{'OK' if g else 'FAIL'}")
        print(line)

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
