"""
scripts/check_vmamba_global_cuda.py
===================================
GPU check for model=vmamba_global (src/models/vmamba_global.py) -- run once at
the start of the first Kaggle session of each global-token arm.

The scan kernel is called exactly as in the baseline (image tokens only, L =
64 in EVERY block); what is new is the fp32 memory / read einsums around it.

  1. Exactness (fp32): every global config's model, the real CUDA kernel vs
     mamba_ssm's selective_scan_ref (patched into BOTH vmamba and
     vmamba_global, since blocks 0..start-1 are plain SS2D), forward and
     input gradient.
  2. End to end (--config): evaluate.build_model, forward + backward under
     bf16 autocast; the scan length is 64 in every block (the tokens never
     enter the scan); S / G / ve / read / token rows are fp32; the slots and
     every alpha_* get finite, non-zero gradients.
  3. Cost: train-step it/s and peak memory at batch 128, bf16, for the
     baseline, the m = 16 and m = 32 class prefixes and the four global
     configs.
  4. fvcore (src/flops_counter.count_complexity): params and MACs of the same
     list, against the plan's estimate (an estimate: only printed).

Run (on GPU, from the repo root):
    PYTHONPATH=. python scripts/check_vmamba_global_cuda.py --config full
    (--config writeonly | softmax | full-m32; --skip-cost to stop after 2)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import src.models.vmamba as vm
import src.models.vmamba_global as vg
from scripts.evaluate import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="full", choices=("full", "writeonly", "softmax", "full-m32"))
ap.add_argument("--skip-cost", action="store_true")
args = ap.parse_args()

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True

CFG_DIR = "configs/tiny_imagenet"
GLOBAL = {n: f"{CFG_DIR}/jit-s2-vmamba-global-{n}.yaml"
          for n in ("full", "writeonly", "softmax", "full-m32")}
BASE = f"{CFG_DIR}/jit-s2-vmamba-baseline.yaml"
PREFIX = f"{CFG_DIR}/jit-s2-vmamba-incontext-class.yaml"
PARAMS = {"full": 31_046_208, "writeonly": 31_043_520, "softmax": 31_046_208,
          "full-m32": 31_052_352}
# MACs over the baseline: the token rows' linears (no scan) + the new einsums
# (block 11: no G, no token read; softmax there: no S either).
EST_DMACS = {"full": 233_963_520, "writeonly": 222_953_472, "softmax": 234_094_592,
             "full-m32": 444_334_080}


def load(path, **model_over):
    cfg = yaml.safe_load(open(path))
    cfg["model"].update(model_over)
    return cfg


def set_scan(fn):
    vm.selective_scan_fn = fn
    vg.selective_scan_fn = fn


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


# ── 1. exactness: CUDA kernel vs reference (fp32), every global config ───────
for name, path in GLOBAL.items():
    cfg = load(path)
    mc = cfg["model"]
    res, n_cls = mc["input_size"], mc["num_classes"]
    torch.manual_seed(0)
    m = build_model(cfg).to(dev).float()
    with torch.no_grad():                      # leave adaLN-Zero's identity
        for p in m.parameters():
            p.add_(0.02 * torch.randn_like(p))
    x = torch.randn(4, 3, res, res, device=dev, requires_grad=True)
    t = torch.rand(4, device=dev)
    y = torch.randint(0, n_cls, (4,), device=dev)

    def fwd_bwd(scan):
        set_scan(scan)
        try:
            x.grad = None
            out = m(x, t, y)
            out.pow(2).mean().backward()
            return out.detach(), x.grad.detach().clone()
        finally:
            set_scan(selective_scan_fn)

    o_k, g_k = fwd_bwd(selective_scan_fn)
    o_r, g_r = fwd_bwd(selective_scan_ref)
    e_o = (o_k - o_r).abs().max().item() / max(o_r.abs().max().item(), 1e-12)
    e_g = (g_k - g_r).abs().max().item() / max(g_r.abs().max().item(), 1e-12)
    good = e_o < 1e-3 and e_g < 1e-3
    ok &= good
    print(f"1. kernel vs reference, {name:<9} fp32: rel max|d out| = {e_o:.2e}, "
          f"rel max|d grad_x| = {e_g:.2e}  {'OK' if good else 'FAIL'}")
    del m, x
    torch.cuda.empty_cache()

# ── 2. end to end under bf16 (--config) ──────────────────────────────────────
cfg = load(GLOBAL[args.config])
mc = cfg["model"]
res, n_cls = mc["input_size"], mc["num_classes"]
hw = (res // mc["patch_size"]) ** 2
SEEN, DTYPES = [], set()


def recording(u, *a, **kw):
    SEEN.append(u.shape[-1])
    return selective_scan_fn(u, *a, **kw)


terms_orig = vg.SS2DGlobal.terms


def terms_rec(self, x, H, W):
    out = terms_orig(self, x, H, W)
    for key in ("ve", "S", "G", "read", "y_e"):
        if out[key] is not None:
            DTYPES.add((key, out[key].dtype))
    return out


torch.manual_seed(0)
m = build_model(cfg).to(dev)
with torch.no_grad():
    for blk in m.blocks:                       # let gradients reach the tokens
        blk.adaLN_modulation[-1].weight.normal_(0, 0.02)
    m.final_layer.linear.weight.normal_(0, 0.02)
n_par = sum(p.numel() for p in m.parameters())
xb = torch.randn(2, 3, res, res, device=dev)
set_scan(recording)
vg.SS2DGlobal.terms = terms_rec
try:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(xb, torch.rand(2, device=dev), torch.randint(0, n_cls, (2,), device=dev))
    out.float().pow(2).mean().backward()
finally:
    set_scan(selective_scan_fn)
    vg.SS2DGlobal.terms = terms_orig
grads = {"incontext_pos_embed": m.incontext_pos_embed.grad}
grads.update({n: p.grad for n, p in m.named_parameters() if ".alpha_" in n})
grads_ok = all(g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0
               for g in grads.values())
fp32 = all(dt == torch.float32 for _, dt in DTYPES) and len(DTYPES) > 0
good = (out.shape == xb.shape and SEEN == [hw] * mc["depth"] and fp32 and grads_ok
        and n_par == PARAMS[args.config])
ok &= good
print(f"2. {args.config} bf16 fwd+bwd: out {tuple(out.shape)}  scan L per block: {runs(SEEN)} "
      f"(want {hw} x{mc['depth']})  memories/reads {sorted({str(d) for _, d in DTYPES})}  "
      f"{len(grads)} slot/gain grads finite & non-zero: {grads_ok}  params {n_par:,}  "
      f"{'OK' if good else 'FAIL'}")
del m, out
torch.cuda.empty_cache()

if args.skip_cost:
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

# ── 3. cost ──────────────────────────────────────────────────────────────────
RUNS = [("baseline", load(BASE)),
        ("prefix m=16", load(PREFIX, in_context_len=16)),
        ("prefix m=32", load(PREFIX))]
RUNS += [(f"global {n}", load(p)) for n, p in GLOBAL.items()]
BS, STEPS, WARM = 128, 20, 5
print(f"3. train step, batch {BS}, bf16:")
for name, c in RUNS:
    torch.manual_seed(0)
    m = build_model(c).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    xb = torch.randn(BS, 3, res, res, device=dev)
    tb = torch.rand(BS, device=dev)
    yb = torch.randint(0, n_cls, (BS,), device=dev)
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(WARM + STEPS):
            if i == WARM:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(xb, tb, yb).float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        its = STEPS / (time.time() - t0)
        print(f"   {name:<17} {its:6.2f} it/s   peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB")
    except torch.cuda.OutOfMemoryError:
        ok = False
        print(f"   {name:<17} OUT OF MEMORY at batch {BS}  FAIL")
    del m, opt
    torch.cuda.empty_cache()

# ── 4. fvcore params / MACs ──────────────────────────────────────────────────
try:
    from src.flops_counter import count_complexity
except ImportError as e:
    ok = False
    print(f"4. MACs  FAIL (fvcore missing: pip install fvcore) ({e})")
else:
    macs = {}
    print("4. fvcore (MACs, the repo's 'GFLOPs'):")
    for name, c in RUNS:
        torch.manual_seed(0)
        r = count_complexity(build_model(c), img_size=c["model"]["input_size"],
                             in_channels=c["model"]["in_channels"],
                             num_classes=c["model"]["num_classes"], device=dev)
        macs[name] = r["macs_total"]
        line = f"   {name:<17} {r['params_total']:>12,} params  {r['macs_total'] / 1e9:.4f} G"
        key = name.split(" ", 1)[1] if name.startswith("global") else None
        if key:
            d = r["macs_total"] - macs["baseline"]
            line += (f"   +{d / 1e6:.2f}M over baseline (estimate +{EST_DMACS[key] / 1e6:.2f}M, "
                     f"ratio {d / EST_DMACS[key]:.3f})")
        print(line)

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
