"""
scripts/check_incontext_diagonal_cuda.py
========================================
GPU check for in_context_layout="diagonal" -- run once at the start of the
first Kaggle session of the diagonal arm.

The diagonal layout hands the kernel an ordinary sequence of length HW + n;
what is new is the per-path gather (placement) and inverse gather (merge) on
the GPU, and their backward (index scatter-add).

  1. Exactness: the diagonal config's model in fp32, real CUDA kernel vs
     mamba_ssm's selective_scan_ref (same weights, same inputs), forward and
     input gradient.
  2. End to end: build via evaluate.build_model, forward + backward under bf16
     autocast, assert the output shape and the scan length per block
     (64 x4, 72 x8 on Tiny-IN with in_context_start 4).
  3. Cost: train-step it/s and peak memory at batch 128, bf16, for the three
     8-token placements (prefix K=8, row, diagonal) -- all scan L = 72 from
     block 4, so any gap is the gather / interleave overhead.

Run (on GPU, from the repo root):
    PYTHONPATH=. python scripts/check_incontext_diagonal_cuda.py
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import src.models.vmamba as vm
from scripts.evaluate import build_model

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True

DIAG = "configs/tiny_imagenet/jit-s2-vmamba-incontext-diagonal-class.yaml"
ROW = "configs/tiny_imagenet/jit-s2-vmamba-incontext-row-class.yaml"
PREFIX = "configs/tiny_imagenet/jit-s2-vmamba-incontext-class.yaml"


def load(path, **model_over):
    cfg = yaml.safe_load(open(path))
    cfg["model"].update(model_over)
    return cfg


def runs(lengths):
    """[64,64,72,...] -> '64 x2, 72 x..' for a one-line report."""
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


cfg = load(DIAG)
mc = cfg["model"]
res, n_cls = mc["input_size"], mc["num_classes"]

# ── 1. exactness: CUDA kernel vs reference through the gather path (fp32) ────
torch.manual_seed(0)
m = build_model(cfg).to(dev).float()
x = torch.randn(4, 3, res, res, device=dev, requires_grad=True)
t = torch.rand(4, device=dev)
y = torch.randint(0, n_cls, (4,), device=dev)


def fwd_bwd(scan):
    vm.selective_scan_fn = scan
    try:
        x.grad = None
        out = m(x, t, y)
        out.pow(2).mean().backward()
        return out.detach(), x.grad.detach().clone()
    finally:
        vm.selective_scan_fn = selective_scan_fn


o_k, g_k = fwd_bwd(selective_scan_fn)
o_r, g_r = fwd_bwd(selective_scan_ref)
e_o = (o_k - o_r).abs().max().item() / max(o_r.abs().max().item(), 1e-12)
e_g = (g_k - g_r).abs().max().item() / max(g_r.abs().max().item(), 1e-12)
good = e_o < 1e-3 and e_g < 1e-3
ok &= good
print(f"1. kernel vs reference, diagonal model fp32: rel max|d out| = {e_o:.2e}, "
      f"rel max|d grad_x| = {e_g:.2e}  {'OK' if good else 'FAIL'}")
del m, x
torch.cuda.empty_cache()

# ── 2. end to end under bf16, scan length per block ──────────────────────────
SEEN = []


def recording(u, *a, **kw):
    SEEN.append(u.shape[-1])
    return selective_scan_fn(u, *a, **kw)


hw = (res // mc["patch_size"]) ** 2
start, depth, n_tok = mc["in_context_start"], mc["depth"], mc["in_context_len"]
want = [hw] * start + [hw + n_tok] * (depth - start)
m = build_model(cfg).to(dev)
xb = torch.randn(2, 3, res, res, device=dev)
vm.selective_scan_fn = recording
try:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(xb, torch.rand(2, device=dev), torch.randint(0, n_cls, (2,), device=dev))
    out.float().mean().backward()
finally:
    vm.selective_scan_fn = selective_scan_fn
good = (out.shape == xb.shape and SEEN == want
        and m.incontext_pos_embed.grad is not None
        and torch.isfinite(m.incontext_pos_embed.grad).all().item())
ok &= good
print(f"2. bf16 fwd+bwd: out {tuple(out.shape)}  scan L per block: {runs(SEEN)} "
      f"(want {runs(want)})  {'OK' if good else 'FAIL'}")
del m, out
torch.cuda.empty_cache()

# ── 3. cost of the three 8-token placements ──────────────────────────────────
BS, STEPS, WARM = 128, 20, 5
print(f"3. train step, batch {BS}, bf16 (scan L = {hw + n_tok} from block {start}):")
for name, c in (("prefix K=8", load(PREFIX, in_context_len=n_tok)),
                ("row", load(ROW)),
                ("diagonal", copy.deepcopy(cfg))):
    torch.manual_seed(0)
    m = build_model(c).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    xb = torch.randn(BS, 3, res, res, device=dev)
    tb = torch.rand(BS, device=dev)
    yb = torch.randint(0, n_cls, (BS,), device=dev)
    torch.cuda.reset_peak_memory_stats()
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
    print(f"   {name:<11} {its:6.2f} it/s   peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB")
    del m, opt
    torch.cuda.empty_cache()

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
