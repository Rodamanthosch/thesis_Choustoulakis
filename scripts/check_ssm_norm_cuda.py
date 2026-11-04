"""
scripts/check_ssm_norm_cuda.py
==============================
GPU check for `ssm_norm` -- Jamba's RMSNorm on the SSM parameters:
  --pair vmamba  : configs/tiny_imagenet/jit-s2-vmamba-ssmnorm-bcdt.yaml
                   vs jit-s2-vmamba-baseline.yaml       (dt_r, B, C; d_state 16)
  --pair spatial : configs/tiny_imagenet/jit-s2-spatial-nogate-ssmnorm-dt.yaml
                   vs jit-s2-spatial-nogate.yaml        (dt_r only; d_state 1)
Run it once at the start of the run's first Kaggle session, like the other
scripts/check_*_cuda.py. The maths is proven on CPU by tests/test_ssm_norm_cpu.py;
what is new on the GPU is the stock CUDA scan fed normalised dt / B / C.

  1. ssm_rms_norm on CUDA (fp32, and bf16 with an fp32 weight) vs a float64
     CPU copy of JambaRMSNorm; bf16 in -> bf16 out.
  2. The mixer with the norm (random gains) on CUDA in fp32 matches a CPU copy
     of the same weights run through mamba_ssm's pure-PyTorch selective_scan_ref.
  3. Scale invariance on the real kernel: scaling every block's x_proj dt (and
     B, C for bcdt) rows by 5 leaves the model output unchanged (eps only),
     while it moves the model without the norm.
  4. The config builds through evaluate.build_model and runs forward + backward
     under bf16 autocast at batch 128 with a finite loss and finite grads that
     reach every norm gain.
  5. Training-step speed (fwd+bwd+AdamW, bf16, batch 128) and peak memory,
     the norm config vs its base config -- record them for the complexity table.

Run (on GPU, from the repo root):
    python scripts/check_ssm_norm_cuda.py --pair vmamba
    python scripts/check_ssm_norm_cuda.py --pair spatial
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

import src.models.vmamba as vmamba_mod
import src.models.spatial_mamba as spatial_mod
from src.models.vmamba import SS2D
from src.models.spatial_mamba import StructureAwareSSM
from src.primitives import ssm_rms_norm
from scripts.evaluate import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--pair", choices=("vmamba", "spatial", "all"), default="all")
args = ap.parse_args()

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True
PAIRS = {
    "vmamba": dict(
        base="configs/tiny_imagenet/jit-s2-vmamba-baseline.yaml",
        norm="configs/tiny_imagenet/jit-s2-vmamba-ssmnorm-bcdt.yaml",
        module=vmamba_mod, mode="bcdt",
        mixer=lambda: SS2D(384, d_state=16, ssm_norm="bcdt")),
    "spatial": dict(
        base="configs/tiny_imagenet/jit-s2-spatial-nogate.yaml",
        norm="configs/tiny_imagenet/jit-s2-spatial-nogate-ssmnorm-dt.yaml",
        module=spatial_mod, mode="dt",
        mixer=lambda: StructureAwareSSM(384, d_state=1, gate=False,
                                        dilations=(1, 2, 3), ssm_norm="dt")),
}
pairs = list(PAIRS) if args.pair == "all" else [args.pair]


# ── 1. the norm op on CUDA ───────────────────────────────────────────────────
torch.manual_seed(0)
w = torch.randn(16, device=dev)
x = 3 * torch.randn(8, 16, 64, device=dev)                  # (B, n, L), normalised over n
xd, wd = x.double().cpu(), w.double().cpu()
ref = wd[None, :, None] * xd / torch.sqrt(xd.pow(2).mean(1, keepdim=True) + 1e-6)
y32 = ssm_rms_norm(x, w[None, :, None], dim=1)
y16 = ssm_rms_norm(x.bfloat16(), w[None, :, None], dim=1)
e32 = (y32.double().cpu() - ref).abs().max().item()
e16 = ((y16.double().cpu() - ref).abs() / (ref.abs() + 1)).max().item()
good = y16.dtype == torch.bfloat16 and e32 < 1e-5 and e16 < 2e-2
ok &= good
print(f"1. ssm_rms_norm on CUDA: fp32 max|diff| vs float64 = {e32:.1e}, bf16 max rel "
      f"= {e16:.1e}, bf16 in -> {str(y16.dtype).split('.')[-1]}  {'OK' if good else 'FAIL'}")


for name in pairs:
    P = PAIRS[name]
    mod = P["module"]

    # ── 2. mixer: CUDA fp32 vs CPU selective_scan_ref ────────────────────────
    torch.manual_seed(1)
    cpu = P["mixer"]().eval()
    with torch.no_grad():
        for p in cpu.parameters():
            p.add_(0.02 * torch.randn_like(p))              # random gains too
    gpu = copy.deepcopy(cpu).to(dev)
    xm = torch.randn(4, 64, 384)
    with torch.no_grad():
        y_gpu = gpu(xm.to(dev), 8, 8).cpu()
        mod.selective_scan_fn = selective_scan_ref          # CPU reference path
        try:
            y_cpu = cpu(xm, 8, 8)
        finally:
            mod.selective_scan_fn = selective_scan_fn
    e = ((y_gpu - y_cpu).abs().max() / y_cpu.abs().max()).item()
    good = e < 1e-4
    ok &= good
    print(f"2. [{name}] {type(cpu).__name__} ssm_norm={P['mode']!r}: CUDA fp32 vs CPU ref "
          f"max rel |dy| = {e:.2e}  {'OK' if good else 'FAIL'}")
    del cpu, gpu

    # ── 3. scale invariance on the real kernel ───────────────────────────────
    def scale_effect(path):
        cfg = yaml.safe_load(open(path))
        mc = cfg["model"]
        torch.manual_seed(0)
        m = build_model(cfg).to(dev).eval()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.02 * torch.randn_like(p))
        torch.manual_seed(2)
        xi = torch.randn(4, 3, mc["input_size"], mc["input_size"], device=dev)
        t = torch.rand(4, device=dev)
        y = torch.randint(0, mc["num_classes"], (4,), device=dev)
        blk = m.blocks[0].mixer
        R, N = blk.dt_rank, blk.d_state
        rows = [slice(0, R)] + ([slice(R, R + N), slice(R + N, R + 2 * N)]
                                if P["mode"] == "bcdt" else [])
        with torch.no_grad():
            o0 = m(xi, t, y)
            for b in m.blocks:
                for sl in rows:
                    b.mixer.x_proj_weight[..., sl, :] *= 5.0
            o1 = m(xi, t, y)
        del m
        return ((o1 - o0).abs().max() / o0.abs().max()).item()

    s_norm, s_base = scale_effect(P["norm"]), scale_effect(P["base"])
    good = s_norm < 1e-3 and s_base > 1e-2
    ok &= good
    print(f"3. [{name}] x_proj rows x5 (fp32): with the norm the output moves {s_norm:.1e}, "
          f"without it {s_base:.2e}  {'OK' if good else 'FAIL'}")

    # ── 4. the config under bf16 at the training batch ───────────────────────
    cfg = yaml.safe_load(open(P["norm"]))
    mc = cfg["model"]
    torch.manual_seed(0)
    m = build_model(cfg).to(dev).train()
    with torch.no_grad():                    # leave the zero (identity) init so
        for p in m.parameters():             # every branch receives gradient
            p.add_(0.02 * torch.randn_like(p))
    bs = 128
    xi = torch.randn(bs, 3, mc["input_size"], mc["input_size"], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(xi, torch.rand(bs, device=dev),
                torch.randint(0, mc["num_classes"], (bs,), device=dev))
    loss = (out.float() - xi).square().mean()
    loss.backward()
    gains = [p for n, p in m.named_parameters() if n.endswith("_norm_weight")]
    good = (torch.isfinite(loss).item() and len(gains) > 0
            and all(torch.isfinite(p.grad).all().item()
                    for p in m.parameters() if p.grad is not None)
            and all(p.grad.abs().sum().item() > 0 for p in gains))
    ok &= good
    n = sum(p.numel() for p in m.parameters())
    print(f"4. [{name}] {os.path.basename(P['norm'])}: {n:,} params, {len(gains)} gain "
          f"vectors, bf16 fwd+bwd at batch {bs}, loss {loss.item():.4f}  "
          f"{'OK' if good else 'FAIL'}")
    del m, out, loss
    torch.cuda.empty_cache()

    # ── 5. training-step speed and peak memory ───────────────────────────────
    def step_rate(path, bs=128, warmup=5, iters=20):
        cfg = yaml.safe_load(open(path))
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

    r_ref = None
    for label, path in (("base (reference)", P["base"]), (f"+ ssm_norm {P['mode']}", P["norm"])):
        try:
            r, peak = step_rate(path)
        except torch.cuda.OutOfMemoryError:
            ok = False
            print(f"5. [{name}] {label:<20} OUT OF MEMORY at batch 128  FAIL")
            torch.cuda.empty_cache()
            continue
        r_ref = r_ref or r
        print(f"5. [{name}] {label:<20} {r:6.2f} it/s ({r / r_ref:.3f}x reference)  "
              f"peak {peak:5.2f} GiB")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
