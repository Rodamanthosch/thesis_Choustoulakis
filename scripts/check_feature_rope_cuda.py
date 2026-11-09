"""
scripts/check_feature_rope_cuda.py
==================================
GPU check for `feature_rope` (src/feature_rope.py), on the Tiny-ImageNet configs:
  --pair vmamba  : jit-s2-vmamba-feature-rope-{1d,2d-axial}.yaml
                   vs jit-s2-vmamba-baseline.yaml        (d_state 16)
  --pair spatial : jit-s2-spatial-nogate-feature-rope-{1d,2d-axial}.yaml
                   vs jit-s2-spatial-nogate.yaml         (d_state 1)
Run it once at the start of the run's first Kaggle session, like the other
scripts/check_*_cuda.py. The maths and the placement are proven on CPU by
tests/test_feature_rope_cpu.py with a stubbed scan; what is new here is the real
CUDA kernel fed rotated features, and the rotation tables living on the GPU.

  1. apply_feature_rope on CUDA (fp32, and bf16 in -> bf16 out) vs a float64 CPU
     reference; tables on the input's device; a table first built under
     inference_mode still works with autograd.
  2. The mixer with RoPE on CUDA in fp32 matches a CPU copy of the same weights
     run through mamba_ssm's pure-PyTorch selective_scan_ref.
  3. The same weights with RoPE off give a different output on the real kernel.
  4. The config builds through evaluate.build_model and runs forward + backward
     under bf16 autocast at batch 128 with a finite loss and finite gradients.
  5. Training-step speed (fwd+bwd+AdamW, bf16, batch 128) and peak memory, the
     RoPE config vs its base config -- record them for the complexity table.

Run (on GPU, from the repo root):
    python scripts/check_feature_rope_cuda.py --pair vmamba --mode 1d
    python scripts/check_feature_rope_cuda.py --pair spatial --mode 2d_axial
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
from src.feature_rope import apply_feature_rope
from src.models.vmamba import SS2D
from src.models.spatial_mamba import StructureAwareSSM
from scripts.evaluate import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--pair", choices=("vmamba", "spatial", "all"), default="all")
ap.add_argument("--mode", choices=("1d", "2d_axial", "all"), default="all")
args = ap.parse_args()

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True
THETA = 10000.0
CFG = "configs/tiny_imagenet/"
PAIRS = {
    "vmamba": dict(
        base=CFG + "jit-s2-vmamba-baseline.yaml",
        stem=CFG + "jit-s2-vmamba-feature-rope-",
        module=vmamba_mod,
        mixer=lambda mode: SS2D(384, d_state=16, feature_rope=mode)),
    "spatial": dict(
        base=CFG + "jit-s2-spatial-nogate.yaml",
        stem=CFG + "jit-s2-spatial-nogate-feature-rope-",
        module=spatial_mod,
        mixer=lambda mode: StructureAwareSSM(
            384, d_state=1, gate=False, dilations=(1, 2, 3), feature_rope=mode)),
}
pairs = list(PAIRS) if args.pair == "all" else [args.pair]
modes = ["1d", "2d_axial"] if args.mode == "all" else [args.mode]


def reference(x, mode):
    """x: float64 on the CPU, (B, D, H, W). The formula, written out again."""
    _, D, H, W = x.shape
    kw = dict(dtype=torch.float64)
    if mode == "1d":
        omega = THETA ** (-2.0 * torch.arange(D // 2, **kw) / D)
        angles = torch.arange(H * W, **kw).reshape(H, W, 1) * omega
    else:
        omega = THETA ** (-4.0 * torch.arange(D // 4, **kw) / D)
        rows = (torch.arange(H, **kw).reshape(H, 1, 1) * omega).expand(H, W, -1)
        cols = (torch.arange(W, **kw).reshape(1, W, 1) * omega).expand(H, W, -1)
        angles = torch.cat([rows, cols], dim=-1)
    angles = angles.permute(2, 0, 1)[None]
    x0, x1 = x[:, 0::2], x[:, 1::2]
    return torch.stack((x0 * angles.cos() - x1 * angles.sin(),
                        x0 * angles.sin() + x1 * angles.cos()), dim=2).flatten(1, 2)


def perturbed(model, scale=0.02):
    """Leave the zero (identity) init, so every branch affects the output."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn_like(p))
    return model


# ── 1. the rotation on CUDA ──────────────────────────────────────────────────
for mode in modes:
    torch.manual_seed(0)
    x = torch.randn(4, 384, 8, 8, device=dev)
    ref = reference(x.double().cpu(), mode)
    y32 = apply_feature_rope(x, mode, THETA)
    y16 = apply_feature_rope(x.bfloat16(), mode, THETA)
    e32 = (y32.double().cpu() - ref).abs().max().item()
    e16 = ((y16.double().cpu() - ref).abs() / (ref.abs() + 1)).max().item()
    with torch.inference_mode():
        apply_feature_rope(x[:, :32], mode, THETA)      # a width not cached yet
    xg = x[:, :32].clone().requires_grad_()
    apply_feature_rope(xg, mode, THETA).square().sum().backward()
    good = (y32.device == x.device and y32.dtype == torch.float32
            and y16.dtype == torch.bfloat16 and e32 < 1e-4 and e16 < 2e-2
            and torch.isfinite(xg.grad).all().item())
    ok &= good
    print(f"1. {mode}: rotation on CUDA, fp32 max|diff| vs float64 = {e32:.1e}, bf16 max rel "
          f"= {e16:.1e}, bf16 in -> {str(y16.dtype).split('.')[-1]}, autograd after "
          f"inference_mode  {'OK' if good else 'FAIL'}")


for name in pairs:
    P = PAIRS[name]
    mod = P["module"]
    for mode in modes:
        config = P["stem"] + mode.replace("_", "-") + ".yaml"
        tag = f"[{name} {mode}]"

        # ── 2. mixer: CUDA fp32 vs CPU selective_scan_ref ────────────────────
        torch.manual_seed(1)
        cpu = perturbed(P["mixer"](mode).eval())
        gpu = copy.deepcopy(cpu).to(dev)
        xm = torch.randn(4, 64, 384)
        with torch.no_grad():
            y_gpu = gpu(xm.to(dev), 8, 8).cpu()
            mod.selective_scan_fn = selective_scan_ref      # CPU reference path
            try:
                y_cpu = cpu(xm, 8, 8)
            finally:
                mod.selective_scan_fn = selective_scan_fn
        e = ((y_gpu - y_cpu).abs().max() / y_cpu.abs().max()).item()
        good = e < 1e-4
        ok &= good
        print(f"2. {tag} {type(cpu).__name__}: CUDA fp32 vs CPU ref max rel |dy| = {e:.2e}  "
              f"{'OK' if good else 'FAIL'}")

        # ── 3. RoPE acts on the real kernel ──────────────────────────────────
        off = P["mixer"]("none").to(dev).eval()
        off.load_state_dict(gpu.state_dict(), strict=True)
        with torch.no_grad():
            y_off = off(xm.to(dev), 8, 8).cpu()
        moved = ((y_off - y_gpu).abs().max() / y_gpu.abs().max()).item()
        good = moved > 1e-3
        ok &= good
        print(f"3. {tag} same weights, RoPE off vs on: max rel |dy| = {moved:.2e}  "
              f"{'OK' if good else 'FAIL'}")
        del cpu, gpu, off

        # ── 4. the config under bf16 at the training batch ───────────────────
        with open(config) as f:
            cfg = yaml.safe_load(f)
        mc = cfg["model"]
        assert mc["feature_rope"] == mode, config
        torch.manual_seed(0)
        m = perturbed(build_model(cfg).to(dev).train())
        bs = 128
        xi = torch.randn(bs, 3, mc["input_size"], mc["input_size"], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(xi, torch.rand(bs, device=dev),
                    torch.randint(0, mc["num_classes"], (bs,), device=dev))
        loss = (out.float() - xi).square().mean()
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        good = (torch.isfinite(loss).item() and len(grads) > 0
                and all(torch.isfinite(g).all().item() for g in grads))
        ok &= good
        n = sum(p.numel() for p in m.parameters())
        print(f"4. {tag} {os.path.basename(config)}: {n:,} params, bf16 fwd+bwd at batch "
              f"{bs}, loss {loss.item():.4f}  {'OK' if good else 'FAIL'}")
        del m, out, loss, grads
        torch.cuda.empty_cache()

        # ── 5. training-step speed and peak memory ───────────────────────────
        def step_rate(path, bs=128, warmup=5, iters=20):
            with open(path) as f:
                c = yaml.safe_load(f)
            mc = c["model"]
            torch.manual_seed(0)
            torch.cuda.reset_peak_memory_stats()
            m = build_model(c).to(dev).train()
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
        for label, path in (("base (reference)", P["base"]), (f"+ feature_rope {mode}", config)):
            try:
                r, peak = step_rate(path)
            except torch.cuda.OutOfMemoryError:
                ok = False
                print(f"5. {tag} {label:<24} OUT OF MEMORY at batch 128  FAIL")
                torch.cuda.empty_cache()
                continue
            r_ref = r_ref or r
            print(f"5. {tag} {label:<24} {r:6.2f} it/s ({r / r_ref:.3f}x reference)  "
                  f"peak {peak:5.2f} GiB")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
