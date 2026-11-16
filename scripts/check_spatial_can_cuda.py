"""
scripts/check_spatial_can_cuda.py
=================================
GPU check for JiT-S2-Spatial-Mamba with CAN-style conditional kernels
(configs/tiny_imagenet/jit-s2-spatial-nogate-can.yaml: dwconv_can + sasf_can).
Run it once at the start of the run's first Kaggle session, in the same spirit
as scripts/check_spatial_auxswap_cuda.py.

What is new on the GPU: per-sample depthwise kernels K_l + dK(c_b) as one
grouped conv over B * C channels (cuDNN), in fp32 under bf16 autocast.

  1. Kernel exactness: the mixer with random dK on CUDA (fp32) matches a CPU
     copy run through mamba_ssm's pure-PyTorch selective_scan_ref, and the
     batched CUDA conv matches a per-sample loop of static-kernel copies.
  2. The config builds through evaluate.build_model and runs forward + backward
     under bf16 autocast: finite output, finite non-zero grads into both
     generators; the CAN site outputs keep the static path's dtypes.
  3. Training-step speed (fwd+bwd+AdamW, bf16, batch 128) and peak memory for
     the config and jit-s2-spatial-nogate.
  4. src/flops_counter.count_complexity on both configs: the difference must be
     the generators' 9*D^2 + 27*D^2 = 5,308,416 MACs (~0.0053 G), within 1%;
     the exact count is printed (needs fvcore). The per-sample convs cost the
     static convs' MACs.

Run (on GPU, from the repo root):  python scripts/check_spatial_can_cuda.py
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import src.models.spatial_mamba as spatial_mod
from src.models.spatial_mamba import StructureAwareSSM
from scripts.evaluate import build_model

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True
CONFIG = "configs/tiny_imagenet/jit-s2-spatial-nogate-can.yaml"
BASE = "configs/tiny_imagenet/jit-s2-spatial-nogate.yaml"
EXPECTED_DMACS = 9 * 384 * 384 + 27 * 384 * 384           # 5,308,416


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# ── 1. kernel exactness ──────────────────────────────────────────────────────
torch.manual_seed(1)
cpu = StructureAwareSSM(384, gate=False, dilations=(1, 2, 3)).eval()
with torch.no_grad():
    for p in cpu.parameters():
        p.add_(0.02 * torch.randn_like(p))
B = 4
x = torch.randn(B, 64, 384)
can = (0.05 * torch.randn(B, 384, 3, 3), 0.05 * torch.randn(B, 3, 384, 3, 3))
gpu = copy.deepcopy(cpu).to(dev)
can_gpu = tuple(t.to(dev) for t in can)
with torch.no_grad():
    y_gpu = gpu(x.to(dev), 8, 8, can_gpu)
    spatial_mod.selective_scan_fn = selective_scan_ref   # CPU reference path
    try:
        y_cpu = cpu(x, 8, 8, can)
    finally:
        spatial_mod.selective_scan_fn = selective_scan_fn
    loop = []
    for b in range(B):                                   # static-kernel copies, CUDA
        m = copy.deepcopy(gpu)
        m.conv2d.weight.add_(can_gpu[0][b].unsqueeze(1))
        for j, k in enumerate(m.state_fusion.kernels):
            k.add_(can_gpu[1][b, j].unsqueeze(1))
        loop.append(m(x[b:b + 1].to(dev), 8, 8))
    loop = torch.cat(loop)
e1, e2 = rel(y_gpu.cpu(), y_cpu), rel(y_gpu, loop)
g = e1 < 1e-4 and e2 < 1e-5
ok &= g
print(f"1. mixer with dK, CUDA fp32: vs CPU ref {e1:.2e}, vs per-sample loop {e2:.2e}  "
      f"{'OK' if g else 'FAIL'}")

# ── 2. the config end to end under bf16 ──────────────────────────────────────
cfg = yaml.safe_load(open(CONFIG))
mc = cfg["model"]
torch.manual_seed(0)
m = build_model(cfg).to(dev).train()
with torch.no_grad():                        # leave the zero init so every
    for p in m.parameters():                 # branch receives gradient
        p.add_(0.02 * torch.randn_like(p))
xi = torch.randn(2, 3, mc["input_size"], mc["input_size"], device=dev)
ti = torch.rand(2, device=dev)
yi = torch.randint(0, mc["num_classes"], (2,), device=dev)
dt = {}
mx = m.blocks[0].mixer
hooks = [mx.act.register_forward_hook(lambda mod, i, o: dt.__setitem__("dw", i[0].dtype)),
         mx.state_fusion.register_forward_hook(
             lambda mod, i, o: dt.update(sasf_in=i[0].dtype, sasf_out=o.dtype))]
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(xi, ti, yi)
for h in hooks:
    h.remove()
out.float().square().mean().backward()
grads = [m.can_dwconv.weight.grad, m.can_sasf.weight.grad]
good = (out.shape == xi.shape and torch.isfinite(out).all().item()
        and all(g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0
                for g in grads)
        and dt["dw"] == torch.bfloat16 and dt["sasf_out"] == dt["sasf_in"])
ok &= good
n = sum(p.numel() for p in m.parameters())
print(f"2. {os.path.basename(CONFIG)}: {n:,} params, bf16 fwd+bwd, grads into both "
      f"generators; DWConv out {dt['dw']}, SASF {dt['sasf_in']} -> {dt['sasf_out']}  "
      f"{'OK' if good else 'FAIL'}")
del m, out
torch.cuda.empty_cache()


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


r_ref = None
for name, path in (("spatial no gate (reference)", BASE), ("+ CAN dwconv + sasf", CONFIG)):
    try:
        r, peak = step_rate(yaml.safe_load(open(path)))
    except torch.cuda.OutOfMemoryError:
        ok = False
        print(f"3. {name:<30} OUT OF MEMORY at batch 128  FAIL")
        torch.cuda.empty_cache()
        continue
    r_ref = r_ref or r
    print(f"3. {name:<30} {r:6.2f} it/s ({r / r_ref:.3f}x reference)  peak {peak:5.2f} GiB")

# ── 4. MACs: only the two generators are new ─────────────────────────────────
try:
    from src.flops_counter import count_complexity
except ImportError as e:
    ok = False
    print(f"4. MACs  FAIL (fvcore missing: pip install fvcore) ({e})")
else:
    macs = {}
    for name, path in (("base", BASE), ("can", CONFIG)):
        c = yaml.safe_load(open(path))
        torch.manual_seed(0)
        r = count_complexity(build_model(c), img_size=c["model"]["input_size"],
                             in_channels=c["model"]["in_channels"],
                             num_classes=c["model"]["num_classes"], device=dev)
        macs[name] = r["macs_total"]
    d = macs["can"] - macs["base"]
    g = abs(d - EXPECTED_DMACS) <= 0.01 * EXPECTED_DMACS     # exact count printed

    ok &= g
    print(f"4. MACs {macs['base'] / 1e9:.4f} G -> {macs['can'] / 1e9:.4f} G: "
          f"+{d / 1e9:.4f} G ({d:,} vs expected {EXPECTED_DMACS:,})  {'OK' if g else 'FAIL'}")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
