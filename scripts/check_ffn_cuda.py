"""
scripts/check_ffn_cuda.py
=========================
GPU check for the conv FFNs (ffn: convglu | glumbconv). Run it once at the start
of the first Kaggle session of an FFN run, in the same spirit as
scripts/check_incontext_row_cuda.py.

The FFNs touch no mamba kernel. What is new on the GPU is the cuDNN depthwise
conv, the NCHW round trip on the token grid, and the prefix/grid split under
bf16 autocast.

  1. Each FFN at D=384 on the 8x8 Tiny-ImageNet grid with a 4-token prefix:
     CUDA fp32 vs a CPU float64 copy of the same module, and the prefix/grid
     isolation (grid output == the grid run alone) on the GPU.
  2. Both FFN configs, plus each with an in-context prefix: build them through
     evaluate.build_model, run forward + backward under bf16 autocast, and
     assert a finite output and finite grads into the depthwise conv.
  3. Training-step throughput (fwd+bwd, bf16, batch 128) against the same
     config with ffn=swiglu, for the complexity table.

Run (on GPU, from the repo root):  python scripts/check_ffn_cuda.py
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from src.primitives import ConvGLUFFN, GLUMBConvFFN
from scripts.evaluate import build_model

assert torch.cuda.is_available(), "This check needs a GPU."
dev = "cuda"
ok = True

# ── 1. module accuracy + prefix isolation on the real conv ───────────────────
torch.manual_seed(42)
D, Hg, P = 384, 8, 4
for cls in (ConvGLUFFN, GLUMBConvFFN):
    ref = cls(D, 4 * D).double().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.02 * torch.randn_like(p))       # non-zero conv bias too
    gpu = copy.deepcopy(ref).float().to(dev)
    x = torch.randn(4, P + Hg * Hg, D, dtype=torch.float64)
    with torch.no_grad():
        y_ref = ref(x, Hg, Hg)
        y_gpu = gpu(x.float().to(dev), Hg, Hg)
        err = (y_gpu.double().cpu() - y_ref).abs().max().item()
        iso = (y_gpu[:, P:] - gpu(x[:, P:].float().to(dev), Hg, Hg)).abs().max().item()
    good = err < 1e-4 and iso < 1e-5
    ok &= good
    print(f"1. {cls.__name__:<13} CUDA fp32 vs CPU fp64: max|dy| = {err:.2e}  "
          f"grid-alone max|dy| = {iso:.2e}  {'OK' if good else 'FAIL'}")

# ── 2. end-to-end on both configs (and with an in-context prefix) ────────────
CONFIGS = [
    "configs/tiny_imagenet/jit-s2-vmamba-ffn-convglu.yaml",
    "configs/tiny_imagenet/jit-s2-vmamba-ffn-glumbconv.yaml",
]


def with_prefix(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(in_context_len=4, in_context_content="class",
                        in_context_start=4)
    return cfg


for path in CONFIGS:
    base = yaml.safe_load(open(path))
    for tag, cfg in (("", base), (" +prefix K=4", with_prefix(base))):
        mc = cfg["model"]
        res, n_cls = mc["input_size"], mc["num_classes"]
        torch.manual_seed(0)
        m = build_model(cfg).to(dev).train()
        x = torch.randn(2, 3, res, res, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, torch.rand(2, device=dev),
                    torch.randint(0, n_cls, (2,), device=dev))
        out.float().square().mean().backward()
        conv = m.blocks[-1].mlp.dwconv if mc["ffn"] == "convglu" \
            else m.blocks[-1].mlp.depth_conv
        good = (out.shape == x.shape and torch.isfinite(out).all().item()
                and conv.weight.grad is not None
                and torch.isfinite(conv.weight.grad).all().item())
        ok &= good
        n = sum(p.numel() for p in m.parameters())
        print(f"2. {os.path.basename(path) + tag:<48} out {tuple(out.shape)}  "
              f"{n / 1e6:.2f}M params  {'OK' if good else 'FAIL'}")
        del m, out
        torch.cuda.empty_cache()

# ── 3. training-step throughput vs swiglu ────────────────────────────────────
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


base = yaml.safe_load(open(CONFIGS[0]))
swi = copy.deepcopy(base)
swi["model"]["ffn"] = "swiglu"
r_swi = step_rate(swi)
print(f"3. swiglu    {r_swi:6.2f} it/s (bs 128, bf16, fwd+bwd+AdamW)")
for path in CONFIGS:
    cfg = yaml.safe_load(open(path))
    r = step_rate(cfg)
    print(f"3. {cfg['model']['ffn']:<10}{r:6.2f} it/s  ({r / r_swi:.3f}x swiglu)")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
