"""
src/utils.py
============
Shared utilities extracted from the notebooks.

Checkpoint helpers: exact from all three notebooks (Cell 25 — identical in all).
LR scheduler:       exact from all three notebooks (Cell 23 — identical in all).
FID/IS evaluation:  exact from all three notebooks (Cell 38 — identical in all).
Config loading:     load YAML hyperparameter files from configs/.
Throughput:         from the throughput cell (Cell 40) in all notebooks.
"""

import os
import time
import json
import yaml
from collections import OrderedDict

import torch


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checkpoint helpers (exact from notebooks Cell 25) ─────────────────────────

def _ema_to_state_dict(denoiser, which):
    """Convert ema_params{which} (a list of tensors aligned with parameters()) into
    an OrderedDict keyed by named_parameters() — so it's a regular state_dict shape."""
    ema = denoiser.ema_params1 if which == 1 else denoiser.ema_params2
    if ema is None:
        return None
    sd = OrderedDict()
    for (name, _), tensor in zip(denoiser.named_parameters(), ema):
        sd[name] = tensor.detach().cpu().clone()
    return sd


def _load_ema_from_state_dict(denoiser, state_dict, which):
    """Inverse of _ema_to_state_dict — write into denoiser.ema_params{which}."""
    if denoiser.ema_params1 is None or denoiser.ema_params2 is None:
        denoiser.init_ema()
    ema = denoiser.ema_params1 if which == 1 else denoiser.ema_params2
    name_to_idx = {name: i for i, (name, _) in enumerate(denoiser.named_parameters())}
    device = next(denoiser.parameters()).device
    with torch.no_grad():
        for name, tensor in state_dict.items():
            i = name_to_idx[name]
            ema[i].copy_(tensor.to(device))


def save_checkpoint(path, epoch, global_step, denoiser, optimizer, losses, best_loss):
    payload = {
        "epoch":        epoch,
        "global_step":  global_step,
        "model":        denoiser.net.state_dict(),
        "ema1":         _ema_to_state_dict(denoiser, 1),
        "ema2":         _ema_to_state_dict(denoiser, 2),
        "optimizer":    optimizer.state_dict(),
        "losses":       list(losses),
        "best_loss":    best_loss,
    }
    torch.save(payload, path)


def load_checkpoint(path, denoiser, optimizer=None, map_location=None):
    payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    denoiser.net.load_state_dict(payload["model"])
    if payload.get("ema1") is not None:
        _load_ema_from_state_dict(denoiser, payload["ema1"], 1)
    if payload.get("ema2") is not None:
        _load_ema_from_state_dict(denoiser, payload["ema2"], 2)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


# ── LR scheduler (exact from notebooks Cell 23) ──────────────────────────────

def paper_peak_lr(batch_size: int, blr: float = 5e-5) -> float:
    """
    Paper's exact LR rule (from main_jit.py):
        actual_lr = blr * total_batch / 256,  with blr = 5e-5
    At total batch 1024 (8 GPUs × 128) the paper gets actual_lr = 2e-4 (Table 9).
    For single-GPU batch 128: 5e-5 * 128 / 256 = 2.5e-5
    """
    return blr * batch_size / 256


def make_lr_fn(peak_lr: float, warmup_steps: int):
    """
    Returns the lr_at_step function used in all notebooks.
    Linear warmup, then constant (matches all three Cell 23 blocks exactly).
    """
    def lr_at_step(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        return peak_lr
    return lr_at_step


# ── FID / IS evaluation (exact from notebooks Cell 38) ───────────────────────

def evaluate_fid_is(gen_dir: str, real_dir: str, device: str = "cuda", n_samples: int = 10000):
    """
    Compute FID and IS between generated and real image folders.
    Uses torch-fidelity if available (same tool as official JiT repo),
    falls back to pytorch-fid + torchmetrics.
    Returns dict: {fid, is_mean, is_std}
    """
    import subprocess, sys, importlib
    import numpy as np

    def _install(pkg):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    # Try torch-fidelity first — gives FID and IS in one call.
    try:
        importlib.import_module("torch_fidelity")
    except ImportError:
        try:
            _install("torch-fidelity")
            importlib.invalidate_caches()
        except Exception as e:
            print(f"torch-fidelity install failed: {e}")

    USE_TF = False
    try:
        import torch_fidelity
        USE_TF = True
    except ImportError:
        pass

    if USE_TF:
        print("Using torch-fidelity (same tool as the official JiT repo)\n")
        metrics = torch_fidelity.calculate_metrics(
            input1=gen_dir,
            input2=real_dir,
            cuda=(device == "cuda"),
            isc=True,
            fid=True,
            kid=False,
            verbose=False,
            samples_find_deep=False,
        )
        return {
            "fid":     metrics["frechet_inception_distance"],
            "is_mean": metrics["inception_score_mean"],
            "is_std":  metrics["inception_score_std"],
        }

    print("Falling back to pytorch-fid + torchmetrics for IS\n")
    try:
        importlib.import_module("pytorch_fid")
    except ImportError:
        _install("pytorch-fid")
    try:
        importlib.import_module("torchmetrics")
    except ImportError:
        _install("torchmetrics[image]")

    import glob
    from PIL import Image
    from pytorch_fid import fid_score
    from torchmetrics.image.inception import InceptionScore

    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir], batch_size=256, device=device, dims=2048,
    )

    inception = InceptionScore(normalize=True).to(device)
    paths = sorted(glob.glob(f"{gen_dir}/*.png"))
    bsz = 100
    for i in range(0, len(paths), bsz):
        batch = []
        for p in paths[i:i + bsz]:
            img = torch.from_numpy(
                np.array(Image.open(p).convert("RGB"))
            ).permute(2, 0, 1).float() / 255.0
            batch.append(img)
        batch = torch.stack(batch).to(device)
        inception.update(batch)
    is_mean, is_std = inception.compute()

    return {
        "fid":     fid_value,
        "is_mean": float(is_mean),
        "is_std":  float(is_std),
    }


# ── Throughput measurement (from notebooks Cell 40) ──────────────────────────

@torch.no_grad()
def measure_throughput(model, input_shape, device, num_classes=10, n_warmup=5, n_iters=50):
    """
    Measure inference throughput (images/sec).
    model must have signature forward(x, t, y).
    """
    model.eval()
    B, C, H, W = input_shape
    x = torch.randn(B, C, H, W, device=device)
    t = torch.rand(B, device=device)
    y = torch.randint(0, num_classes, (B,), device=device)

    for _ in range(n_warmup):
        model(x, t, y)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iters):
        model(x, t, y)
    if device == "cuda":
        torch.cuda.synchronize()

    return (n_iters * B) / (time.time() - start)


# ── Metric logger ─────────────────────────────────────────────────────────────

class MetricLogger:
    """Simple in-memory metric logger that saves to JSON."""

    def __init__(self):
        self.history = []

    def log(self, step: int, **kwargs):
        self.history.append({"step": step, **kwargs})

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Metrics saved → {path}")
