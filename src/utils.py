"""
src/utils.py
============
Shared utilities extracted from the notebooks.

Checkpoint helpers: exact from all three notebooks (Cell 25 — identical in all).
LR scheduler:       exact from all three notebooks (Cell 23 — identical in all).
FID/IS evaluation:  exact from all three notebooks (Cell 38 — identical in all).
Config loading:     load YAML hyperparameter files from configs/.
Throughput:         from the throughput cell (Cell 40) in all notebooks.
ImageNet crop:      exact from LTH14/JiT util/crop.py (ADM center_crop_arr).
"""

import os
import time
import json
import yaml
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


# ── ImageNet crop (exact from LTH14/JiT util/crop.py) ─────────────────────────

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CenterCropArr:
    """center_crop_arr as a transform. JiT wraps it in T.Lambda(lambda ...); a
    class pickles, so it also works with spawn-based DataLoader workers."""
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, pil_image):
        return center_crop_arr(pil_image, self.image_size)

    def __repr__(self):
        return f"{self.__class__.__name__}(image_size={self.image_size})"


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


# ── Official ImageNet FID / IS (exact from LTH14/JiT engine_jit.py) ──────────
#
# JiT scores 50K samples against PRECOMPUTED statistics (mu, sigma) of the
# ADM-cropped ImageNet TRAIN set, with LTH14's torch-fidelity fork, whose only
# functional change is the `fid_statistics_file` argument (input2=None). Both
# the fork and the stats are pinned to a commit; the stats are sha256-checked.

JIT_TORCH_FIDELITY = ("git+https://github.com/LTH14/torch-fidelity.git"
                      "@bfe72b995473e4fee9351705d7b005f17442a4ca")
JIT_STATS_URL = ("https://raw.githubusercontent.com/LTH14/JiT/"
                 "cbc743a2ada5e9762697da2c83f8c4f8379e8c17/fid_stats/{}")
JIT_FID_STATS = {   # img_size -> (file, sha256)
    256: ("jit_in256_stats.npz",
          "412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c"),
    512: ("jit_in512_stats.npz",
          "cdd15b54f9d1f26a881fcdc920a3410fa7a934047d6c4f5b8395b87182a9ab42"),
}
FID_STATS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "fid_stats")


def uses_jit_fid(cfg) -> bool:
    """True when JiT's reference stats apply: ImageNet-1k at 256 or 512.
    Tiny-ImageNet (200 classes, 64 px), CIFAR-10 and other ImageFolder
    subsets keep the folder-based evaluate_fid_is."""
    m = cfg["model"]
    return (cfg["experiment"]["dataset"] == "imagenet"
            and m["num_classes"] == 1000 and m["input_size"] in JIT_FID_STATS)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jit_fid_stats_path(img_size: int, stats_dir: str = FID_STATS_DIR) -> str:
    """Path to JiT's reference stats for img_size, downloaded once (pinned
    commit) and sha256-verified on every call."""
    import urllib.request
    fname, sha = JIT_FID_STATS[img_size]
    path = os.path.join(stats_dir, fname)
    if not os.path.exists(path):
        os.makedirs(stats_dir, exist_ok=True)
        print(f"Downloading {fname} (LTH14/JiT reference stats) -> {stats_dir}")
        tmp = path + ".part"
        urllib.request.urlretrieve(JIT_STATS_URL.format(fname), tmp)
        os.replace(tmp, path)
    got = _sha256(path)
    if got != sha:
        raise RuntimeError(f"{path}: sha256 {got} != expected {sha}. "
                           f"Delete it to re-download.")
    return path


def _has_jit_torch_fidelity() -> bool:
    try:
        from torch_fidelity.defaults import DEFAULTS
    except ImportError:
        return False
    return "fid_statistics_file" in DEFAULTS


def ensure_jit_torch_fidelity():
    """Import LTH14's torch-fidelity fork, installing the pinned commit over a
    stock torch-fidelity if needed (--no-deps: never touches torch)."""
    import importlib, subprocess, sys
    if not _has_jit_torch_fidelity():
        print(f"Installing LTH14/torch-fidelity ({JIT_TORCH_FIDELITY})")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--force-reinstall", "--no-deps", JIT_TORCH_FIDELITY])
        for name in [n for n in sys.modules
                     if n == "torch_fidelity" or n.startswith("torch_fidelity.")]:
            del sys.modules[name]
        importlib.invalidate_caches()
        if not _has_jit_torch_fidelity():
            raise RuntimeError("torch-fidelity has no `fid_statistics_file`: the LTH14 "
                               f"fork is required. pip install '{JIT_TORCH_FIDELITY}'")
    import torch_fidelity
    return torch_fidelity


def evaluate_fid_is_jit(gen_dir: str, stats_path: str, device: str = "cuda"):
    """FID / IS of gen_dir against JiT's reference stats, with the exact
    calculate_metrics call of LTH14/JiT engine_jit.py evaluate(). No fallback:
    a different tool would give numbers that are not comparable."""
    torch_fidelity = ensure_jit_torch_fidelity()
    metrics = torch_fidelity.calculate_metrics(
        input1=gen_dir,
        input2=None,
        fid_statistics_file=stats_path,
        cuda=(device == "cuda"),
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    return {
        "fid":     metrics["frechet_inception_distance"],
        "is_mean": metrics["inception_score_mean"],
        "is_std":  metrics["inception_score_std"],
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
