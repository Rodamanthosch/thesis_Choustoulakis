"""
scripts/measure_spatial_global.py
=================================
Diagnostic for the global condition tokens of JiT-S2-Spatial-Global
(model: spatial_global; src/models/spatial_global.py). Inference only.

Images: n per class (default 4, every class) from the TRAINING folder (the
only Tiny-ImageNet split whose ImageFolder labels are the classes; val/ is one
images/ folder), centre-cropped as in training, noised as the denoiser does,
z_t = t x + (1 - t) e, on a t grid (default linspace(0, 1, 11)), fixed noise
seed. Each image is run with its own class, and one image per class is also
run with the null (CFG) class. A forward pre-hook on every SpatialGlobalSSM
recomputes its terms through the mixer's own SpatialGlobalSSM.terms (no
duplicated maths). Per block, t and class:

  r_xe        ||C * SASF(alpha_xe S)|| / ||C * SASF(states)||, over (d_inner, L):
              the image's read of the token memory next to its read of the
              scan states (both after the fusion, before the D term)
  g_over_s    ||G|| / ||S|| (NaN in write-only mode and in the last block,
              which has no G)
  s_norm      ||S||
  class_frac  ||S(t, y) - mean_y' S(t, y')|| / ||S(t, y)||, S(t, y) the mean
              memory of class y's images at t (mean over the real classes):
              how much of the memory depends on the class at fixed t
  time_frac   ||S(t, y) - mean_t' S(t', y)|| / ||S(t, y)||: the same over t
At d_state 1 S is one value per channel (rank 1), so no rank is reported.

Outputs in --out-dir:
  global_ratio.csv      block, t: each metric's mean / std / min / max over the
                        real classes, and the null class's value
  global_by_class.csv   block, class: every metric averaged over t
  global_diagnostic.npz t, blocks, the metric arrays (n_blocks, T, Y) and the
                        mean memories S(t, y), (n_blocks, T, Y, d_inner)
  summary.json          per block, each metric's mean over t and the real classes

Run (from the repo root):
  python scripts/measure_spatial_global.py \
      --config configs/tiny_imagenet/jit-s2-spatial-nogate-global-full.yaml \
      --checkpoint OUT/checkpoint-last.pt --ema 1 --out-dir OUT/global_diagnostic
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

PER_SAMPLE = ("r_xe", "g_over_s", "s_norm")
METRICS = PER_SAMPLE + ("class_frac", "time_frac")


# ── loading (self-contained: strict model + EMA, unpacked Kaggle folders) ────

def repack_checkpoint(folder):
    """Zip a checkpoint Kaggle unpacked into a folder (data.pkl, data/, ...) back
    into a loadable file in the temp directory."""
    digest = hashlib.sha256(str(folder).encode("utf8")).hexdigest()[:16]
    target = Path(tempfile.gettempdir()) / "unpacked_checkpoints" / f"{digest}.pt"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for directory, _, names in os.walk(folder):
                for name in sorted(names):
                    full = os.path.join(directory, name)
                    z.write(full, "checkpoint/" + os.path.relpath(full, folder).replace(os.sep, "/"))
        partial.replace(target)
    return str(target)


def load_weights(net, checkpoint, ema):
    """Strict model load, then (ema 1 / 2) the EMA copy by parameter name."""
    if os.path.isdir(checkpoint):
        checkpoint = repack_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model"], strict=True)
    if ema:
        state = payload.get(f"ema{ema}")
        names = {"net." + name for name, _ in net.named_parameters()}
        if state is None or set(state) != names:
            raise ValueError(f"EMA{ema} is absent or its parameter keys do not match this model")
        with torch.no_grad():
            for name, p in net.named_parameters():
                p.copy_(state["net." + name])
    return {k: payload.get(k) for k in ("epoch", "global_step")}


def class_images(cfg, n_per_class):
    """(C, n, 3, H, W) in [-1, 1]: the first n training images of every class."""
    import torchvision
    import torchvision.transforms as T
    from src.utils import CenterCropArr
    root = os.path.join(cfg["experiment"].get("data_dir", "./data"), "train")
    ds = torchvision.datasets.ImageFolder(root, transform=T.Compose([
        CenterCropArr(cfg["model"]["input_size"]), T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3)]))
    n_cls = cfg["model"]["num_classes"]
    assert len(ds.classes) == n_cls, (len(ds.classes), n_cls)
    picks = [[] for _ in range(n_cls)]
    for i, y in enumerate(ds.targets):
        if len(picks[y]) < n_per_class:
            picks[y].append(i)
    return torch.stack([torch.stack([ds[i][0] for i in idx]) for idx in picks])


# ── measurement ──────────────────────────────────────────────────────────────

def _ratio(num, den):
    return torch.where(den > 0, num / den.clamp_min(1e-300), torch.full_like(den, float("nan")))


@torch.no_grad()
def block_metrics(mixer, x, H, W):
    """Per-sample metrics of one SpatialGlobalSSM call: {metric: (B,)}, S (B, d)."""
    t = mixer.terms(x, H, W)
    B, d = x.shape[0], mixer.d_inner
    sf = mixer.state_fusion
    C = t["C_x"].double()                                               # (B, 1, L)
    anchor_map = t["anchor"].double().expand(B, d, H * W).reshape(B, d, H, W)
    read = (sf(anchor_map).reshape(B, d, -1) * C).flatten(1).norm(dim=-1)
    scan = (sf(t["states"].double().view(B, d, H, W)).reshape(B, d, -1) * C).flatten(1).norm(dim=-1)
    S = t["S"].double()[..., 0]                                         # (B, d)
    s_norm = S.norm(dim=-1)
    if t["G"] is not None:
        g_over_s = _ratio(t["G"].double()[..., 0].norm(dim=-1), s_norm)
    else:
        g_over_s = torch.full_like(s_norm, float("nan"))
    return {"r_xe": _ratio(read, scan), "g_over_s": g_over_s, "s_norm": s_norm}, S


def _fraction(a, centre):
    """||a - centre|| / ||a|| over the last dim; NaN where a = 0."""
    num, den = (a - centre).norm(dim=-1), a.norm(dim=-1)
    return torch.where(den > 0, num / den.clamp_min(1e-300), torch.full_like(den, float("nan")))


@torch.no_grad()
def measure(net, images, t_steps=11, batch_size=256, seed=0):
    """images: (C, n, 3, H, W). Returns {"t", "blocks", metric: (nb, T, Y),
    "S_mean": (nb, T, Y, d_inner)} with Y = C + 1 (last = null class), float64."""
    from src.models.spatial_global import SpatialGlobalSSM
    device = next(net.parameters()).device
    mixers = [(i, b.mixer) for i, b in enumerate(net.blocks) if isinstance(b.mixer, SpatialGlobalSSM)]
    assert mixers, "this model has no SpatialGlobalSSM mixer (global_len 0?)"
    C, n = images.shape[:2]
    assert C == net.num_classes, (C, net.num_classes)
    ts = torch.linspace(0, 1, t_steps)
    nb, T, Y, d = len(mixers), t_steps, C + 1, mixers[0][1].d_inner
    sums = {m: torch.zeros(nb, T, Y, dtype=torch.float64) for m in PER_SAMPLE}
    s_sum = torch.zeros(nb, T, Y, d, dtype=torch.float64)
    counts = torch.zeros(T, Y, dtype=torch.float64)

    xs = images.reshape(C * n, *images.shape[2:])
    labels = torch.arange(C).repeat_interleave(n)
    xs = torch.cat([xs, images[:, 0]])                  # one image per class, null label
    labels = torch.cat([labels, torch.full((C,), C)])
    noise = torch.randn(xs.shape, generator=torch.Generator().manual_seed(seed))

    current = {}

    def make_hook(j):
        def hook(mod, args):
            x, H, W = args[:3]
            if x.shape[1] != H * W:
                current[j] = block_metrics(mod, x, H, W)
        return hook

    hooks = [mx.register_forward_pre_hook(make_hook(j)) for j, (_, mx) in enumerate(mixers)]
    try:
        for ti, tv in enumerate(ts):
            for s in range(0, len(xs), batch_size):
                xb = xs[s:s + batch_size].to(device)
                eb = noise[s:s + batch_size].to(device)
                yb = labels[s:s + batch_size].to(device)
                current.clear()
                net(tv * xb + (1 - tv) * eb, torch.full((len(xb),), float(tv), device=device), yb)
                yc = yb.cpu()
                for j in range(nb):
                    res, S = current[j]
                    for m in PER_SAMPLE:
                        sums[m][j, ti].index_add_(0, yc, res[m].cpu())
                    s_sum[j, ti].index_add_(0, yc, S.cpu())
                counts[ti].index_add_(0, yc, torch.ones(len(yc), dtype=torch.float64))
    finally:
        for h in hooks:
            h.remove()
    out = {"t": ts.double().numpy(), "blocks": np.array([i for i, _ in mixers])}
    for m in PER_SAMPLE:
        out[m] = (sums[m] / counts[None]).numpy()
    S_mean = s_sum / counts[None, :, :, None]                          # (nb, T, Y, d)
    out["class_frac"] = _fraction(S_mean, S_mean[:, :, :C].mean(2, keepdim=True)).numpy()
    out["time_frac"] = _fraction(S_mean, S_mean.mean(1, keepdim=True)).numpy()
    out["S_mean"] = S_mean.numpy()
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).any():
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)),
            "min": float(np.nanmin(a)), "max": float(np.nanmax(a))}


def write_reports(res, out_dir, n_real, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    ts, blocks = res["t"], res["blocks"]
    rows_ratio, rows_class = [], []
    summary = {"meta": meta or {}, "blocks": [int(b) for b in blocks], "metrics": {}}
    for j, blk in enumerate(blocks):
        for i, t in enumerate(ts):
            row = dict(block=int(blk), t=round(float(t), 4))
            for m in METRICS:
                row.update({f"{m}_{q}": v for q, v in _stats(res[m][j, i, :n_real]).items()})
                row[f"{m}_null"] = float(res[m][j, i, n_real])
            rows_ratio.append(row)
        for y in range(n_real + 1):
            row = dict(block=int(blk), cls="null" if y == n_real else y)
            row.update({f"{m}_mean_over_t": _stats(res[m][j, :, y])["mean"] for m in METRICS})
            rows_class.append(row)
    for m in METRICS:
        summary["metrics"][m] = {
            "mean_per_block": [_stats(res[m][j, :, :n_real])["mean"] for j in range(len(blocks))],
            "null_mean_per_block": [_stats(res[m][j, :, n_real])["mean"] for j in range(len(blocks))],
        }
    for name, rows in (("global_ratio.csv", rows_ratio), ("global_by_class.csv", rows_class)):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    np.savez(os.path.join(out_dir, "global_diagnostic.npz"), **res)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary):
    blocks = summary["blocks"]
    print("\nper block, mean over t and the real classes:")
    print(f"  {'metric':<11} " + " ".join(f"{'b' + str(b):>7}" for b in blocks))
    for m, s in summary["metrics"].items():
        print(f"  {m:<11} " + " ".join(f"{v:7.3f}" for v in s["mean_per_block"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help=".pt file or a Kaggle-unpacked checkpoint folder")
    ap.add_argument("--ema", type=int, default=1, choices=[0, 1, 2], help="0 = raw weights")
    ap.add_argument("--n-per-class", type=int, default=4)
    ap.add_argument("--t-steps", type=int, default=11)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import yaml
    from scripts.evaluate import build_model
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    assert cfg["experiment"]["model"] == "spatial_global", "this diagnostic is for model=spatial_global"
    net = build_model(cfg)
    meta = load_weights(net, args.checkpoint, args.ema)
    net = net.to(args.device).eval()
    meta.update(config=args.config, checkpoint=args.checkpoint, ema=args.ema,
                n_per_class=args.n_per_class, seed=args.seed)
    images = class_images(cfg, args.n_per_class)
    res = measure(net, images, args.t_steps, args.batch_size, args.seed)
    summary = write_reports(res, args.out_dir, net.num_classes, meta)
    print(f"{cfg['experiment']['name']}  epoch {meta.get('epoch')}  EMA {args.ema}")
    print_summary(summary)
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
