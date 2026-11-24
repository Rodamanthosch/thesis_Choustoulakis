"""
scripts/measure_vmamba_global.py
================================
Diagnostic for the global condition tokens of JiT-S2-VMamba-Global
(model: vmamba_global; src/models/vmamba_global.py). Inference only.

Images: n per class (default 4, every class) from the TRAINING folder (the
only Tiny-ImageNet split whose ImageFolder labels are the classes; val/ is one
images/ folder), centre-cropped as in training, noised as the denoiser does,
z_t = t x + (1 - t) e, on a t grid (default linspace(0, 1, 11)), fixed noise
seed. Each image is run with its own class, and one image per class is also
run with the null (CFG) class. A forward pre-hook on every SS2DGlobal mixer
recomputes its terms through the mixer's own SS2DGlobal.terms (no duplicated
maths), per direction k:

  r_xe        ||alpha_xe * read||_F / ||scan||_F, over (d_inner, L): how much the
              image <- token term adds to the scan (scan = kernel output minus
              its D u term, pre-merge; under softmax, read is the softmax term)
  erank_S     effective rank of S_k (d_inner x N), exp(entropy(sigma / sum sigma))
              (Roy & Vetterli); erank_aS the same for alpha_ee * S_k
  g_over_s    ||G_k||_F / ||S_k||_F;  ag_over_as  ||alpha_ex G_k|| / ||alpha_ee S_k||
              (NaN in write-only mode, and in the last block -- which has no G /
              alpha_ee -- unless --last-block-g recomputes G there)

Outputs in --out-dir:
  global_ratio.csv      block, k, t: each metric's mean / std / min / max over
                        the real classes (per-class mean over its images), and
                        the null class's value
  global_by_class.csv   block, k, class: every metric averaged over t
  global_svd.npz        t, blocks, the metric arrays (n_blocks, 4, T, Y) and the
                        mean singular values of S_k, (n_blocks, 4, T, Y, N)
  summary.json          per block, each metric's mean over k, t, real classes

Run (from the repo root):
  python scripts/measure_vmamba_global.py \
      --config configs/tiny_imagenet/jit-s2-vmamba-global-full.yaml \
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

METRICS = ("r_xe", "erank_S", "erank_aS", "g_over_s", "ag_over_as")


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

def _erank(S):
    """Effective rank of each (d_inner, N) matrix in S (..., d, N), float64."""
    sv = torch.linalg.svdvals(S.double())                              # (..., N)
    p = sv / sv.sum(-1, keepdim=True).clamp_min(1e-300)
    ent = -(p * torch.log(p.clamp_min(1e-300))).sum(-1)
    return torch.exp(ent), sv


def _norm(a):
    return a.double().flatten(2).norm(dim=-1)                          # (B, K)


def _ratio(num, den):
    return torch.where(den > 0, num / den.clamp_min(1e-300), torch.full_like(den, float("nan")))


@torch.no_grad()
def block_metrics(mixer, x, H, W, last_block_g=False):
    """Per-sample metrics of one SS2DGlobal call: {metric: (B, K)}, sv (B, K, N)."""
    t = mixer.terms(x, H, W)
    K = mixer.K
    M = t["u_e"].shape[-1]
    S = t["S"]
    if S is None:                       # softmax mode, last block: S feeds nothing
        S = torch.einsum("bkdm,bknm->bkdn", t["ve"], t["B_e"].to(t["ve"].dtype)) / M
    G = t["G"]
    if G is None and last_block_g and mixer.readback:
        ct = t["ve"].dtype
        vx = torch.nn.functional.softplus(
            t["dt"].to(ct) + mixer.dt_projs_bias.to(ct)[None, :, :, None]) * t["xs"].to(ct)
        G = torch.einsum("bkdl,bknl->bkdn", vx, t["B_x"].to(ct)) / t["xs"].shape[-1]
    scan = t["y_scan"].double() - mixer.Ds.double()[None, :, :, None] * t["xs"].double()
    a_xe = mixer.alpha_xe.double()[None, None, :, None]
    out = {"r_xe": _ratio(_norm(a_xe * t["read"].double()), _norm(scan))}
    out["erank_S"], sv = _erank(S)
    nan = torch.full((x.shape[0], K), float("nan"), dtype=torch.float64, device=x.device)
    has_ee = hasattr(mixer, "alpha_ee")
    a_ee = mixer.alpha_ee.double()[None, None, :, None] if has_ee else None
    out["erank_aS"] = _erank(a_ee * S)[0] if has_ee else nan
    out["g_over_s"] = _ratio(_norm(G), _norm(S)) if G is not None else nan
    if G is not None and has_ee and hasattr(mixer, "alpha_ex"):
        out["ag_over_as"] = _ratio(_norm(mixer.alpha_ex.double()[None, None, :, None] * G),
                                   _norm(a_ee * S))
    else:
        out["ag_over_as"] = nan
    return out, sv


@torch.no_grad()
def measure(net, images, t_steps=11, batch_size=256, seed=0, last_block_g=False):
    """images: (C, n, 3, H, W). Returns {"t", "blocks", metric: (nb, K, T, Y),
    "sv": (nb, K, T, Y, N)} with Y = C + 1 (last = null class), float64."""
    from src.models.vmamba_global import SS2DGlobal
    device = next(net.parameters()).device
    mixers = [(i, b.mixer) for i, b in enumerate(net.blocks) if isinstance(b.mixer, SS2DGlobal)]
    assert mixers, "this model has no SS2DGlobal mixer (global_len 0?)"
    C, n = images.shape[:2]
    assert C == net.num_classes, (C, net.num_classes)
    ts = torch.linspace(0, 1, t_steps)
    K, N, nb, T, Y = 4, net.blocks[mixers[0][0]].mixer.d_state, len(mixers), t_steps, C + 1
    sums = {m: torch.zeros(nb, K, T, Y, dtype=torch.float64) for m in METRICS}
    counts = torch.zeros(T, Y, dtype=torch.float64)
    sv_sum = torch.zeros(nb, K, T, Y, N, dtype=torch.float64)

    # (image, label) jobs: every image with its class, one per class with null
    xs = images.reshape(C * n, *images.shape[2:])
    labels = torch.arange(C).repeat_interleave(n)
    xs = torch.cat([xs, images[:, 0]])
    labels = torch.cat([labels, torch.full((C,), C)])
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(xs.shape, generator=gen)

    current = {}

    def make_hook(j):
        def hook(mod, args):
            x, H, W = args[:3]
            if x.shape[1] == H * W:
                return
            res, sv = block_metrics(mod, x, H, W, last_block_g)
            current[j] = (res, sv)
        return hook

    hooks = [mx.register_forward_pre_hook(make_hook(j)) for j, (_, mx) in enumerate(mixers)]
    try:
        for ti, tv in enumerate(ts):
            for s in range(0, len(xs), batch_size):
                xb = xs[s:s + batch_size].to(device)
                eb = noise[s:s + batch_size].to(device)
                yb = labels[s:s + batch_size].to(device)
                z = tv * xb + (1 - tv) * eb
                current.clear()
                net(z, torch.full((len(xb),), float(tv), device=device), yb)
                yc = yb.cpu()
                for j in range(nb):
                    res, sv = current[j]
                    for m in METRICS:
                        v = res[m].cpu()                                   # (b, K)
                        sums[m][j, :, ti].index_add_(1, yc, v.T)
                    sv_sum[j, :, ti].index_add_(1, yc, sv.cpu().permute(1, 0, 2))
                counts[ti].index_add_(0, yc, torch.ones(len(yc), dtype=torch.float64))
    finally:
        for h in hooks:
            h.remove()
    out = {"t": ts.double().numpy(), "blocks": np.array([i for i, _ in mixers])}
    for m in METRICS:
        out[m] = (sums[m] / counts[None, None]).numpy()
    out["sv"] = (sv_sum / counts[None, None, :, :, None]).numpy()
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
        for k in range(4):
            for i, t in enumerate(ts):
                row = dict(block=int(blk), k=k, t=round(float(t), 4))
                for m in METRICS:
                    s = _stats(res[m][j, k, i, :n_real])
                    row.update({f"{m}_{q}": v for q, v in s.items()})
                    row[f"{m}_null"] = float(res[m][j, k, i, n_real])
                rows_ratio.append(row)
            for y in range(n_real + 1):
                row = dict(block=int(blk), k=k, cls="null" if y == n_real else y)
                row.update({f"{m}_mean_over_t": _stats(res[m][j, k, :, y])["mean"] for m in METRICS})
                rows_class.append(row)
    for m in METRICS:
        summary["metrics"][m] = {
            "mean_per_block": [_stats(res[m][j, :, :, :n_real])["mean"] for j in range(len(blocks))],
            "null_mean_per_block": [_stats(res[m][j, :, :, n_real])["mean"] for j in range(len(blocks))],
        }
    for name, rows in (("global_ratio.csv", rows_ratio), ("global_by_class.csv", rows_class)):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    np.savez(os.path.join(out_dir, "global_svd.npz"), **res)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary):
    blocks = summary["blocks"]
    print("\nper block, mean over directions, t and the real classes:")
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
    ap.add_argument("--last-block-g", action="store_true",
                    help="recompute G in the last block (it has none) for g_over_s")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import yaml
    from scripts.evaluate import build_model
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    assert cfg["experiment"]["model"] == "vmamba_global", "this diagnostic is for model=vmamba_global"
    net = build_model(cfg)
    meta = load_weights(net, args.checkpoint, args.ema)
    net = net.to(args.device).eval()
    meta.update(config=args.config, checkpoint=args.checkpoint, ema=args.ema,
                n_per_class=args.n_per_class, seed=args.seed)
    images = class_images(cfg, args.n_per_class)
    res = measure(net, images, args.t_steps, args.batch_size, args.seed, args.last_block_g)
    summary = write_reports(res, args.out_dir, net.num_classes, meta)
    print(f"{cfg['experiment']['name']}  epoch {meta.get('epoch')}  EMA {args.ema}")
    print_summary(summary)
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
