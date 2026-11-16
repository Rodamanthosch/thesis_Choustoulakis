"""
scripts/measure_can.py
======================
Diagnostic for the CAN-style conditional kernels of JiT-S2-Spatial-Mamba
(model.dwconv_can / model.sasf_can; src/models/spatial_mamba.py). Inference
only: it reads the generators and the static kernels, never runs the network.

The generators are shared by all blocks, dK = W SiLU(c) with c = t_emb(t) +
y_emb(y), and block l convolves with K_l + dK. Over a grid of t (default
linspace(0, 1, 11)) and every class plus the null (CFG) class:

  1. size, per block and site:      ||dK(t, y)||_F / ||K_l||_F
  2. class vs time, per site (dK is the same in every block):
       class fraction  ||dK(t, y) - mean_y' dK(t, y')||_F / ||dK(t, y)||_F
                       (the part of dK that changes with the class at fixed t;
                        the mean is over the real classes only)
       time fraction   ||dK(t, y) - mean_t' dK(t', y)||_F / ||dK(t, y)||_F
                       (the part that changes with t at fixed class)
     Near 0: dK is a constant offset along that axis. The fractions are not
     bounded by 1 (a dK pointing away from the mean gives > 1). The null class
     is reported separately: its class fraction measures how far the
     unconditional kernel sits from the mean class kernel.

Sites: dwconv (the mixer DWConv), sasf_d<k> (one SASF dilation), sasf (all
dilations together). With zero (untrained) generators every ratio is 0 and
the fractions are NaN.

Outputs in --out-dir:
  can_ratio.csv            block, site, t: ratio mean / std / min / max over
                           the real classes, and the null class's ratio
  can_split.csv            site, t: class and time fractions (mean / min / max
                           over the real classes, null separately)
  can_split_by_class.csv   site, class: both fractions averaged over t
  can_diagnostic.npz       the full arrays
  summary.json             headline numbers per site

Run (from the repo root):
  python scripts/measure_can.py --config configs/tiny_imagenet/jit-s2-spatial-nogate-can.yaml \
      --checkpoint OUT/checkpoint-last.pt --ema 1 --out-dir OUT/can_diagnostic
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


# ── measurement ──────────────────────────────────────────────────────────────

def _fraction(dk, centre):
    """||dk - centre|| / ||dk|| over the trailing kernel dims; NaN where dk = 0."""
    num = (dk - centre).flatten(2).norm(dim=-1)
    den = dk.flatten(2).norm(dim=-1)
    return torch.where(den > 0, num / den.clamp_min(1e-300), torch.full_like(den, float("nan")))


@torch.no_grad()
def measure(net, t_steps=11):
    """Arrays of the diagnostic for a JiTSpatialMamba with dwconv_can / sasf_can.

    Returns {"t": (T,), "sites": {site: {"ratio": (L, T, Y), "class_frac": (T, Y),
    "time_frac": (T, Y)}}} with Y = num_classes + 1 (last = null class), float64.
    """
    assert net.dwconv_can or net.sasf_can, "this model has no CAN generator"
    device = next(net.parameters()).device
    ts = torch.linspace(0, 1, t_steps, device=device)
    ys = torch.arange(net.num_classes + 1, device=device)            # last = null
    T, Y, n_real = len(ts), len(ys), net.num_classes
    t_rep = ts.repeat_interleave(Y)                                   # (T*Y,)
    y_rep = ys.repeat(T)
    c = (net.t_embedder(t_rep) + net.y_embedder(y_rep)).double()      # (T*Y, D)

    sites = {}                                   # name -> (dK (T, Y, ...), [K_l])
    if net.dwconv_can:
        dk = net.can_dwconv(c).view(T, Y, *net.can_dwconv.out_shape)
        sites["dwconv"] = (dk, [b.mixer.conv2d.weight[:, 0].double() for b in net.blocks])
    if net.sasf_can:
        dk = net.can_sasf(c).view(T, Y, *net.can_sasf.out_shape)     # (T, Y, n_dil, d, 3, 3)
        dil = net.blocks[0].mixer.state_fusion.dilations
        for j, d in enumerate(dil):
            sites[f"sasf_d{d}"] = (dk[:, :, j], [b.mixer.state_fusion.kernels[j][:, 0].double()
                                                for b in net.blocks])
        sites["sasf"] = (dk, [torch.stack([k[:, 0] for k in b.mixer.state_fusion.kernels]).double()
                              for b in net.blocks])

    out = {"t": ts.double().cpu().numpy(), "sites": {}}
    for name, (dk, statics) in sites.items():
        dk = dk.double()
        norms = dk.flatten(2).norm(dim=-1)                             # (T, Y)
        ratio = torch.stack([norms / K.norm() for K in statics])       # (L, T, Y)
        class_mean = dk[:, :n_real].mean(dim=1, keepdim=True)          # over real classes
        time_mean = dk.mean(dim=0, keepdim=True)                       # over t
        out["sites"][name] = {
            "ratio": ratio.cpu().numpy(),
            "class_frac": _fraction(dk, class_mean).cpu().numpy(),
            "time_frac": _fraction(dk, time_mean).cpu().numpy(),
        }
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def _stats(a):
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)),
            "min": float(np.nanmin(a)), "max": float(np.nanmax(a))} if np.isfinite(a).any() \
        else {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}


def write_reports(res, out_dir, n_real, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    ts = res["t"]
    rows_ratio, rows_split, rows_class, summary = [], [], [], {"meta": meta or {}, "sites": {}}
    for site, a in res["sites"].items():
        ratio, cf, tf = a["ratio"], a["class_frac"], a["time_frac"]
        for l in range(ratio.shape[0]):
            for i, t in enumerate(ts):
                s = _stats(ratio[l, i, :n_real])
                rows_ratio.append(dict(block=l, site=site, t=round(float(t), 4),
                                       **{f"ratio_{k}": v for k, v in s.items()},
                                       ratio_null=float(ratio[l, i, n_real])))
        for i, t in enumerate(ts):
            c, d = _stats(cf[i, :n_real]), _stats(tf[i, :n_real])
            rows_split.append(dict(site=site, t=round(float(t), 4),
                                   class_frac_mean=c["mean"], class_frac_min=c["min"],
                                   class_frac_max=c["max"], class_frac_null=float(cf[i, n_real]),
                                   time_frac_mean=d["mean"], time_frac_min=d["min"],
                                   time_frac_max=d["max"], time_frac_null=float(tf[i, n_real])))
        for y in range(n_real + 1):
            rows_class.append(dict(site=site, cls="null" if y == n_real else y,
                                   class_frac_mean_over_t=_stats(cf[:, y])["mean"],
                                   time_frac_mean_over_t=_stats(tf[:, y])["mean"]))
        summary["sites"][site] = {
            "ratio_mean": _stats(ratio[:, :, :n_real])["mean"],
            "ratio_max": _stats(ratio[:, :, :n_real])["max"],
            "ratio_mean_per_block": [_stats(ratio[l, :, :n_real])["mean"] for l in range(ratio.shape[0])],
            "class_frac_mean": _stats(cf[:, :n_real])["mean"],
            "class_frac_null_mean": _stats(cf[:, n_real])["mean"],
            "time_frac_mean": _stats(tf[:, :n_real])["mean"],
        }
    for name, rows in (("can_ratio.csv", rows_ratio), ("can_split.csv", rows_split),
                       ("can_split_by_class.csv", rows_class)):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    np.savez(os.path.join(out_dir, "can_diagnostic.npz"), t=ts,
             **{f"{s}__{k}": v for s, a in res["sites"].items() for k, v in a.items()})
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary):
    print(f"\n{'site':<10} {'||dK||/||K|| mean':>18} {'max':>8}   {'class frac':>10} "
          f"{'null vs mean':>12} {'time frac':>10}")
    for site, s in summary["sites"].items():
        print(f"{site:<10} {s['ratio_mean']:>18.4f} {s['ratio_max']:>8.4f}   "
              f"{s['class_frac_mean']:>10.4f} {s['class_frac_null_mean']:>12.4f} "
              f"{s['time_frac_mean']:>10.4f}")
    print("\nper block, mean ||dK|| / ||K_l|| over t and the real classes:")
    for site, s in summary["sites"].items():
        print(f"  {site:<10} " + " ".join(f"{v:.3f}" for v in s["ratio_mean_per_block"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help=".pt file or a Kaggle-unpacked checkpoint folder")
    ap.add_argument("--ema", type=int, default=1, choices=[0, 1, 2], help="0 = raw weights")
    ap.add_argument("--t-steps", type=int, default=11)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import yaml
    from scripts.evaluate import build_model
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    net = build_model(cfg)
    meta = load_weights(net, args.checkpoint, args.ema)
    net = net.to(args.device).eval()
    meta.update(config=args.config, checkpoint=args.checkpoint, ema=args.ema)
    res = measure(net, args.t_steps)
    summary = write_reports(res, args.out_dir, net.num_classes, meta)
    print(f"{cfg['experiment']['name']}  epoch {meta.get('epoch')}  EMA {args.ema}")
    print_summary(summary)
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
