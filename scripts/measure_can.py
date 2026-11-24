"""
scripts/measure_can.py
======================
Diagnostic for the CAN-style conditional kernels of JiT-S2-Spatial-Mamba
(model.dwconv_can / model.sasf_can; src/models/spatial_mamba.py). Inference
only: it reads the generators and the static kernels, never runs the network.

The generators are shared by all blocks, dK = W SiLU(c) with c = t_emb(t) +
y_emb(y), and block l convolves with K_l + dK. dK is read on the training
density of t (t = sigmoid(z), z ~ N(P_mean, P_std) from the config's
`diffusion`, Gauss-Hermite nodes with weights w_t: logit_normal_nodes) times
the real classes (uniform; the null class is excluded).

Energy decomposition, per site (two-way ANOVA of the vector dK(t, y) with the
product weights w_t / n_classes, so the four parts are orthogonal):

    m    = E_t mean_y dK                       constant
    a_t  = mean_y dK(t, .) - m                 time effect
    b_y  = E_t dK(., y) - m                    class effect
    r    = dK - m - a_t - b_y                  interaction
    E_t mean_y ||dK||^2 = ||m||^2 + E_t ||a_t||^2 + mean_y ||b_y||^2 + E_t mean_y ||r||^2

reported as shares of the total (summing to 1). They map onto
`evaluate.py --can_freeze` on the same nodes: "all" keeps exactly m, "class"
(averages out y) keeps m + a_t, "time" (averages out t) keeps m + b_y.

Size, per block and site: sqrt(E_t mean_y ||dK||^2) / ||K_l|| (RMS of dK
against the block's own static kernel) and ||m|| / ||K_l|| (the constant part).

Sites: dwconv (the mixer DWConv), sasf_d<k> (one SASF dilation), sasf (all
dilations together). With zero (untrained) generators the total energy is 0
and the shares are NaN.

Outputs in --out-dir:
  can_energy.csv   site: total energy and the four shares
  can_size.csv     block, site: ||K_l||, RMS ratio, constant ratio
  summary.json     both, plus the run metadata

Run (from the repo root):
  python scripts/measure_can.py --config configs/tiny_imagenet/jit-s2-spatial-nogate-can.yaml \
      --checkpoint OUT/checkpoint-last.pt --ema 1 --out-dir OUT/can_diagnostic
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

def _sites(net):
    """{site: (slice of the generator output per row, [K_l flattened per block])}."""
    sites = {}
    if net.dwconv_can:
        sites["dwconv"] = ("can_dwconv", None,
                           [b.mixer.conv2d.weight[:, 0].double().flatten() for b in net.blocks])
    if net.sasf_can:
        dil = net.blocks[0].mixer.state_fusion.dilations
        for j, d in enumerate(dil):
            sites[f"sasf_d{d}"] = ("can_sasf", j, [b.mixer.state_fusion.kernels[j][:, 0].double().flatten()
                                                  for b in net.blocks])
        sites["sasf"] = ("can_sasf", None,
                         [torch.stack([k[:, 0] for k in b.mixer.state_fusion.kernels]).double().flatten()
                          for b in net.blocks])
    return sites


@torch.no_grad()
def measure(net, P_mean, P_std, n_nodes=64):
    """Energy decomposition and size ratios for a JiTSpatialMamba with CAN.

    Streams over the quadrature nodes, so memory is one (n_classes, |dK|) slab
    per site. Returns {"t": nodes, "w": weights, "sites": {site: {"energy",
    "shares": {constant, time, class, interaction}, "size": [per block]}}}.
    """
    from src.models.spatial_mamba import logit_normal_nodes
    assert net.dwconv_can or net.sasf_can, "this model has no CAN generator"
    device = net.pos_embed.device
    ts, ws = logit_normal_nodes(P_mean, P_std, n_nodes)
    nc = net.num_classes
    y_emb = net.y_embedder(torch.arange(nc, device=device))              # null excluded
    sites = _sites(net)

    acc = {s: {"E": 0.0, "row_means": [], "class_sum": 0.0} for s in sites}
    for t_i, w_i in zip(ts.tolist(), ws.tolist()):
        c = net.t_embedder(torch.full((nc,), t_i, device=device)) + y_emb
        gens = {name: getattr(net, name)(c).double() for name in {v[0] for v in sites.values()}}
        for s, (name, j, _) in sites.items():
            g = gens[name]
            d = (g[:, j] if j is not None else g).reshape(nc, -1)          # (nc, P)
            a = acc[s]
            a["E"] += w_i * d.pow(2).sum(1).mean().item()
            a["row_means"].append(d.mean(0))                               # mean_y at t_i
            a["class_sum"] = a["class_sum"] + w_i * d                      # E_t per class

    out = {"t": ts.tolist(), "w": ws.tolist(), "sites": {}}
    for s, (_, _, statics) in sites.items():
        a = acc[s]
        rows = torch.stack(a["row_means"])                                 # (T, P)
        m = (ws.to(rows) [:, None] * rows).sum(0)
        e_const = m.pow(2).sum().item()
        e_time = (ws.to(rows) * (rows - m).pow(2).sum(1)).sum().item()
        e_class = (a["class_sum"] - m).pow(2).sum(1).mean().item()
        E = a["E"]
        e_inter = E - e_const - e_time - e_class
        shares = ({k: v / E for k, v in (("constant", e_const), ("time", e_time),
                                         ("class", e_class), ("interaction", e_inter))}
                  if E > 0 else {k: float("nan") for k in ("constant", "time", "class", "interaction")})
        out["sites"][s] = {
            "energy": E,
            "shares": shares,
            "size": [{"block": l, "K_norm": K.norm().item(),
                      "rms_ratio": math.sqrt(E) / K.norm().item(),
                      "const_ratio": math.sqrt(e_const) / K.norm().item()}
                     for l, K in enumerate(statics)],
        }
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def write_reports(res, out_dir, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    rows_e, rows_s = [], []
    for s, a in res["sites"].items():
        rows_e.append(dict(site=s, n_nodes=len(res["t"]), energy=a["energy"],
                           **{f"share_{k}": v for k, v in a["shares"].items()}))
        rows_s += [dict(site=s, **r) for r in a["size"]]
    for name, rows in (("can_energy.csv", rows_e), ("can_size.csv", rows_s)):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary = {"meta": meta or {}, "t_nodes": res["t"], "t_weights": res["w"],
               "sites": res["sites"]}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(res):
    print(f"\nenergy shares (training t density x real classes; null excluded)")
    print(f"{'site':<10} {'constant':>9} {'time':>9} {'class':>9} {'interact':>9}   "
          f"{'RMS dK/K':>9} {'|m|/K':>7}")
    for s, a in res["sites"].items():
        sh, sz = a["shares"], a["size"]
        rms = sum(r["rms_ratio"] for r in sz) / len(sz)
        cst = sum(r["const_ratio"] for r in sz) / len(sz)
        print(f"{s:<10} {sh['constant']:>9.4f} {sh['time']:>9.4f} {sh['class']:>9.4f} "
              f"{sh['interaction']:>9.4f}   {rms:>9.3f} {cst:>7.3f}")
    print("(RMS dK/K and |m|/K: mean over blocks; per block in can_size.csv)")
    print("\nper block, RMS ||dK|| / ||K_l||:")
    for s, a in res["sites"].items():
        print(f"  {s:<10} " + " ".join(f"{r['rms_ratio']:.3f}" for r in a["size"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help=".pt file or a Kaggle-unpacked checkpoint folder")
    ap.add_argument("--ema", type=int, default=1, choices=[0, 1, 2], help="0 = raw weights")
    ap.add_argument("--t-nodes", type=int, default=64,
                    help="Gauss-Hermite nodes on the training t density (as --can_freeze)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import yaml
    from scripts.evaluate import build_model
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    net = build_model(cfg)
    meta = load_weights(net, args.checkpoint, args.ema)
    net = net.to(args.device).eval()
    dif = cfg["diffusion"]
    meta.update(config=args.config, checkpoint=args.checkpoint, ema=args.ema,
                P_mean=dif["P_mean"], P_std=dif["P_std"])
    res = measure(net, dif["P_mean"], dif["P_std"], args.t_nodes)
    write_reports(res, args.out_dir, meta)
    print(f"{cfg['experiment']['name']}  epoch {meta.get('epoch')}  EMA {args.ema}")
    print_summary(res)
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
