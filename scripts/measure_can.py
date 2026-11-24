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

Time component, per site, on the same nodes: T(t) = a_t = mean_y dK(t, .) - m
(so E_t ||T||^2 is the time energy above), against the block's effective mean
kernel M_l = K_l + m (what --can_freeze all convolves with):

  1. effective rank: singular values of the (n_t, P) matrix sqrt(w_t) T(t);
     their energy fractions s_k^2 / sum s^2 and how many reach 90 / 99 %.
  2. alpha-like share, per block: each channel's 9 taps of T(t) projected on
     that channel's M_l, ||proj||^2 / ||T||^2 ("channel": a free scale per
     channel and t), and on the whole M_l of the site ("one": ONE signed scale
     a_l(t) = <T(t), M_l> / ||M_l||^2 for the site, i.e. for one SASF dilation
     the kernel turns into M_l (1 + a_l(t)) -- a time-dependent alpha). Shares
     averaged over t are ratios of E_t energies (the per-t ratios are in the CSV).
  3. centre vs ring: share of ||T(t)||^2 on the centre tap vs the 8 outer
     taps (block independent), next to the same share of M_l for reference.
  4. ||T(t)|| / ||M_l|| per block and t: where along the noise level (t = 0
     noise, 1 data) the kernels change most; RMS over t = sqrt(time share) *
     RMS ||dK|| / ||M_l||.

The sampler visits t on a uniform grid, not the training density: the per-t
tables are the ones that say what changes along a sampling trajectory.

Sites: dwconv (the mixer DWConv), sasf_d<k> (one SASF dilation), sasf (all
dilations together). With zero (untrained) generators the total energy is 0
and the shares are NaN.

Outputs in --out-dir:
  can_energy.csv        site: total energy and the four shares
  can_size.csv          block, site: ||K_l||, RMS ratio, constant ratio
  can_time_rank.csv     site, k: singular value, energy fraction, cumulative
  can_time_alpha.csv    site, block, t: scale a_l(t), share one / channel
  can_time_centre.csv   site, t: centre / ring share of ||T(t)||^2
  can_time_ratio.csv    site, block, t: ||T(t)||, ||M_l||, ratio
  summary.json          all of the above but the per-t rows, plus the metadata

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


def _nan_div(a, b):
    return a / b if b > 0 else float("nan")


def analyse_time(T, ws, statics, taps):
    """The four analyses of the time component (module docstring).

    T: (n_t, P) float64, T(t) = mean_y dK(t, .) - m, rows weighted by ws (sum 1).
    statics: per block, the effective mean kernel M_l = K_l + m flattened (P,).
    taps: kernel taps per channel (9 for 3x3); the centre is taps // 2.
    Aggregates over t are ratios of E_t energies; "per_t" entries are lists
    over the nodes (kept out of summary.json).
    """
    w = ws.to(T)
    n_t = T.shape[0]
    T3 = T.view(n_t, -1, taps)                                  # (n_t, C, taps)
    tt = T3.pow(2).sum((1, 2))                                  # ||T(t)||^2
    e_time = (w * tt).sum().item()
    per_t = lambda v: [_nan_div(a, b) for a, b in zip(v.tolist(), tt.tolist())]

    # 1. effective rank
    s = torch.linalg.svdvals(w.sqrt()[:, None] * T)
    e_s = s.pow(2).sum().item()
    frac = (s.pow(2) / e_s) if e_s > 0 else torch.full_like(s, float("nan"))
    cum = frac.cumsum(0)

    def reach(p):            # fewest directions holding a fraction p of the energy
        if not e_s > 0:
            return None
        return min(int((cum < p - 1e-12).sum().item()) + 1, len(s))

    rank = {"sv": s.tolist(), "frac": frac.tolist(), "cum": cum.tolist(),
            "n90": reach(0.90), "n99": reach(0.99)}

    # 3. centre vs ring (block independent)
    ct = T3[:, :, taps // 2].pow(2).sum(1)
    centre = {"share": _nan_div((w * ct).sum().item(), e_time), "per_t": per_t(ct)}

    # 2. alpha-like projections and 4. size, per block
    blocks = []
    for l, M in enumerate(statics):
        M3 = M.view(-1, taps)
        mn2 = M3.pow(2).sum(1)                                  # (C,)
        mm = mn2.sum().item()
        dot = (T3 * M3).sum(-1)                                 # (n_t, C)
        p_ch = torch.where(mn2 > 0, dot.pow(2) / mn2.clamp_min(1e-300), torch.zeros_like(dot)).sum(1)
        scale = dot.sum(1) / mm if mm > 0 else torch.full_like(tt, float("nan"))
        p_one = scale.pow(2) * mm
        blocks.append({
            "block": l, "M_norm": math.sqrt(mm),
            "M_centre_share": _nan_div(M3[:, taps // 2].pow(2).sum().item(), mm),
            "share_channel": _nan_div((w * p_ch).sum().item(), e_time),
            "share_one": _nan_div((w * p_one).sum().item(), e_time),
            "rms_ratio": _nan_div(math.sqrt(e_time), math.sqrt(mm)),
            "per_t": {"scale": scale.tolist(), "share_one": per_t(p_one),
                      "share_channel": per_t(p_ch),
                      "ratio": [_nan_div(math.sqrt(v), math.sqrt(mm)) for v in tt.tolist()]},
        })
    return {"energy": e_time, "T_norm": tt.sqrt().tolist(), "rank": rank,
            "centre": centre, "blocks": blocks}


@torch.no_grad()
def measure(net, P_mean, P_std, n_nodes=64):
    """Energy decomposition, size ratios and the time-component analyses for a
    JiTSpatialMamba with CAN.

    Streams over the quadrature nodes, so memory is one (n_classes, |dK|) slab
    per site. Returns {"t": nodes, "w": weights, "P_mean", "P_std", "sites":
    {site: {"energy", "shares": {constant, time, class, interaction},
    "size": [per block], "time": analyse_time(...)}}}.
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

    taps = net.blocks[0].mixer.conv2d.weight[0, 0].numel() if net.dwconv_can else 9
    out = {"t": ts.tolist(), "w": ws.tolist(), "P_mean": P_mean, "P_std": P_std, "sites": {}}
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
            "time": analyse_time(rows - m, ws, [K + m for K in statics],
                                 taps if s == "dwconv" else 9),     # SASF kernels are 3x3
        }
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def t_cdf(t, P_mean, P_std):
    """Fraction of the training t density below t."""
    z = math.log(t / (1 - t))
    return 0.5 * (1 + math.erf((z - P_mean) / (P_std * math.sqrt(2))))


def _drop_per_t(d):
    if isinstance(d, dict):
        return {k: _drop_per_t(v) for k, v in d.items() if k not in ("per_t", "T_norm")}
    if isinstance(d, list):
        return [_drop_per_t(v) for v in d]
    return d


def write_reports(res, out_dir, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    rows_e, rows_s, rows_r, rows_a, rows_c, rows_q = [], [], [], [], [], []
    for s, a in res["sites"].items():
        rows_e.append(dict(site=s, n_nodes=len(res["t"]), energy=a["energy"],
                           **{f"share_{k}": v for k, v in a["shares"].items()}))
        rows_s += [dict(site=s, **r) for r in a["size"]]
        tm = a["time"]
        rk = tm["rank"]
        rows_r += [dict(site=s, k=k + 1, singular_value=v, energy_frac=f, cum_frac=c)
                   for k, (v, f, c) in enumerate(zip(rk["sv"], rk["frac"], rk["cum"]))]
        for i, (t, w) in enumerate(zip(res["t"], res["w"])):
            rows_c.append(dict(site=s, t=t, weight=w, T_norm=tm["T_norm"][i],
                               centre_share=tm["centre"]["per_t"][i],
                               ring_share=1 - tm["centre"]["per_t"][i]))
            for b in tm["blocks"]:
                p = b["per_t"]
                rows_a.append(dict(site=s, block=b["block"], t=t, weight=w, scale=p["scale"][i],
                                   share_one=p["share_one"][i], share_channel=p["share_channel"][i]))
                rows_q.append(dict(site=s, block=b["block"], t=t, weight=w, T_norm=tm["T_norm"][i],
                                   M_norm=b["M_norm"], ratio=p["ratio"][i]))
    for name, rows in (("can_energy.csv", rows_e), ("can_size.csv", rows_s),
                       ("can_time_rank.csv", rows_r), ("can_time_alpha.csv", rows_a),
                       ("can_time_centre.csv", rows_c), ("can_time_ratio.csv", rows_q)):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary = {"meta": meta or {}, "t_nodes": res["t"], "t_weights": res["w"],
               "sites": _drop_per_t(res["sites"])}
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
    print_time(res)


PROFILE_T = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)


def _mean(v):
    return sum(v) / len(v)


def print_time(res):
    sites = res["sites"]
    print("\ntime component T(t) = mean_y dK(t, .) - E_t mean_y dK, against M_l = K_l + E dK")
    print(f"{'site':<10} {'n90':>4} {'n99':>4} {'s1^2':>6} {'s2^2':>6} {'s3^2':>6}   "
          f"{'proj/ch':>7} {'proj/one':>8}   {'centre':>6} {'(M_l)':>6}   {'RMS T/M':>7}")
    for s, a in sites.items():
        tm = a["time"]
        f = (tm["rank"]["frac"] + [float("nan")] * 3)[:3]
        bl = tm["blocks"]
        print(f"{s:<10} {str(tm['rank']['n90']):>4} {str(tm['rank']['n99']):>4} "
              f"{f[0]:>6.3f} {f[1]:>6.3f} {f[2]:>6.3f}   "
              f"{_mean([b['share_channel'] for b in bl]):>7.3f} {_mean([b['share_one'] for b in bl]):>8.3f}   "
              f"{tm['centre']['share']:>6.3f} {_mean([b['M_centre_share'] for b in bl]):>6.3f}   "
              f"{_mean([b['rms_ratio'] for b in bl]):>7.3f}")
    print("(n90 / n99: directions holding 90 / 99 % of E_t ||T||^2; s_k^2: their fractions. "
          "proj: share of ||T||^2 along M_l, a free scale per channel / one scale per site. "
          "centre: share of ||T||^2 on the centre tap, (M_l) the same for M_l; 1/9 = 0.111 is "
          "uniform. Means over blocks; per block in can_time_*.csv)")

    # t profile at the nodes nearest a few t values
    ts = res["t"]
    logit = [math.log(t / (1 - t)) for t in ts]
    picks = sorted({min(range(len(ts)), key=lambda i: abs(logit[i] - math.log(p / (1 - p))))
                    for p in PROFILE_T})
    names = list(sites)
    width = max(10, max(len(n) for n in names) + 1)
    print("\nt profiles at the nodes nearest a few t  [t: 0 = noise, 1 = data; "
          "cdf: training density below t]")
    for title, fmt, fn in (
            ("||T(t)|| / ||M_l||  (mean over blocks)", ".4f",
             lambda tm, i: _mean([b["per_t"]["ratio"][i] for b in tm["blocks"]])),
            ("centre-tap share of ||T(t)||^2", ".4f",
             lambda tm, i: tm["centre"]["per_t"][i]),
            ("signed scale a_l(t) = <T(t), M_l> / ||M_l||^2  (mean over blocks)", ".2e",
             lambda tm, i: _mean([b["per_t"]["scale"][i] for b in tm["blocks"]])),
            ("share of ||T(t)||^2 along M_l, one scale  (mean over blocks)", ".4f",
             lambda tm, i: _mean([b["per_t"]["share_one"][i] for b in tm["blocks"]]))):
        print(f"\n{title}")
        print(f"{'t':>6} {'cdf':>5} " + "".join(f"{n:>{width}}" for n in names))
        for i in picks:
            print(f"{ts[i]:>6.3f} {t_cdf(ts[i], res['P_mean'], res['P_std']):>5.2f} "
                  + "".join(f"{fn(sites[n]['time'], i):>{width}{fmt}}" for n in names))


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
