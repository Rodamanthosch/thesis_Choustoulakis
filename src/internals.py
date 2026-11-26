"""Inference-only internal diagnostics for JiT-VMamba.

Measures, on trained models, the quantities of Figures 1-7 of "Registers Matter
for Pixel-Space Diffusion Transformers" (arXiv 2605.16147): token norms, patch
norms, total variation, PCA maps, linear probes and token/image influence maps.

SS2D has no attention, so `scan_contributions` is the attention analogue. Given a
block's own Delta, B and C, the selective scan is exactly linear in its input u:

    y_i[c] = D[c] u_i[c]
             + sum_{j<=i} sum_n C_i[n] prod_{q=j+1..i} exp(Delta_q[c] A[c,n])
                          Delta_j[c] B_j[n] u_j[c]

The influence of source j on output i is the L2 norm over channels of its term,
summed over the four paths after both i and j return to canonical order.
"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

FILE_PREFERENCE = ("checkpoint-best.pt", "checkpoint-last.pt")


class _Stop(Exception):
    pass


def repack_checkpoint(folder, cache=None):
    """Zip a checkpoint that Kaggle unpacked into a folder (data.pkl, data/, ...) back into a file."""
    digest = hashlib.sha256(str(folder).encode("utf8")).hexdigest()[:16]
    cache = Path(cache or tempfile.gettempdir()) / "unpacked_checkpoints"
    target = cache / f"{digest}.pt"
    if not target.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
            for directory, _, names in os.walk(folder):
                for name in sorted(names):
                    full = os.path.join(directory, name)
                    archive.write(full, "checkpoint/" + os.path.relpath(full, folder).replace(os.sep, "/"))
        partial.replace(target)
        print(f"Rebuilt unpacked checkpoint {folder} as {target}", flush=True)
    return target


def checkpoint_mismatch(path, shapes, mmap=True):
    """None if the file's model weights have exactly these names and shapes, else the reason."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)["model"]
    except Exception as error:
        return f"unreadable ({type(error).__name__}: {error})"
    missing, extra = sorted(set(shapes) - set(state)), sorted(set(state) - set(shapes))
    if missing or extra:
        return f"{len(missing)} missing keys {missing[:2]}, {len(extra)} unexpected keys {extra[:2]}"
    for key, shape in shapes.items():
        if tuple(state[key].shape) != shape:
            return f"{key} has shape {tuple(state[key].shape)}, the model needs {shape}"
    return None


def discover_checkpoint(net, root, hint="", cache=None):
    """The preferred checkpoint of the single run directory whose weights fit `net`."""
    shapes = {k: tuple(v.shape) for k, v in net.state_dict().items()}
    candidates = []   # (run directory, file name, location, is an unpacked folder)
    for directory, _, names in os.walk(root, followlinks=True):
        folder = Path(directory)
        candidates += [(folder, name, folder / name, False) for name in names if name.endswith(".pt")]
        if "data.pkl" in names:
            name = folder.name if folder.name.endswith(".pt") else folder.name + ".pt"
            candidates.append((folder.parent, name, folder, True))
    matches, rejected = {}, []
    for run, name, location, unpacked in sorted(candidates, key=lambda c: str(c[2])):
        if hint.lower() not in str(location).lower():
            continue
        path = repack_checkpoint(location, cache) if unpacked else location
        reason = checkpoint_mismatch(str(path), shapes, mmap=not unpacked)   # a rebuilt zip is not page-aligned
        if reason is None:
            matches.setdefault(run, []).append((name, path))
        else:
            rejected.append(f"{location}: {reason}")
    if len(matches) != 1:
        listing = "\n  ".join(str(p) for p in matches) or "(none)"
        details = "\n  ".join(rejected) or "(none)"
        raise ValueError(f"Need exactly one matching run under {root} (hint={hint!r}), found:\n  {listing}\n"
                         f"Rejected:\n  {details}\nPass that checkpoint's path instead of auto.")
    files = next(iter(matches.values()))
    ranked = sorted(files, key=lambda f: (FILE_PREFERENCE.index(f[0]) if f[0] in FILE_PREFERENCE else 9, f[0]))
    return str(ranked[0][1])


def tokens_at(net, block):
    """Number of in-context tokens present in block `block`'s output."""
    return net.in_context_len if net.in_context_len and block >= net.in_context_start else 0


@contextmanager
def _hooks(handles):
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _check_eval(net, blocks):
    if net.training:
        raise ValueError("Diagnostics are evaluation-only; call net.eval() first")
    blocks = sorted(set(blocks))
    if not blocks or blocks[0] < 0 or blocks[-1] >= len(net.blocks):
        raise ValueError(f"Invalid block indices: {blocks}")
    return blocks


@torch.inference_mode()
def block_features(net, z, t, y, blocks):
    """{block: (B, tokens_at(block) + HW, D)}; later blocks are not computed."""
    blocks = _check_eval(net, blocks)
    captured = {}

    def hook(index):
        def run(_module, _inputs, output):
            captured[index] = output.float()
            if index == blocks[-1]:
                raise _Stop
        return run

    with _hooks([net.blocks[i].register_forward_hook(hook(i)) for i in blocks]):
        try:
            net(z, t, y)
        except _Stop:
            pass
    return captured


@torch.inference_mode()
def mixer_inputs(net, z, t, y, blocks):
    """{block: modulated norm1(x)}, the exact tensor each SS2D receives."""
    blocks = _check_eval(net, blocks)
    captured = {}

    def hook(index):
        def run(_module, args):
            captured[index] = args[0].float()
            if index == blocks[-1]:
                raise _Stop
        return run

    with _hooks([net.blocks[i].mixer.register_forward_pre_hook(hook(i)) for i in blocks]):
        try:
            net(z, t, y)
        except _Stop:
            pass
    return captured


def noised(images, noise, t, noise_scale):
    """The training interpolant: t = 0 is pure noise, t = 1 the clean image."""
    return t * images + (1 - t) * noise * noise_scale


def total_variation(grid, H, W):
    """Per image: mean L2 distance between 4-neighbouring tokens. (B, HW, D) -> (B,)."""
    maps = grid.reshape(grid.shape[0], H, W, -1)
    vertical = (maps[:, 1:] - maps[:, :-1]).norm(dim=-1).flatten(1)
    horizontal = (maps[:, :, 1:] - maps[:, :, :-1]).norm(dim=-1).flatten(1)
    return torch.cat([vertical, horizontal], dim=1).mean(dim=1)


@torch.inference_mode()
def feature_statistics(net, images, labels, noise, timesteps, blocks, noise_scale):
    """Per-image norms and TV after every requested block and timestep.

    Returns norm [T, block, image, n + HW] with NaN where a block has no tokens
    yet, tv [T, block, image] and grid_norm [T, block, image].
    """
    H = W = net.grid_size
    n, HW = net.in_context_len, H * W
    shape = (len(timesteps), len(blocks), len(images))
    norm = np.full(shape + (n + HW,), np.nan, np.float32)
    tv, grid_norm = np.empty(shape, np.float32), np.empty(shape, np.float32)
    for ti, t in enumerate(timesteps):
        time = torch.full((len(images),), float(t), device=images.device)
        features = block_features(net, noised(images, noise, t, noise_scale), time, labels, blocks)
        for bi, block in enumerate(blocks):
            x = features[block]
            if not torch.isfinite(x).all():
                raise RuntimeError(f"Nonfinite features: t={t}, block={block}")
            present = tokens_at(net, block)
            norm[ti, bi, :, n - present:] = x.norm(dim=-1).cpu().numpy()
            grid = x[:, present:]
            tv[ti, bi] = total_variation(grid, H, W).cpu().numpy()
            grid_norm[ti, bi] = grid.norm(dim=-1).mean(dim=1).cpu().numpy()
    return dict(norm=norm, tv=tv, grid_norm=grid_norm)


def pca_rgb(grid, H, W):
    """Three leading components of one image's tokens as RGB in [0, 1]."""
    centred = grid.float() - grid.float().mean(dim=0, keepdim=True)
    if float(centred.abs().max()) == 0:
        return torch.zeros(H, W, 3)
    _, _, components = torch.linalg.svd(centred, full_matrices=False)
    projected = centred @ components[:3].T
    if projected.shape[1] < 3:
        projected = torch.cat(
            [projected, torch.zeros(projected.shape[0], 3 - projected.shape[1], device=projected.device)], dim=1)
    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    scaled = (projected - low) / (high - low).clamp_min(1e-8)
    return scaled.clamp(0, 1).reshape(H, W, 3)


def canonical_scan_orders(H, W, n, layout):
    """Long tensor (4, n + HW): scan step s of path k reads joint index order[k, s].

    Joint order is [n tokens, HW cells row-major], the layout SS2D returns.
    """
    if layout not in ("prefix", "row"):
        raise ValueError(f"Unsupported layout for scan_contributions: {layout}")
    row_major = [n + r * W + c for r in range(H) for c in range(W)]
    col_major = [n + r * W + c for c in range(W) for r in range(H)]
    paths = [row_major, row_major[::-1], col_major, col_major[::-1]]
    if n == 0 or layout == "prefix":
        orders = [list(range(n)) + path for path in paths]
    else:
        if H != W or n % H:
            raise ValueError("Row layout needs a square grid and n divisible by H")
        unit = n // H
        orders = []
        for path in paths:
            order = []
            for group in range(H):
                order += list(range(group * unit, (group + 1) * unit))
                order += path[group * W:(group + 1) * W]
            orders.append(order)
    return torch.tensor(orders, dtype=torch.long)


@torch.inference_mode()
def scan_contributions(mixer, x, H, W):
    """Exact per-source decomposition of one SS2D forward.

    x: the mixer's input (B, n + HW, D). Returns
      influence (B, L, L): L2 norm over channels of source j's contribution to
                           output i, summed over paths; the diagonal is zeroed,
      output    (B, L, D): the mixer's output rebuilt from this decomposition.
    Sources are the scan inputs u (after in_proj and, on the grid, DWConv + SiLU);
    influence is measured before out_norm, out_proj and the adaLN gate.
    """
    if mixer.ssc != "none" or mixer.state_init != "none" or mixer.ssm_norm != "none":
        raise ValueError("scan_contributions supports the baseline and in-context arms only")
    batch, length, _ = x.shape
    HW = H * W
    n = length - HW
    K, d_inner = mixer.K, mixer.d_inner
    order = canonical_scan_orders(H, W, n, mixer.in_context_layout).to(x.device)

    projected = mixer.in_proj(x)
    grid = projected[:, n:].view(batch, H, W, d_inner).permute(0, 3, 1, 2)
    grid = mixer.act(mixer.conv2d(grid)).reshape(batch, d_inner, HW).transpose(1, 2)
    joint = torch.cat([projected[:, :n], grid], dim=1)              # (B, L, d)
    u = joint[:, order].float()                                     # (B, K, L, d)

    x_dbl = torch.einsum("bkld,kod->bkol", u, mixer.x_proj_weight.float())
    dt_r, B_ssm, C_ssm = torch.split(x_dbl, [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=2)
    delta = torch.einsum("bkrl,kdr->bkdl", dt_r, mixer.dt_projs_weight.float())
    delta = F.softplus(delta + mixer.dt_projs_bias.float()[None, :, :, None])       # (B, K, d, L)
    A = -torch.exp(mixer.A_logs.float())                            # (K, d, N)
    D = mixer.Ds.float()                                            # (K, d)

    contribution = torch.zeros(batch, length, length, d_inner, device=x.device)
    merged = torch.zeros(batch, length, d_inner, device=x.device)
    for k in range(K):
        # state[b, c, n, j]: the part of channel c's state n written by scan step j,
        # as a coefficient of u_j[c]. Each step decays it and appends one source.
        state = torch.zeros(batch, d_inner, mixer.d_state, length, device=x.device)
        for s in range(length):
            step = delta[:, k, :, s]
            state = state * torch.exp(step[:, :, None] * A[k])[..., None]
            state[:, :, :, s] = step[:, :, None] * B_ssm[:, k, :, s][:, None, :]
            weights = torch.einsum("bdnj,bn->bdj", state, C_ssm[:, k, :, s])
            terms = (weights * u[:, k].transpose(1, 2)).transpose(1, 2)     # (B, L, d)
            i = order[k, s]
            contribution[:, i, order[k]] += terms
            merged[:, i] += terms.sum(dim=1) + D[k] * u[:, k, s]

    influence = contribution.norm(dim=-1)
    influence[:, torch.arange(length), torch.arange(length)] = 0
    output = mixer.proj_drop(mixer.out_proj(mixer.out_norm(merged)))
    return influence, output


def influence_maps(influence, n, H, W):
    """Attention-map analogues from a (B, L, L) influence matrix.

    cells: mean influence of each image cell over all other image cells (Fig. 1);
    read:  (B, n, H, W) influence of each cell on token e ("the token attends");
    write: (B, n, H, W) influence of token e on each cell.
    """
    batch, HW = influence.shape[0], H * W
    grid = influence[:, n:, n:]
    cells = (grid.sum(dim=1) / max(HW - 1, 1)).reshape(batch, H, W)
    read = influence[:, :n, n:].reshape(batch, n, H, W)
    write = influence[:, n:, :n].transpose(1, 2).reshape(batch, n, H, W)
    return cells, read, write


def fit_probes(train_x, train_y, val_x, val_y, num_classes, *, weight_decays=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
               iterations=2000, batch_size=1024, lr=2e-3, holdout=0.1, seed=0, device="cpu"):
    """One independent multinomial logistic probe per token position.

    train_x, val_x: (N, tokens, D). The L2 strength of each token's probe is
    selected on a held-out part of the training images only, then that probe is
    scored on the validation images. Returns accuracy and the chosen L2 per token.
    """
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_x), generator=generator)
    held = int(round(len(train_x) * holdout))
    if not 0 < held < len(train_x):
        raise ValueError("holdout must leave images on both sides")
    held_idx, fit_idx = permutation[:held], permutation[held:]
    x_fit, y_fit = train_x[fit_idx], train_y[fit_idx]
    mean = x_fit.mean(dim=0, keepdim=True)
    std = x_fit.std(dim=0, keepdim=True)
    # A feature that never varies carries nothing: zero it rather than amplify noise.
    scale = torch.where(std > 1e-5 * (1 + mean.abs()), 1 / std.clamp_min(1e-12), torch.zeros_like(std))

    def prepare(data):
        return (data - mean) * scale

    x_fit = prepare(x_fit).to(device)
    y_fit = y_fit.to(device)
    tokens, dim = train_x.shape[1], train_x.shape[2]
    decays = torch.tensor(weight_decays, device=device, dtype=torch.float32)
    heads = len(decays)
    weight = torch.nn.Parameter(torch.zeros(tokens, heads, dim, num_classes, device=device))
    bias = torch.nn.Parameter(torch.zeros(tokens, heads, 1, num_classes, device=device))
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    batch_gen = torch.Generator(device=device).manual_seed(seed + 1)

    def logits_of(x):
        return x[:, None] @ weight + bias                           # (T, heads, N, C)

    with torch.enable_grad():
        for _ in range(iterations):
            chosen = torch.randint(len(x_fit), (min(batch_size, len(x_fit)),),
                                   generator=batch_gen, device=device)
            x_batch = x_fit[chosen].transpose(0, 1)                 # (T, N, D)
            logits = logits_of(x_batch)
            target = y_fit[chosen][None, None].expand(tokens, heads, -1)
            loss = F.cross_entropy(logits.flatten(0, 2), target.flatten(), reduction="sum") / len(chosen)
            loss = loss + 0.5 * (decays[None, :, None, None] * weight.square()).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()

    @torch.no_grad()
    def correct(x, y):
        """Number correct, shape (T, heads)."""
        hits = torch.zeros(tokens, heads, device=device)
        for start in range(0, len(x), 2048):
            part = prepare(x[start:start + 2048]).to(device).transpose(0, 1)
            target = y[start:start + 2048].to(device)
            hits += (logits_of(part).argmax(dim=-1) == target[None, None]).float().sum(dim=-1)
        return hits

    best = correct(train_x[held_idx], train_y[held_idx]).argmax(dim=1)
    accuracy = correct(val_x, val_y)[torch.arange(tokens), best] / len(val_x)
    return accuracy.cpu().numpy(), decays[best].cpu().numpy()
