"""
src/train.py
============
Denoiser: flow-matching training and sampling wrapper.
Base extracted exactly from all three notebooks (Denoiser class is identical in all).

EXTENDED with CFG support (off by default = matches notebooks exactly):
  - label_drop_prob > 0.0  → enables label dropout during training (required for CFG)
  - cfg_scale > 1.0        → enables CFG during sampling
  - cfg_interval           → restrict CFG to timestep range [low, high] (JiT paper Fig 6)

With all defaults, this is byte-for-byte the same as the notebook Denoiser.

PATCH: BUG #6 — CFG interval mask now uses the strict-on-low, strict-on-high
       form from LTH14/JiT denoiser.py:
           (t < high) & ((low == 0) | (t > low))
       (was: (t >= low) & (t <= high))

PATCH: JiT-DENOISER — two more LTH14/JiT denoiser.py behaviours:
  - labels are dropped only in training mode (self.training), so a loss
    computed under .eval() sees the true labels;
  - _forward_sample is denoiser.py's: a conditional and an unconditional
    pass, each turned into v, mixed in v-space, at EVERY cfg_scale (was:
    cfg_scale 1.0 skipped the unconditional pass; >1.0 mixed x in one doubled
    batch). Same maths up to rounding; cfg 1.0 sampling now costs 2 passes.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class Denoiser(nn.Module):
    """
    JiT flow-matching denoiser.

    Training matches all three notebooks exactly with default arguments:
        label_drop_prob = 0.0   (no CFG)
        cfg_scale       = 1.0   (no CFG)
    Sampling follows LTH14/JiT denoiser.py (see JiT-DENOISER above): at
    cfg_scale 1.0 it equals the conditional prediction up to rounding.

    Enable CFG:
        label_drop_prob = 0.1   (drop label 10% of time during training)
        cfg_scale       = 2.5   (guidance strength at sampling time)
        cfg_interval    = (0.1, 1.0)  (optional: restrict CFG to this t range)
    """
    def __init__(
        self,
        net,
        img_size: int,
        num_classes: int = 10,
        # flow hyperparameters (paper Table 9)
        P_mean: float = -0.8,
        P_std: float = 0.8,
        t_eps: float = 0.05,
        noise_scale: float = 1.0,
        # ema
        ema_decay1: float = 0.9999,
        ema_decay2: float = 0.9996,
        # sampling
        sampling_method: str = "heun",         # "heun" | "euler"
        num_sampling_steps: int = 50,
        # ── CFG extensions (off by default) ──
        label_drop_prob: float = 0.0,          # 0.0 = no dropout = notebook default
        cfg_scale: float = 1.0,                # 1.0 = no CFG = notebook default
        cfg_interval: tuple = (0.0, 1.0),      # restrict CFG to this t range
    ):
        super().__init__()
        self.net = net
        self.img_size = img_size
        self.num_classes = num_classes

        # flow
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale

        # ema (populated lazily after model is on its final device)
        self.ema_decay1 = ema_decay1
        self.ema_decay2 = ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # sampling
        self.method = sampling_method
        self.steps = num_sampling_steps

        # CFG
        self.label_drop_prob = label_drop_prob
        self.cfg_scale = cfg_scale
        self.cfg_interval = cfg_interval

    # ── EMA (exact from notebooks) ────────────────────────────────────────────

    def init_ema(self):
        """Call once, after the model is on its final device."""
        with torch.no_grad():
            self.ema_params1 = [p.detach().clone() for p in self.parameters()]
            self.ema_params2 = [p.detach().clone() for p in self.parameters()]

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)

    def swap_ema(self, which: int = 1):
        ema = self.ema_params1 if which == 1 else self.ema_params2
        if ema is None:
            raise RuntimeError("EMA not initialized; call init_ema() first.")
        class _Swap:
            def __enter__(_self):
                _self.backup = [p.detach().clone() for p in self.parameters()]
                with torch.no_grad():
                    for p, e in zip(self.parameters(), ema):
                        p.copy_(e)
                return self
            def __exit__(_self, *a):
                with torch.no_grad():
                    for p, b in zip(self.parameters(), _self.backup):
                        p.copy_(b)
        return _Swap()

    # ── Flow utilities (exact from notebooks) ─────────────────────────────────

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    # ── Training ──────────────────────────────────────────────────────────────

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def forward(self, x, labels):
        """
        v-loss for one batch.
        Label dropout (CFG training) as in LTH14/JiT: training mode only.
        label_drop_prob = 0 skips the draw (JiT draws and drops nothing), so
        runs without CFG keep their exact random stream.
        """
        labels_dropped = self.drop_labels(labels) \
            if self.training and self.label_drop_prob > 0.0 else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e

        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss

    # ── Sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, labels, progress: bool = False):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)

        # (steps+1, bsz, 1, 1, 1) — same broadcast shape as denoiser.py
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(self.method)

        iterator = range(self.steps - 1)
        if progress:
            iterator = tqdm(iterator, desc=f"{self.method.upper()}", leave=False)
        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)

        # last step always Euler (matches denoiser.py)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        """Verbatim LTH14/JiT denoiser.py: conditional and unconditional pass,
        CFG in v-space, at every cfg_scale (1.0 included).
        t: (bsz, 1, 1, 1), as generate() slices it from `timesteps`."""
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred
