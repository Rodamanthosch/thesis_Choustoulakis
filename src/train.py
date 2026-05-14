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
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class Denoiser(nn.Module):
    """
    JiT flow-matching denoiser.

    Matches all three notebooks exactly when using default arguments:
        label_drop_prob = 0.0   (no CFG)
        cfg_scale       = 1.0   (no CFG)

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

    def forward(self, x, labels):
        """
        v-loss for one batch.
        Label dropout is applied when label_drop_prob > 0 (CFG training mode).
        With label_drop_prob=0.0 this is identical to the notebook Denoiser.
        """
        # CFG label dropout
        if self.label_drop_prob > 0.0:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            labels = torch.where(drop, torch.full_like(labels, self.num_classes), labels)

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e

        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        x_pred = self.net(z, t.flatten(), labels)
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
        """
        Single model pass.
        With cfg_scale=1.0 (default): conditional only, no CFG — matches notebooks exactly.
        With cfg_scale>1.0: runs conditional + unconditional pass and applies guidance.
        """
        if self.cfg_scale == 1.0:
            # No CFG — exact notebook behaviour
            x_pred = self.net(z, t.flatten(), labels)
            return (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

        # CFG: run conditional and unconditional in one doubled batch
        t_flat = t.flatten()
        bsz = z.shape[0]
        z_double = torch.cat([z, z], dim=0)
        t_double = torch.cat([t_flat, t_flat], dim=0)
        y_null   = torch.full_like(labels, self.num_classes)
        y_double = torch.cat([labels, y_null], dim=0)

        x_pred_double = self.net(z_double, t_double, y_double)
        x_cond, x_uncond = x_pred_double[:bsz], x_pred_double[bsz:]

        # CFG interval: only apply guidance within [low, high] timestep range
        low, high = self.cfg_interval
        t_scalar = t_flat.view(bsz, 1, 1, 1)
        in_interval = (t_scalar >= low) & (t_scalar <= high)
        scale = torch.where(in_interval,
                             torch.full_like(t_scalar, self.cfg_scale),
                             torch.ones_like(t_scalar))

        x_guided = x_uncond + scale * (x_cond - x_uncond)
        t_view = t.view(bsz, *([1] * (z.ndim - 1)))
        return (x_guided - z) / (1.0 - t_view).clamp_min(self.t_eps)

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
