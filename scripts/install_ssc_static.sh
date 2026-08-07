#!/usr/bin/env bash
# ============================================================
# install_ssc_static.sh — add ssc="static" (Mamba-3 bias-only control)
# ------------------------------------------------------------
# Idempotent. Run from the repo root:  bash install_ssc_static.sh
#   1. backs up src/models/vmamba.py -> src/models/vmamba.py.bak-static
#   2. applies 5 anchored edits (assert, param split, forward, docs, init)
#   3. py_compile syntax check (restores backup on failure)
#   4. writes configs/tiny_imagenet/jit-s2-vmamba-ssc-static.yaml
# Existing arms (none/bc/abc) proven byte-identical after this patch.
# ============================================================
set -euo pipefail
export PYTHONUTF8=1

TARGET="src/models/vmamba.py"
[ -f "$TARGET" ] || { echo "ERROR: run from the repo root ($TARGET not found)"; exit 1; }

if grep -q '"static"' "$TARGET"; then
  echo "vmamba.py already supports ssc='static' — skipping patch."
else
  cp "$TARGET" "$TARGET.bak-static"
  python - "$TARGET" <<'PYEOF'
import sys, py_compile

PATH = sys.argv[1]
src = open(PATH).read()

edits = [
    ('ssc: str = "none",            # "none" | "bc" | "abc"  (DiM-2 SSC)',
     'ssc: str = "none",            # "none" | "static" | "bc" | "abc"  (DiM-2 SSC; "static" = bias-only control)'),
    ('''        assert ssc in ("none", "bc", "abc"), ssc
        self.ssc = ssc
        if ssc != "none":
            self.bias_B = nn.Parameter(torch.ones(K, d_state))       # static, Mamba-3 style
            self.bias_C = nn.Parameter(torch.ones(K, d_state))
            self.cond_B_proj = nn.Linear(d_model, K * d_state, bias=False)  # zero-init (see initialize_weights)
            self.cond_C_proj = nn.Linear(d_model, K * d_state, bias=False)''',
     '''        assert ssc in ("none", "static", "bc", "abc"), ssc
        self.ssc = ssc
        if ssc != "none":
            self.bias_B = nn.Parameter(torch.ones(K, d_state))       # static, Mamba-3 style
            self.bias_C = nn.Parameter(torch.ones(K, d_state))
        if ssc in ("bc", "abc"):
            self.cond_B_proj = nn.Linear(d_model, K * d_state, bias=False)  # zero-init (see initialize_weights)
            self.cond_C_proj = nn.Linear(d_model, K * d_state, bias=False)'''),
    ('''        ssc_gate = None
        if self.ssc != "none" and cond is not None:
            assert extra_len == 0 and self.state_init == "none", (
                "ssc, state_init and the in-context prefix are separate "
                "conditioning arms; enable at most one per run."
            )
            B_ssm = B_ssm + (
                self.bias_B[None, :, :, None]
                + self.cond_B_proj(cond).view(B, K, d_state)[..., None]
            ).to(B_ssm.dtype)
            C_ssm = C_ssm + (
                self.bias_C[None, :, :, None]
                + self.cond_C_proj(cond).view(B, K, d_state)[..., None]
            ).to(C_ssm.dtype)
            if self.ssc == "abc":''',
     '''        ssc_gate = None
        if self.ssc != "none" and (self.ssc == "static" or cond is not None):
            assert extra_len == 0 and self.state_init == "none", (
                "ssc, state_init and the in-context prefix are separate "
                "conditioning arms; enable at most one per run."
            )
            add_B = self.bias_B[None, :, :, None]
            add_C = self.bias_C[None, :, :, None]
            if self.ssc in ("bc", "abc"):
                add_B = add_B + self.cond_B_proj(cond).view(B, K, d_state)[..., None]
                add_C = add_C + self.cond_C_proj(cond).view(B, K, d_state)[..., None]
            B_ssm = B_ssm + add_B.to(B_ssm.dtype)
            C_ssm = C_ssm + add_C.to(C_ssm.dtype)
            if self.ssc == "abc":'''),
    ('''        #   "none" \u2192 baseline / other arms (unchanged, byte-for-byte)
        #   "bc"   \u2192 B' = B + b0 + W_B c, C' = C + c0 + W_C c''',
     '''        #   "none"   \u2192 baseline / other arms (unchanged, byte-for-byte)
        #   "static" \u2192 B' = B + b0, C' = C + c0 ONLY (Mamba-3 bias control;
        #              equals "bc" at step 0, no condition path \u2014 isolates
        #              the static-bias effect from the conditioning effect)
        #   "bc"   \u2192 B' = B + b0 + W_B c, C' = C + c0 + W_C c'''),
    ('''        if self.ssc != "none":
            for block in self.blocks:
                nn.init.constant_(block.mixer.cond_B_proj.weight, 0)
                nn.init.constant_(block.mixer.cond_C_proj.weight, 0)''',
     '''        if self.ssc in ("bc", "abc"):
            for block in self.blocks:
                nn.init.constant_(block.mixer.cond_B_proj.weight, 0)
                nn.init.constant_(block.mixer.cond_C_proj.weight, 0)'''),
]

for old, new in edits:
    assert src.count(old) == 1, "anchor not found or not unique:\n" + old[:120]
    src = src.replace(old, new)

open(PATH, "w").write(src)
py_compile.compile(PATH, doraise=True)
print(f"Patched {PATH}: ssc='static' added (5 edits), syntax OK.")
PYEOF
  if [ $? -ne 0 ]; then
    echo "PATCH FAILED — restoring backup"; cp "$TARGET.bak-static" "$TARGET"; exit 1
  fi
fi

CFG="configs/tiny_imagenet/jit-s2-vmamba-ssc-static.yaml"
if [ -f "$CFG" ]; then
  echo "$CFG already exists — leaving it untouched."
else
  mkdir -p configs/tiny_imagenet
  cat > "$CFG" <<'YAMLEOF'
# ============================================================
# JiT-S2-VMamba (SS2D) on Tiny-ImageNet 64x64 — SSC-STATIC (bias-only control)
# ------------------------------------------------------------
# CONTROL arm for the DiM-2 SSC ladder. Applies ONLY the Mamba-3-style
# static biases (arXiv 2603.15569, Table 10: both biases, ones-init):
#   B' = B + b0,   C' = C + c0        (per direction, no condition path)
# There are NO condition projections W_B / W_C: this arm is EXACTLY what
# the "bc" arm equals at step 0 (bc's condition projections are zero-init),
# trained to convergence WITHOUT ever opening the condition path.
#
# Purpose — deconfounding: "bc" bundles two changes vs the adaLN baseline,
# (a) the static ones-init biases and (b) the DiM-2 condition biases W_B c /
# W_C c. If bc beats baseline, this control tells you how much of the gain
# is the pure Mamba-3 static-bias effect. Ladder:
#   baseline (ssc: none) -> +static bias (this) -> +condition (bc) -> +A-gate (abc)
#
# adaLN-Zero stays ON (shared scaffold, all arms). Condition information
# reaches this model ONLY through adaLN — SSC carries none here.
# Params: +2*K*d_state per block = +1,536 total at S/12 (vs +0.6M for bc).
# ============================================================

experiment:
  name: jit-s2-vmamba-tinyin-ssc-static
  model: vmamba
  dataset: imagenet
  data_dir: ./data/tiny-imagenet-200
  seed: 42

model:
  input_size: 64
  patch_size: 8
  in_channels: 3
  hidden_size: 384
  depth: 12
  num_heads: 6
  mlp_ratio: 4.0
  bottleneck_dim: 128
  num_classes: 200
  attn_drop: 0.0
  proj_drop: 0.0
  d_state: 16
  d_conv: 3
  expand: 1
  K: 4
  # -- conditioning arm: SSC static-bias-only control --
  in_context_len: 0                 # in-context prefix OFF (separate arm)
  state_init: none                  # state-init OFF (separate arm)
  ssc: static                       # none | static | bc | abc

training:
  epochs: 100
  batch_size: 128
  warmup_epochs: 5
  blr: 5.0e-5
  optimizer: adamw
  betas: [0.9, 0.95]
  weight_decay: 0.0
  ema_decay1: 0.9999
  ema_decay2: 0.9996
  amp: true

diffusion:
  P_mean: -0.8
  P_std: 0.8
  t_eps: 0.05
  noise_scale: 1.0

cfg:
  label_drop_prob: 0.1
  cfg_scale: 2.5
  cfg_interval: [0.1, 1.0]

sampling:
  method: heun
  steps: 50

checkpoint:
  save_last_freq: 5
  save_archive_freq: 25
  resume_from: null
  output_dir: experiments/tiny_imagenet/jit-s2-vmamba-ssc-static
YAMLEOF
  echo "Wrote $CFG"
fi

echo "install_ssc_static.sh — DONE."
