#!/bin/bash
# =============================================================================
# internals_spatial.sbatch -- register-style diagnostics of two JiT-Spatial-Mamba
# checkpoints (scripts/measure_internals_spatial.py) on ONE GPU. Inference only.
#
#   sbatch --account=ACCOUNT --partition=PARTITION scripts/icarus/internals_spatial.sbatch
#   SMOKE=1 sbatch ...      # ~10 min check of the whole pipeline first
#
# Environment (all optional; the defaults follow train_imagenet.sbatch's layout):
#   JIM_ROOT       folder made by scripts/icarus/setup_env.sh
#   DATA_DIR       ImageNet root with train/ and val/ class folders   ($DATA/imagenet)
#   CONFIG_BASE    reference model config     (B/16 no-gate Spatial-Mamba)
#   CKPT_BASE      its checkpoint             ($RUNS/imagenet/spatial-nogate-b16/checkpoint-ep250.pt)
#   NAME_TOKENS    label of the token model   (class10)
#   CONFIG_TOKENS  token model config         (B/16 no-gate + 10 class tokens)
#   CKPT_TOKENS    its checkpoint             ($RUNS/imagenet/spatial-nogate-b16-incontext10/checkpoint-ep250.pt)
#   OUT            new output folder          ($RUNS/internals/spatial-b16-<NAME_TOKENS>-<date>)
#   ATTN_BATCH     images per exact decomposition (float64, ~2.5 GB each at 266 tokens; default 4)
#   EXTRA_ARGS     appended to the python call, e.g. a third "--model NAME CONFIG CKPT"
# Compare checkpoints from the SAME epoch: the baseline run stops at 250.
# =============================================================================
#SBATCH --job-name=jim-internals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

JIM_ROOT=${JIM_ROOT:-/nfs/scratch/prod-2026-17/n.spanos/jim}
[ -f "$JIM_ROOT/env.sh" ] || { echo "FATAL: $JIM_ROOT/env.sh missing -- run scripts/icarus/setup_env.sh"; exit 1; }
source "$JIM_ROOT/env.sh" >/dev/null

DATA_DIR=${DATA_DIR:-$DATA/imagenet}
CONFIG_BASE=${CONFIG_BASE:-configs/imagenet/jit-b16-spatial-nogate-imagenet.yaml}
CKPT_BASE=${CKPT_BASE:-$RUNS/imagenet/spatial-nogate-b16/checkpoint-ep250.pt}
NAME_TOKENS=${NAME_TOKENS:-class10}
CONFIG_TOKENS=${CONFIG_TOKENS:-configs/imagenet/jit-b16-spatial-nogate-incontext-class10-imagenet.yaml}
CKPT_TOKENS=${CKPT_TOKENS:-$RUNS/imagenet/spatial-nogate-b16-incontext10/checkpoint-ep250.pt}
SMOKE=${SMOKE:-0}
TAG=$([ "$SMOKE" = 1 ] && echo smoke || echo full)
OUT=${OUT:-$RUNS/internals/spatial-b16-$NAME_TOKENS-$TAG-$(date +%Y%m%d-%H%M%S)}
ATTN_BATCH=${ATTN_BATCH:-4}
WORKERS=${SLURM_CPUS_PER_TASK:-8}

cd "$REPO"
[ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/val" ] || { echo "FATAL: $DATA_DIR needs train/ and val/"; exit 1; }
for f in "$CKPT_BASE" "$CKPT_TOKENS"; do
  [ -e "$f" ] || { echo "FATAL: checkpoint not found: $f (set CKPT_BASE / CKPT_TOKENS)"; exit 1; }
done

if [ "$SMOKE" = 1 ]; then
  SIZE=(--stats-per-class 1 --probe-train-per-class 2 --probe-val-per-class 2 --probe-iterations 50
        --attn-images 4 --figure-images 2 --dynamics-per-class 1 --ablation-per-class 1
        --ablation-t .5 --probe-t .3 --dynamics-t .5)
else
  SIZE=()   # the script's defaults: 5k images for norms, 20k + 10k for probes, 32 for influence
fi

echo "=============================================================="
echo "  job ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-$(hostname)}  ($(date '+%F %T'))"
echo "  commit=$(git rev-parse --short HEAD)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  base:   $CONFIG_BASE  $CKPT_BASE"
echo "  $NAME_TOKENS: $CONFIG_TOKENS  $CKPT_TOKENS"
echo "  data=$DATA_DIR  out=$OUT  smoke=$SMOKE"
echo "=============================================================="

python scripts/measure_internals_spatial.py \
    --data-dir "$DATA_DIR" --out-dir "$OUT" \
    --model base "$CONFIG_BASE" "$CKPT_BASE" \
    --model "$NAME_TOKENS" "$CONFIG_TOKENS" "$CKPT_TOKENS" \
    --batch-size 128 --attn-batch-size "$ATTN_BATCH" --workers "$WORKERS" \
    ${SIZE[@]+"${SIZE[@]}"} ${EXTRA_ARGS:-}

echo "Done: $OUT"
