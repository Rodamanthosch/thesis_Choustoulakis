#!/bin/bash
# =============================================================================
# train_imagenet.sbatch -- one Slurm job of an ImageNet-256 run on ARIS.
#
# Submit it through submit_imagenet.sh (it sets account / partition / GPUs / time
# and chains jobs). Each job resumes from $OUT/checkpoint-last.pt if it exists,
# trains until the wall time kills it (or EPOCHS is reached), and the next
# chained job picks up from there. checkpoint-last.pt is written EVERY epoch
# here (save_last_freq=1), so a wall-time kill loses at most one epoch; the
# RNG state rides along, so the chain equals one uninterrupted run.
#
# Environment (from submit_imagenet.sh):
#   MODEL     jit | vim | vmamba | spatial | spatial-nogate | vmamba2 | sasf   (required)
#   EPOCHS    total epochs of the run                              (default 100)
#   ACCUM     training.accum_iter; global batch stays 1024         (default 1)
#   JIM_ROOT  folder made by scripts/icarus/setup_env.sh           (default ARIS work dir)
#   DATA_DIR  folder holding train/                                (default $DATA/imagenet)
#   OUT       checkpoint folder                                    (default $RUNS/imagenet/$MODEL)
# =============================================================================
#SBATCH --job-name=jim-imagenet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
set -euo pipefail

MODEL=${MODEL:?set MODEL (jit|vim|vmamba|spatial|spatial-nogate|vmamba2|sasf)}
EPOCHS=${EPOCHS:-100}
ACCUM=${ACCUM:-1}
JIM_ROOT=${JIM_ROOT:-/gpfs/work/pr019/nspanos/jim}
[ -f "$JIM_ROOT/env.sh" ] || { echo "FATAL: $JIM_ROOT/env.sh missing -- run scripts/icarus/setup_env.sh with JIM_ROOT=$JIM_ROOT"; exit 1; }
source "$JIM_ROOT/env.sh" >/dev/null
DATA_DIR=${DATA_DIR:-$DATA/imagenet}
OUT=${OUT:-$RUNS/imagenet/$MODEL}

case "$MODEL" in
  jit)     CONFIG=configs/imagenet/jit-s-imagenet.yaml ;;
  vim)     CONFIG=configs/imagenet/jit-s2-vim-imagenet.yaml ;;
  vmamba)  CONFIG=configs/imagenet/jit-s16-vmamba-imagenet.yaml ;;
  spatial) CONFIG=configs/imagenet/jit-s2-spatial-gate-imagenet.yaml ;;
  spatial-nogate) CONFIG=configs/imagenet/jit-s2-spatial-nogate-imagenet.yaml ;;
  vmamba2) CONFIG=configs/imagenet/jit-s2-vmamba2-imagenet.yaml ;;
  sasf)    CONFIG=configs/imagenet/jit-s2-vmamba-sasf-imagenet.yaml ;;
  *) echo "FATAL: unknown MODEL=$MODEL"; exit 1 ;;
esac

cd "$REPO"
[ -d "$DATA_DIR/train" ] || { echo "FATAL: $DATA_DIR/train not found"; exit 1; }

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
NNODES=${SLURM_NNODES:-1}
# Loader workers per GPU: the node's cores split over its GPUs, one left for the
# training process, capped at JiT's 12.
CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}
WORKERS=${WORKERS:-$(( CPUS / GPUS_PER_NODE - 1 ))}
(( WORKERS > 12 )) && WORKERS=12
(( WORKERS < 2 ))  && WORKERS=2

RESUME=()
if [ -f "$OUT/checkpoint-last.pt" ]; then
  RESUME=(checkpoint.resume_from="$OUT/checkpoint-last.pt")
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# One progress-bar line per minute instead of a carriage-return stream in the log.
export TQDM_MININTERVAL=60

echo "=============================================================="
echo "  job $SLURM_JOB_ID on $SLURM_JOB_NODELIST  ($(date '+%F %T'))"
echo "  model=$MODEL  config=$CONFIG  commit=$(git rev-parse --short HEAD)"
echo "  nodes=$NNODES x gpus=$GPUS_PER_NODE ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
echo "  epochs=$EPOCHS  accum_iter=$ACCUM  workers/gpu=$WORKERS"
echo "  data=$DATA_DIR  out=$OUT"
echo "  resume: ${RESUME[*]:-none (fresh start)}"
echo "=============================================================="

srun --ntasks-per-node=1 torchrun \
    --nnodes="$NNODES" --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:29500" \
    scripts/run_experiment.py --config "$CONFIG" \
    experiment.data_dir="$DATA_DIR" \
    training.epochs="$EPOCHS" \
    training.accum_iter="$ACCUM" \
    training.num_workers="$WORKERS" \
    checkpoint.output_dir="$OUT" \
    checkpoint.save_last_freq=1 \
    ${RESUME[@]+"${RESUME[@]}"}   # empty-array-safe under set -u on bash < 4.4
