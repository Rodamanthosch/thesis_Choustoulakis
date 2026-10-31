#!/bin/bash
# =============================================================================
# submit_imagenet.sh -- submit an ImageNet-256 run on ARIS as a chain of Slurm jobs.
#
#   ACCOUNT=<proj> GPUS=<per node> bash scripts/aris/submit_imagenet.sh <model> [N_JOBS] [EPOCHS]
#
#   ACCOUNT=pr019xxx GPUS=4 bash scripts/aris/submit_imagenet.sh spatial-nogate 8 100
#
# N_JOBS jobs of train_imagenet.sbatch are queued, each starting after the previous
# one ends (afterany: a wall-time kill is not "ok"), and each resumes from
# checkpoint-last.pt. Jobs left over once EPOCHS is reached exit within a minute.
# Cancel the rest of a chain with:  scancel -n jim-imagenet-<model>
#
# Knobs (environment variables):
#   ACCOUNT    Slurm account (required)    sacctmgr show assoc user=$USER format=account
#   GPUS       GPUs per node (required)    sinfo -o "%P %G %l %D"
#   PARTITION  default gpu
#   NODES      default 1 (DDP across nodes via torchrun c10d rendezvous)
#   TIME       wall time per job, default 48:00:00 (the partition's limit)
#   ACCUM      grad accumulation, default the config's (S configs 1, B/16 configs 4);
#              raise it if 1024/(NODES*GPUS*ACCUM) per GPU OOMs
#   JIM_ROOT   default /gpfs/work/pr019/nspanos/jim (layout of scripts/icarus/setup_env.sh)
#   DATA_DIR   default $JIM_ROOT/data/imagenet;  OUT default $JIM_ROOT/runs/imagenet/<model>
#
# Smoke test (1 epoch on the 50k val images, separate output folder):
#   ACCOUNT=... GPUS=... TIME=01:00:00 DATA_DIR=$JIM_ROOT/data/imagenet_quick \
#     OUT=$JIM_ROOT/runs/imagenet/spatial-nogate-smoke bash scripts/aris/submit_imagenet.sh spatial-nogate 1 1
# =============================================================================
set -euo pipefail

MODEL=${1:?usage: submit_imagenet.sh <jit|vim|vmamba|vmamba-incontext1|spatial|spatial-nogate|spatial-nogate-b16|vmamba2|sasf> [N_JOBS] [EPOCHS]}
N_JOBS=${2:-1}
EPOCHS=${3:-100}
ACCOUNT=${ACCOUNT:?set ACCOUNT (sacctmgr show assoc user=\$USER format=account)}
GPUS=${GPUS:?set GPUS, the GPUs per node of PARTITION (sinfo -o \"%P %G %l %D\")}
PARTITION=${PARTITION:-gpu}
NODES=${NODES:-1}
TIME=${TIME:-48:00:00}
export JIM_ROOT=${JIM_ROOT:-/gpfs/work/pr019/nspanos/jim}
mkdir -p "$JIM_ROOT/logs"

HERE=$(cd "$(dirname "$0")" && pwd)
prev=
for i in $(seq 1 "$N_JOBS"); do
  jid=$(sbatch --parsable \
          --account="$ACCOUNT" --partition="$PARTITION" \
          --nodes="$NODES" --gres=gpu:"$GPUS" --time="$TIME" \
          --job-name="jim-imagenet-$MODEL" \
          --output="$JIM_ROOT/logs/imagenet-$MODEL-%j.out" \
          ${prev:+--dependency=afterany:$prev} \
          --export=ALL,MODEL="$MODEL",EPOCHS="$EPOCHS" \
          "$HERE/train_imagenet.sbatch")
  echo "job $i/$N_JOBS: $jid${prev:+ (after $prev)}"
  prev=$jid
done
echo "logs: $JIM_ROOT/logs/imagenet-$MODEL-<jobid>.out"
