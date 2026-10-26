#!/bin/bash
# =============================================================================
# setup_data.sh -- ImageNet-1k (ILSVRC2012) on Icarus, in ImageFolder layout:
#   $DATA/imagenet/train/<wnid>/*.JPEG   1000 classes, 1,281,167 images
#   $DATA/imagenet/val/<wnid>/*.JPEG     1000 classes x 50 = 50,000 images (devkit labels)
#   $DATA/imagenet_quick/train -> ../imagenet/val   (quick runs, same class order)
#   $DATA/imagenet/raw/                  the downloaded archives
#
# Needs setup_env.sh first (torchvision + scipy parse the devkit). Run on the LOGIN node,
# detached, since the train set is 148 GB:
#   nohup setsid bash setup_data.sh > $JIM_ROOT/logs/setup_data.log 2>&1 &
#   tail -f $JIM_ROOT/logs/setup_data.log
#
# Knobs (environment variables):
#   PARTS="val train"   what to prepare (default both). PARTS=val gives the 6.7 GB val set
#                       and imagenet_quick in minutes, before the big train download.
#   FROM=user@host:/path/to/raw
#                       copy the archives with rsync from a machine that already has them
#                       (e.g. ARIS: nspanos@login06.aris.grnet.gr:/gpfs/work/pr019/nspanos/jim/data/imagenet/raw)
#                       instead of downloading from image-net.org. Needs an SSH key for it.
#   KEEP_TARS=1         keep raw/*.tar after a verified extraction (default 1). KEEP_TARS=0
#                       deletes the 148 GB train tar once train/ is complete.
#   JIM_ROOT=...        same default as setup_env.sh.
#
# Re-running is safe at any point: downloads resume, MD5 checks are skipped once passed,
# and extraction skips what is already complete.
# =============================================================================
set -euo pipefail

JIM_ROOT=${JIM_ROOT:-/nfs/scratch/prod-2026-17/n.spanos/jim}
[ -f "$JIM_ROOT/env.sh" ] || { echo "FATAL: $JIM_ROOT/env.sh missing -- run setup_env.sh first"; exit 1; }
source "$JIM_ROOT/env.sh" >/dev/null
python -c "import torchvision, scipy" || { echo "FATAL: venv incomplete -- re-run setup_env.sh"; exit 1; }

PARTS=${PARTS:-val train}
KEEP_TARS=${KEEP_TARS:-1}
FROM=${FROM:-}
BASE_URL=https://image-net.org/data/ILSVRC/2012

RAW=$DATA/imagenet/raw
TRAIN=$DATA/imagenet/train
VAL=$DATA/imagenet/val
mkdir -p "$RAW" "$JIM_ROOT/logs"
log() { echo "=== [$(date '+%F %T')] $*"; }

# name  size(bytes)  md5 (torchvision's ImageNet checksums)
declare -A SIZE=( [ILSVRC2012_devkit_t12.tar.gz]=2568145
                  [ILSVRC2012_img_val.tar]=6744924160
                  [ILSVRC2012_img_train.tar]=147897477120 )
declare -A MD5=(  [ILSVRC2012_devkit_t12.tar.gz]=fa75699e90414af021442c21a62c3abf
                  [ILSVRC2012_img_val.tar]=29b22e2961454d5413ddabcf34fc5622
                  [ILSVRC2012_img_train.tar]=1d675b47d978889d74fa0da5fadfb00e )

fetch() {   # fetch <archive name>: download (or rsync), then size + md5 check, once
  local f=$1 dst=$RAW/$1
  if [ -f "$dst.md5ok" ]; then echo "$f: verified earlier"; return; fi
  log "fetch $f ($(( ${SIZE[$f]} / 1000000 )) MB)"
  if [ -n "$FROM" ]; then
    until rsync -a --partial --info=progress2 "$FROM/$f" "$dst"; do
      echo "[$(date)] rsync interrupted, retrying in 60s"; sleep 60
    done
  else
    until [ "$(stat -c%s "$dst" 2>/dev/null || echo 0)" -ge "${SIZE[$f]}" ]; do
      curl -sSL --fail --retry 20 --retry-delay 30 --speed-limit 10000 --speed-time 300 \
           -C - -o "$dst" "$BASE_URL/$f" \
        || { echo "[$(date)] curl stopped at $(stat -c%s "$dst" 2>/dev/null || echo 0) bytes, retrying in 60s"; sleep 60; }
    done
  fi
  local got_size; got_size=$(stat -c%s "$dst")
  [ "$got_size" = "${SIZE[$f]}" ] || { echo "FATAL: $f is $got_size bytes, expected ${SIZE[$f]} (delete it and re-run)"; exit 1; }
  log "md5 $f"
  local got; got=$(md5sum "$dst" | cut -d' ' -f1)
  [ "$got" = "${MD5[$f]}" ] || { echo "FATAL: $f md5 $got != ${MD5[$f]} (delete it and re-run)"; exit 1; }
  touch "$dst.md5ok"
  echo "$f: ok"
}

count() {   # count <dir>: "<classes> <images>"
  local c i
  c=$(find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  i=$(find "$1" -name '*.JPEG' 2>/dev/null | wc -l)
  echo "$c $i"
}

# ---- val (+ imagenet_quick) ------------------------------------------------------------
if [[ " $PARTS " == *" val "* ]]; then
  if [ "$(count "$VAL")" = "1000 50000" ]; then
    log "val: already complete ($VAL)"
  else
    fetch ILSVRC2012_devkit_t12.tar.gz
    fetch ILSVRC2012_img_val.tar
    log "val: extract with devkit labels -> $VAL"
    rm -rf "$VAL"
    python - "$RAW" "$DATA/imagenet" <<'EOF'
import os, sys, tarfile, shutil
from torchvision.datasets.imagenet import parse_devkit_archive, load_meta_file
raw, out = sys.argv[1], sys.argv[2]
parse_devkit_archive(raw)                       # writes raw/meta.bin (wnid -> classes, val wnids)
_, val_wnids = load_meta_file(raw)
val_dir = os.path.join(out, "val")
tmp = os.path.join(out, "val_tmp")
shutil.rmtree(tmp, ignore_errors=True)
with tarfile.open(os.path.join(raw, "ILSVRC2012_img_val.tar")) as t:
    t.extractall(tmp)
files = sorted(os.listdir(tmp))
assert len(files) == len(val_wnids) == 50000, (len(files), len(val_wnids))
for wnid in set(val_wnids):
    os.makedirs(os.path.join(tmp, wnid), exist_ok=True)
for wnid, f in zip(val_wnids, files):
    os.rename(os.path.join(tmp, f), os.path.join(tmp, wnid, f))
os.rename(tmp, val_dir)
n = {d: len(os.listdir(os.path.join(val_dir, d))) for d in os.listdir(val_dir)}
print(f"{val_dir}: {len(n)} classes, {sum(n.values())} images, per class min {min(n.values())} max {max(n.values())}")
EOF
    [ "$(count "$VAL")" = "1000 50000" ] || { echo "FATAL: val incomplete: $(count "$VAL")"; exit 1; }
  fi
  mkdir -p "$DATA/imagenet_quick"
  ln -sfn ../imagenet/val "$DATA/imagenet_quick/train"
  echo "imagenet_quick/train -> ../imagenet/val"
fi

# ---- train ----------------------------------------------------------------------------
if [[ " $PARTS " == *" train "* ]]; then
  if [ "$(count "$TRAIN")" = "1000 1281167" ]; then
    log "train: already complete ($TRAIN)"
  else
    fetch ILSVRC2012_img_train.tar
    log "train: extract -> $TRAIN"
    mkdir -p "$TRAIN"
    # The outer tar holds 1000 per-class tars (n01440764.tar, ...). Stream both levels, so no
    # intermediate copy of the 148 GB is written. A class counts as done only after its
    # .done marker, so an interrupted class is redone from scratch.
    python - "$RAW/ILSVRC2012_img_train.tar" "$TRAIN" <<'EOF'
import os, shutil, sys, tarfile, time
tar, train = sys.argv[1], sys.argv[2]
done = n_img = 0
t0 = time.time()
with tarfile.open(tar, mode="r|") as outer:
    for m in outer:
        wnid = os.path.basename(m.name).removesuffix(".tar")
        dst, marker = os.path.join(train, wnid), os.path.join(train, f".{wnid}.done")
        done += 1
        if os.path.exists(marker):
            continue
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        with tarfile.open(fileobj=outer.extractfile(m), mode="r|") as inner:
            k = 0
            for f in inner:
                if f.isfile():
                    with open(os.path.join(dst, os.path.basename(f.name)), "wb") as out:
                        shutil.copyfileobj(inner.extractfile(f), out)
                    k += 1
        open(marker, "w").close()
        n_img += k
        if done % 50 == 0:
            print(f"  {done}/1000 classes ({time.time() - t0:.0f}s)", flush=True)
print(f"extracted this run: {n_img} images")
EOF
    got=$(count "$TRAIN")
    [ "$got" = "1000 1281167" ] || { echo "FATAL: train incomplete: $got (expected 1000 1281167); re-run to resume"; exit 1; }
    if [ "$KEEP_TARS" = "0" ]; then
      rm -f "$RAW/ILSVRC2012_img_train.tar" "$RAW/ILSVRC2012_img_train.tar.md5ok"
      echo "deleted the train tar (KEEP_TARS=0)"
    fi
  fi
fi

# ---- summary ---------------------------------------------------------------------------
log "summary"
[ -d "$TRAIN" ] && echo "train: $(count "$TRAIN") (classes images; expected 1000 1281167)"
[ -d "$VAL" ]   && echo "val:   $(count "$VAL") (expected 1000 50000)"
du -sh "$DATA" 2>/dev/null | sed 's/^/size of DATA: /'
echo "Configs take the folder that holds train/: experiment.data_dir=$DATA/imagenet"
echo "  (quick runs: experiment.data_dir=$DATA/imagenet_quick)"
