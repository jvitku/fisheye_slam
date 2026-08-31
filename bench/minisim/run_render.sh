#!/usr/bin/env bash
# Guarded GPU render of one minisim sequence in the ${MINISIM_IMG:-fisheye/podslam}
# container.
#   bench/minisim/run_render.sh <rig.yaml> <scene> <condition> <out_dir> [extra render args...]
# GUARD_DOCKER_ARGS is exported by guard.sh AFTER preflight, so its expansion
# must happen inside the guarded child (escaped below) — expanding it here
# would yield "" and the container would run unlimited and unlabelled,
# invisible to the guard's watchdog kill and orphan reaper.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RIG=$1; SCENE=$2; COND=$3; OUT=$4; shift 4
mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"
IMG=${MINISIM_IMG:-fisheye/podslam}
EXTRA=$(printf '%q ' "$@")
exec bash "$REPO/bench/guard.sh" --cpus 4 --mem 10G --vram-need 2G --temp-max 98 --temp-hold 6 -- \
  bash -c "exec docker run --rm \${GUARD_DOCKER_ARGS:-} --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
    -v $REPO:$REPO -v $OUT_ABS:$OUT_ABS -w $REPO \
    $IMG python3 -m bench.minisim.render $RIG $SCENE $COND $OUT_ABS $EXTRA"
