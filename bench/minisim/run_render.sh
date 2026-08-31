#!/usr/bin/env bash
# Guarded GPU render of one minisim sequence in the 3dfe/cuvslam-ml container.
#   bench/minisim/run_render.sh <rig.yaml> <scene> <condition> <out_dir> [extra render args...]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RIG=$1; SCENE=$2; COND=$3; OUT=$4; shift 4
mkdir -p "$OUT"
exec bash "$REPO/bench/guard.sh" --cpus 6 --mem 10G --vram-need 2G --temp-max 97 --temp-hold 3 -- \
  docker run --rm ${GUARD_DOCKER_ARGS:-} --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$REPO:$REPO" -v "$(cd "$OUT" && pwd):$(cd "$OUT" && pwd)" -w "$REPO" \
    3dfe/cuvslam-ml python3 -m bench.minisim.render "$RIG" "$SCENE" "$COND" "$(cd "$OUT" && pwd)" "$@"
