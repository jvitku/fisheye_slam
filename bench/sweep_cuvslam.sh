#!/usr/bin/env bash
# One cuVSLAM experiment on one TUM-VI room1 condition bag -> bench/results/room1_sweep/<name>_<cond>/
# (est.tum, frames.csv per-frame stats, run.log with BENCH_STATS). Extra args go to
# experiments/08_oakdpro_slam/cuvslam_track.py (knobs: --multicam-mode, --denoise,
# --no-motion-model, --slam, --mask, --preprocess, --imu-scale, --no-imu, ...).
#
# Usage: bench/sweep_cuvslam.sh <day|night|transition> <name> [cuvslam_track args...]
#   IMAGE=3dfe/cuvslam-ml for --preprocess enhance:<model> (torch)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COND="${1:?day|night|transition}"; NAME="${2:?experiment name}"; shift 2
BAG="$ROOT/datasets/data/tumvi/room1_${COND}.bag"
OUT="$ROOT/bench/results/room1_sweep/${NAME}_${COND}"
[ -f "$BAG" ] || { echo "missing $BAG"; exit 1; }
mkdir -p "$OUT"
docker run --rm ${GUARD_DOCKER_ARGS:-} --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$BAG:/data/input.bag:ro" \
    -v "$ROOT/rigs:/rigs:ro" \
    -v "$ROOT/experiments/08_oakdpro_slam:/scripts:ro" \
    -v "$ROOT/bench/timer_wrap.py:/timer_wrap.py:ro" \
    -v "$ROOT/bench/models:/models:ro" \
    -v "$OUT:/out" \
    "${IMAGE:-3dfe/cuvslam}" \
    python3 /timer_wrap.py python3 /scripts/cuvslam_track.py /data/input.bag /out \
        --rig /rigs/tumvi_room1.yaml --unrectified --stats /out/frames.csv "$@" \
        > "$OUT/run.log" 2>&1 || { echo "FAILED $NAME/$COND"; tail -5 "$OUT/run.log"; exit 1; }
grep -h "tracked\|BENCH_STATS" "$OUT/run.log" | tr '\n' ' '; echo
