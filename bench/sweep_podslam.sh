#!/usr/bin/env bash
# One podslam experiment on one TUM-VI room1 condition bag -> bench/results/room1_sweep/podslam_<name>_<cond>/
# (est.tum, frames.csv, run.log with BENCH_STATS); extra args go to podslam.track_bag
# (--frontend klt|xfeat, --preprocess, --masks, --kf-every, --lag, --max-features, ...).
# Runs on the host (uv env: gtsam + OpenCV; torch only needed for --frontend xfeat / enhance).
#
# Usage: bench/sweep_podslam.sh <day|night|transition> <name> [track_bag args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COND="${1:?day|night|transition}"; NAME="${2:?experiment name}"; shift 2
BAG="$ROOT/datasets/data/tumvi/room1_${COND}.bag"
OUT="$ROOT/bench/results/room1_sweep/podslam_${NAME}_${COND}"
[ -f "$BAG" ] || { echo "missing $BAG"; exit 1; }
mkdir -p "$OUT"
# Thermal budget on a laptop: podslam is single-threaded by design; keep the
# BLAS/OpenCV pools from fanning out (PODSLAM_THREADS is read by track_bag).
export OMP_NUM_THREADS="${PODSLAM_THREADS:-2}" OPENBLAS_NUM_THREADS="${PODSLAM_THREADS:-2}" MKL_NUM_THREADS="${PODSLAM_THREADS:-2}" PODSLAM_THREADS="${PODSLAM_THREADS:-2}"
(cd "$ROOT" && uv run python bench/timer_wrap.py uv run python -m podslam.track_bag "$BAG" "$OUT" \
    --rig rigs/tumvi_room1.yaml --stats "$OUT/frames.csv" "$@") > "$OUT/run.log" 2>&1 \
    || { echo "FAILED $NAME/$COND"; tail -8 "$OUT/run.log"; exit 1; }
grep -h "tracked\|BENCH_STATS" "$OUT/run.log" | tr '\n' ' '; echo
