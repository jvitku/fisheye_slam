#!/usr/bin/env bash
# One podslam run on a Hilti-Oxford 2022 sequence -> bench/results/hilti2022/podslam_<name>_<seq>/
# (est.tum, frames.csv, run.log, eval.json).  Extra args go to podslam.track_bag
# (--cams cam0,cam1 for a subset of the 5 cameras, --noise-gate, --kf-every, ...).
#
# Usage: bench/run_hilti.sh <exp14_basement_2|exp18_corridor_lower_gallery_2|...> <name> [track_bag args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEQ="${1:?sequence}"; NAME="${2:?experiment name}"; shift 2
BAG="$ROOT/datasets/data/hilti2022/${SEQ}.bag"
GT="$ROOT/datasets/data/hilti2022/ground_truth/${SEQ}_imu.txt"
OUT="$ROOT/bench/results/hilti2022/podslam_${NAME}_${SEQ}"
[ -f "$BAG" ] || { echo "missing $BAG"; exit 1; }
[ -f "$GT" ] || { echo "missing dense ground truth $GT"; exit 1; }
mkdir -p "$OUT"
export UV_NO_SYNC=1 OMP_NUM_THREADS="${PODSLAM_THREADS:-1}" OPENBLAS_NUM_THREADS="${PODSLAM_THREADS:-1}" MKL_NUM_THREADS="${PODSLAM_THREADS:-1}" PODSLAM_THREADS="${PODSLAM_THREADS:-1}"
(cd "$ROOT" && uv run python bench/timer_wrap.py uv run python -m podslam.track_bag "$BAG" "$OUT" \
    --rig rigs/hilti2022.yaml --stats "$OUT/frames.csv" "$@") > "$OUT/run.log" 2>&1 \
    || { echo "FAILED $NAME/$SEQ"; tail -8 "$OUT/run.log"; exit 1; }
grep -h "tracked\|BENCH_STATS\|using cameras" "$OUT/run.log" | tr '\n' ' '; echo
(cd "$ROOT" && PYTHONPATH="$ROOT" uv run python bench/evaluate.py "$GT" "$OUT/est.tum" --json "$OUT/eval.json" 2>&1 | tail -3) || true
