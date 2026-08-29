#!/usr/bin/env bash
# Option 1 — Spectacular AI replay on a bag-contract bag (sim or HW).
# Usage: ./run_spectacularai.sh <bag> [--calib calib_<serial>.yaml]
# Sim bags use the rig yaml (exact GT calibration); HW bags pass --calib.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BAG="${1:?usage: run_spectacularai.sh <bag> [--calib file]}"
shift
NAME="$(basename "${BAG%.bag}")"
OUT="${OUT_DIR:-$ROOT/experiments/08_oakdpro_slam/out/spectacularai/$NAME}"  # OUT_DIR: bench/compare_rigs.sh
mkdir -p "$OUT"

CALIB_ARGS=(--rig /rigs/oakdpro.yaml)
EXTRA_MOUNTS=()
if [ "${1:-}" = "--calib" ]; then
    CALIB_ARGS=(--calib /data/calib.yaml)
    EXTRA_MOUNTS=(-v "$(realpath "$2"):/data/calib.yaml:ro")
fi

docker run --rm \
    -v "$(realpath "$BAG"):/data/input.bag:ro" \
    -v "$ROOT/rigs:/rigs:ro" \
    -v "$ROOT/experiments/08_oakdpro_slam:/scripts:ro" \
    -v "$OUT:/out" \
    "${EXTRA_MOUNTS[@]}" \
    3dfe/spectacularai \
    bash -c "python3 /scripts/bag2sai.py /data/input.bag /out/recording ${CALIB_ARGS[*]} && \
             python3 /scripts/sai_replay.py /out/recording /out 2>&1 | tee /out/run.log"

echo "est.tum + map.ply in $OUT — evaluate with bench/evaluate.py gt.tum est.tum"
