#!/usr/bin/env bash
# Run OpenVINS serial (non-ROS-live) estimation on a TUM-VI bag inside docker.
# Usage: ./run.sh [sequence]   (default room1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEQ="${1:-room1}"
BAG="$ROOT/datasets/data/tumvi/dataset-${SEQ}_512_16.bag"
OUT="$ROOT/experiments/01_openvins_tumvi/out/$SEQ"
[ -f "$BAG" ] || { echo "missing $BAG — run datasets/download_tumvi.sh $SEQ"; exit 1; }
mkdir -p "$OUT"

docker run --rm \
    -v "$BAG:/data/input.bag:ro" \
    -v "$OUT:/out" \
    3dfe/openvins \
    bash -lc "source /catkin_ws/devel/setup.bash && \
        (roscore &) && sleep 3 && \
        rosrun ov_msckf ros1_serial_msckf \
            /catkin_ws/src/open_vins/config/tum_vi/estimator_config.yaml \
            _path_bag:=/data/input.bag 2>&1 | tee /out/run.log"

# Trajectory output path is set inside estimator_config.yaml (save_total_state /
# filepath options) — wire that up when the image first builds in Phase 1.

echo "Run log in $OUT/run.log — evaluate trajectories with evo_ape (tum format)."
