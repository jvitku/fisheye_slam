#!/usr/bin/env bash
# Option 3 — RTAB-Map RGB-D mapping on a bag-contract bag (open-source lane).
# Usage: ./run_rtabmap.sh <bag> [odom]
#   odom = rtabmap (default, built-in rgbd odometry)
#        | external (subscribe /uav1/odom — start the OpenVINS live container
#          on the same host network first; see README)
#
# Pipeline inside one 3dfe/rtabmap container:
#   roscore + rosbag play + camera_info publisher (from rig yaml)
#   + rgbd_odometry (or external odom) + rtabmap
#   -> /out/rtabmap.db, octomap.bt, cloud.ply, est.tum
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BAG="${1:?usage: run_rtabmap.sh <bag> [rtabmap|external]}"
ODOM="${2:-rtabmap}"
NAME="$(basename "${BAG%.bag}")"
OUT="$ROOT/experiments/08_oakdpro_slam/out/rtabmap/$NAME"
mkdir -p "$OUT"

NET=()
[ "$ODOM" = "external" ] && NET=(--network host)

docker run --rm "${NET[@]}" \
    -v "$(realpath "$BAG"):/data/input.bag:ro" \
    -v "$ROOT/rigs:/rigs:ro" \
    -v "$ROOT/experiments/08_oakdpro_slam:/scripts:ro" \
    -v "$OUT:/out" \
    3dfe/rtabmap \
    bash -c "/scripts/rtabmap_pipeline.sh /data/input.bag /out $ODOM 2>&1 | tee /out/run.log"

echo "rtabmap.db / octomap.bt / cloud.ply / est.tum in $OUT"
