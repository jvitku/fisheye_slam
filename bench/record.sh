#!/usr/bin/env bash
# Record a benchmark bag (all rig cameras + IMU(s) + ground truth) from the ROS1
# side of the Isaac sim. Runs rosbag in a throwaway ros:noetic container on the
# host network (the ros1-bridge exposes everything on the local roscore).
#
# Usage: ./record.sh <rig.yaml> <output_name> [duration_s]      (RECORD_IMAGES=0: flight only)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RIG="${1:?usage: record.sh <rig.yaml> <output_name> [duration_s]}"
NAME="${2:?output name required}"
DURATION="${3:-120}"
OUTDIR="$ROOT/datasets/data/sim"
mkdir -p "$OUTDIR"

# Topics come from the rig yaml via rig_math (cameras, depth streams of
# `depth: true` cameras, every IMU — one per pod for composite rigs — and the
# ground truth). VERIFY-IN-SIM: depth topic name published by the Pegasus
# depth writer.
# RECORD_IMAGES=0 records the FLIGHT only (IMU streams + ground truth) for the
# offline render pass (bench/render_bag.sh).
TOPICS=$(cd "$ROOT" && uv run python -m bench.rig_topics "$RIG" \
    $( [ "${RECORD_IMAGES:-1}" = 0 ] && echo --no-images ) | tr '\n' ' ')

echo "Recording $DURATION s of: $TOPICS"
docker run --rm ${GUARD_DOCKER_ARGS:-} --network host \
    -v "$OUTDIR:/rec" \
    -e "ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}" \
    ros:noetic-ros-core \
    rosbag record --duration="$DURATION" -O "/rec/$NAME.bag" $TOPICS

echo "Saved $OUTDIR/$NAME.bag"
