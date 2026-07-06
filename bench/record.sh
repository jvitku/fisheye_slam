#!/usr/bin/env bash
# Record a benchmark bag (all rig cameras + IMU + ground truth) from the ROS1
# side of the Isaac sim. Runs rosbag in a throwaway ros:noetic container on the
# host network (the ros1-bridge exposes everything on the local roscore).
#
# Usage: ./record.sh <rig.yaml> <output_name> [duration_s]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RIG="${1:?usage: record.sh <rig.yaml> <output_name> [duration_s]}"
NAME="${2:?output name required}"
DURATION="${3:-120}"
OUTDIR="$ROOT/datasets/data/sim"
mkdir -p "$OUTDIR"

TOPICS=$(cd "$ROOT" && uv run python -c "
import sys, yaml
rig = yaml.safe_load(open('$RIG'))
topics = [f\"/uav1/{c['name']}/color/image_raw\" for c in rig['cameras']]
topics += [rig['imu']['topic'], '/uav1/ground_truth']
print(' '.join(topics))
")

echo "Recording $DURATION s of: $TOPICS"
docker run --rm --network host \
    -v "$OUTDIR:/rec" \
    -e "ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}" \
    ros:noetic-ros-core \
    rosbag record --duration="$DURATION" -O "/rec/$NAME.bag" $TOPICS

echo "Saved $OUTDIR/$NAME.bag"
