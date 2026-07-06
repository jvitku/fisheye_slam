#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# PID 1 is the bash wrapper (python.sh), not the actual Python process.
# Find the real px4_drone.py Python process and send SIGUSR1 to it.
PID=$(docker exec isaac-pegasus pgrep -f 'python3 /workspace/px4_drone.py')
if [ -z "$PID" ]; then
    echo "Error: px4_drone.py process not found in isaac-pegasus container" >&2
    exit 1
fi
docker exec isaac-pegasus kill -USR1 "$PID"
echo "Simulation restart requested (PID $PID)."

# The bridge caches old DDS endpoints; restart it so it discovers
# the recreated ROS2 publishers (lidar, cameras, etc.)
echo "Restarting ros1-bridge to re-discover ROS2 topics..."
docker restart ros1-bridge
echo "Done. Wait ~10s for bridge + FastLIO to stabilize."
