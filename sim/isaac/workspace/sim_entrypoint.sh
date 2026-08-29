#!/usr/bin/env bash
set -euo pipefail

# Interactive mode: launch vanilla Isaac Sim GUI (no Pegasus)
if [ "${INTERACTIVE:-false}" = "true" ]; then
    exec /isaac-sim/isaac-sim.sh "$@"
fi


ARGS=()

# Headless mode: pass --headless to px4_drone.py
if [ "${HEADLESS:-false}" = "true" ]; then
    ARGS+=(--headless)

    # Auto-detect Tailscale IP for WebRTC streaming if not explicitly set
    if [ -z "${STREAM_PUBLIC_IP:-}" ]; then
        STREAM_PUBLIC_IP=$(ip -4 addr show tailscale0 2>/dev/null | grep -oP 'inet \K[^/]+' || echo "")
        export STREAM_PUBLIC_IP
    fi
fi

# SIM_SCRIPT selects the app: bench_drone.py (fisheye_slam multi-fisheye
# benchmark rig, default) or px4_drone.py (swarm_stack original, kept as
# reference).
exec /isaac-sim/python.sh "/workspace/${SIM_SCRIPT:-bench_drone.py}" "${ARGS[@]}"
