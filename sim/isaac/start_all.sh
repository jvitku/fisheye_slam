#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Preflight: Isaac Sim 5.1 + 5 rendered cameras wants a >=24 GB GPU (8 GB is
# NVIDIA's bare minimum), ~20 GB RAM and ~25 GB of disk for the image + caches.
# bench/guard.sh refuses to start when the host cannot afford it; FORCE=1 skips.
if [ "${FORCE:-0}" != 1 ]; then
    ../../bench/guard.sh --check --mem "${ISAAC_MEM:-20G}" --vram-need "${ISAAC_VRAM_NEED:-7G}" \
        --disk-floor "${ISAAC_DISK_FLOOR:-25G}" \
        || { echo "start_all: preflight failed — run on the GPU host, or FORCE=1 to override" >&2; exit 1; }
fi
xhost +local:docker > /dev/null 2>&1 || true
exec docker compose up "$@"
