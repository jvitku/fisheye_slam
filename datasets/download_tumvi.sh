#!/usr/bin/env bash
# Download TUM-VI sequences (512x512 fisheye stereo + synced IMU, rosbag format).
# Usage: ./download_tumvi.sh [room1] [room4] [corridor1] ...
# Default: room1. Data lands in datasets/data/tumvi/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/tumvi
cd data/tumvi

BASE="https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16"
SEQS=("${@:-room1}")

for seq in "${SEQS[@]}"; do
    f="dataset-${seq}_512_16.bag"
    if [ -f "$f" ]; then
        echo "== $f exists, skipping"
        continue
    fi
    echo "== downloading $seq"
    wget -c "$BASE/$f"
done
echo "Done: $(ls -sh *.bag 2>/dev/null)"
