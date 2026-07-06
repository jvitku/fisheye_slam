#!/usr/bin/env bash
# Run Basalt VIO headless on a TUM-VI bag inside docker.
# Usage: ./run.sh [sequence]   (default room1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEQ="${1:-room1}"
BAG="$ROOT/datasets/data/tumvi/dataset-${SEQ}_512_16.bag"
OUT="$ROOT/experiments/02_basalt_tumvi/out/$SEQ"
[ -f "$BAG" ] || { echo "missing $BAG — run datasets/download_tumvi.sh $SEQ"; exit 1; }
mkdir -p "$OUT"

docker run --rm \
    -v "$BAG:/data/input.bag:ro" \
    -v "$OUT:/out" \
    3dfe/basalt \
    basalt_vio \
        --dataset-path /data/input.bag --dataset-type bag \
        --cam-calib /usr/local/etc/basalt/tumvi_512_ds_calib.json \
        --config-path /usr/local/etc/basalt/tumvi_512_config.json \
        --show-gui 0 --save-trajectory tum --result-path /out/vio_result.json

mv trajectory.txt "$OUT/" 2>/dev/null || true
echo "Results in $OUT"
