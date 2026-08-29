#!/usr/bin/env bash
# Option 2 — cuVSLAM (PyCuVSLAM) odometry + optional nvblox TSDF fusion.
# Usage: ./run_cuvslam.sh <bag> [--map]     (--map: also fuse depth in nvblox)
# Needs an NVIDIA GPU (docker --gpus all); images: 3dfe/cuvslam, 3dfe/nvblox.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BAG="${1:?usage: run_cuvslam.sh <bag> [--map]}"
NAME="$(basename "${BAG%.bag}")"
OUT="$ROOT/experiments/08_oakdpro_slam/out/cuvslam/$NAME"
mkdir -p "$OUT"

DUMP=""
[ "${2:-}" = "--map" ] && DUMP="--dump-depth"

docker run --rm --gpus all \
    -v "$(realpath "$BAG"):/data/input.bag:ro" \
    -v "$ROOT/rigs:/rigs:ro" \
    -v "$ROOT/experiments/08_oakdpro_slam:/scripts:ro" \
    -v "$OUT:/out" \
    3dfe/cuvslam \
    python3 /scripts/cuvslam_track.py /data/input.bag /out \
        --rig /rigs/oakdpro.yaml $DUMP 2>&1 | tee "$OUT/run.log"

if [ -n "$DUMP" ]; then
    # nvblox fusion of the dumped depth + tracked poses -> mesh.ply.
    # The fuser wrapper converts depth/*.npy + poses.txt into the 3dmatch
    # layout consumed by nvblox's fuse_3dmatch example. VERIFY-ON-FIRST-RUN.
    docker run --rm --gpus all \
        -v "$OUT:/out" \
        -v "$ROOT/experiments/08_oakdpro_slam:/scripts:ro" \
        -v "$ROOT/rigs:/rigs:ro" \
        3dfe/nvblox \
        python3 /scripts/nvblox_fuse.py /out/depth /out/mesh.ply \
            --rig /rigs/oakdpro.yaml 2>&1 | tee "$OUT/nvblox.log"
fi

echo "outputs in $OUT — evaluate with bench/evaluate.py gt.tum est.tum"
