#!/usr/bin/env bash
# Run one candidate (openvins | basalt) on one condition bag, headless in
# docker, and drop a TUM trajectory + resource stats into bench/results/.
#
# Usage: ./run_candidates.sh <openvins|basalt> <day|night|transition>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAND="${1:?openvins|basalt}"
COND="${2:?day|night|transition}"
BAG="$ROOT/datasets/data/tumvi/room1_${COND}.bag"
OUT="$ROOT/bench/results/room1/${CAND}_${COND}"
[ -f "$BAG" ] || { echo "missing $BAG"; exit 1; }
mkdir -p "$OUT"

case "$CAND" in
openvins)
    docker run --rm \
        -v "$BAG:/data/input.bag:ro" \
        -v "$ROOT/bench/configs/openvins_tumvi:/config:ro" \
        -v "$OUT:/out" \
        3dfe/openvins \
        bash -lc "source /catkin_ws/devel/setup.bash && \
            (roscore >/dev/null 2>&1 &) && sleep 3 && \
            /usr/bin/time -v rosrun ov_msckf ros1_serial_msckf \
                /config/estimator_config.yaml _path_bag:=/data/input.bag \
                2>&1 | tail -40" 2>&1 | tee "$OUT/run.log"
    ;;
basalt)
    docker run --rm \
        -v "$BAG:/data/input.bag:ro" \
        -v "$OUT:/out" \
        -w /out \
        3dfe/basalt \
        bash -c "/usr/bin/time -v basalt_vio \
            --dataset-path /data/input.bag --dataset-type bag \
            --cam-calib /src/basalt/data/tumvi_512_ds_calib.json \
            --config-path /src/basalt/data/tumvi_512_config.json \
            --show-gui 0 --step-by-step 0 --save-trajectory tum \
            2>&1 | tail -40" 2>&1 | tee "$OUT/run.log"
    ;;
*)  echo "unknown candidate $CAND"; exit 1 ;;
esac

echo "=== outputs in $OUT:"
ls -la "$OUT"
