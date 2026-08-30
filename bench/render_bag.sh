#!/usr/bin/env bash
# "Physics once, render offline": turn a FLIGHT bag (IMU + ground truth only,
# bench/record.sh with RECORD_IMAGES=0) into a full benchmark bag by rendering
# the rig's cameras along the recorded trajectory — one camera at a time, so
# it fits an 8 GB GPU — under the given lighting. Day and night bags of the
# SAME flight are just two invocations.
#
# Usage: bench/render_bag.sh <flight.bag> <rig.yaml> <day|night|half> <name> [rate_hz]
#   -> datasets/data/render/<name>/{gt_body.txt,poses.txt,frames/}
#   -> datasets/data/sim/<name>.bag          (then bench/compare_rigs.sh as usual)
#   env: RENDER_CAMERAS (all), RENDER_SETTLE (4), GUARD_OPTS
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${GUARD_ID:-}" ]; then
    exec "$ROOT/bench/guard.sh" ${GUARD_OPTS:---mem 16G --vram-need 5G --disk-floor 25G --timeout 6h} \
        -- "$(realpath "$0")" "$@"
fi
FLIGHT="$(realpath "${1:?usage: render_bag.sh <flight.bag> <rig.yaml> <day|night|half> <name> [rate]}")"
RIG="$(realpath "${2:?rig yaml required}")"
LIGHTING="${3:?day|night|half}"
NAME="${4:?output name required}"
RATE="${5:-20}"
RDIR="$ROOT/datasets/data/render/$NAME"
mkdir -p "$RDIR/frames" "$ROOT/datasets/data/sim"

echo "=== 1/4 ground truth (body frame) + ${RATE} Hz pose samples"
(cd "$ROOT" && uv run python -m bench.gt_extract "$FLIGHT" "$RDIR/gt_body.txt" \
             && uv run python -m bench.pose_sampler "$RDIR/gt_body.txt" "$RDIR/poses.txt" --rate "$RATE")

echo "=== 2/4 render pass (Isaac, headless, one camera at a time, lighting=$LIGHTING)"
(cd "$ROOT/sim/isaac" && docker compose run --rm --no-deps \
    -e HEADLESS=true -e SIM_SCRIPT=render_from_poses.py \
    -e RIG_CONFIG="/rigs/$(basename "$RIG")" -e SIM_LIGHTING="$LIGHTING" \
    -e POSES="/render/$NAME/poses.txt" -e RENDER_OUT="/render/$NAME/frames" \
    -e RENDER_CAMERAS="${RENDER_CAMERAS:-all}" -e RENDER_SETTLE="${RENDER_SETTLE:-4}" \
    isaac-sim)

echo "=== 3/4 assemble bag"
(cd "$ROOT" && uv run python -m bench.frames2bag "$RDIR/frames" "$FLIGHT" "$RIG" "$ROOT/datasets/data/sim/$NAME.bag")

echo "=== 4/4 done: datasets/data/sim/$NAME.bag  (next: bench/compare_rigs.sh datasets/data/sim/$NAME.bag $RIG)"
