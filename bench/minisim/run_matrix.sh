#!/bin/bash
# Serialized minisim experiment matrix: render -> bag -> podslam -> ATE + map eval.
#   bench/minisim/run_matrix.sh <rig.yaml> <scene> <condition> [condition...]
# Env: DUR (110), FPS (20), TRAJ (optional TUM trajectory, e.g. PX4 flight),
#      CLEAN=1 (delete frame dir after bag), SKIP_RENDER=1
set -uo pipefail
cd "$(dirname "$0")/../.."
RIG=$1; SCENE=$2; shift 2
RIGNAME=$(basename "$RIG" .yaml)
DUR=${DUR:-110}; FPS=${FPS:-20}
NF=$(python3 -c "print(int($DUR*$FPS))")
LASTCAM=$(python3 -c "import yaml; print(yaml.safe_load(open('$RIG'))['cameras'][-1]['name'])")
for COND in "$@"; do
  SEQ=${RIGNAME}_${SCENE}_${COND}
  D=datasets/data/minisim/$SEQ
  BAG=datasets/data/minisim/$SEQ.bag
  R=results/minisim/$SEQ
  echo "=== $SEQ ==="
  if [ ! -f "$BAG" ]; then
    if [ "${SKIP_RENDER:-0}" != 1 ]; then
      for i in 1 2 3 4; do
        n=$(ls "$D/$LASTCAM"/*.png 2>/dev/null | wc -l)
        [ "$n" -ge "$NF" ] && break
        echo "render attempt $i ($n/$NF)"
        bench/minisim/run_render.sh "$RIG" "$SCENE" "$COND" "$D" --duration "$DUR" --fps "$FPS" ${TRAJ:+--traj "$TRAJ"} || true
        sleep 90
      done
      n=$(ls "$D/$LASTCAM"/*.png 2>/dev/null | wc -l)
      [ "$n" -lt "$NF" ] && { echo "$SEQ: render incomplete ($n/$NF), skipping"; continue; }
    fi
    UV_NO_SYNC=1 PYTHONPATH=$PWD uv run python -m bench.minisim.flightbag "$D" "$RIG" "$D/flight.bag" && \
    UV_NO_SYNC=1 PYTHONPATH=$PWD uv run python -m bench.frames2bag "$D" "$D/flight.bag" "$RIG" "$BAG" || { echo "$SEQ: bag failed"; continue; }
    [ "${CLEAN:-0}" = 1 ] && rm -rf "$D/cam"*
  fi
  mkdir -p "$R"
  if [ ! -f "$R/est.tum" ]; then
    bash bench/guard.sh --cpus 6 --mem 10G --temp-max 97 --temp-hold 3 -- bash -c \
      "UV_NO_SYNC=1 PYTHONPATH=$PWD uv run python -m podslam.track_bag $BAG $R --rig $RIG --map-out $R/map ${FRONTEND:+--frontend $FRONTEND}" || \
    { echo "$SEQ: track killed, cool + retry"; sleep 120; \
      bash bench/guard.sh --cpus 6 --mem 10G --temp-max 97 --temp-hold 3 -- bash -c \
      "UV_NO_SYNC=1 PYTHONPATH=$PWD uv run python -m podslam.track_bag $BAG $R --rig $RIG --map-out $R/map ${FRONTEND:+--frontend $FRONTEND}"; } || { echo "$SEQ: track failed"; continue; }
  fi
  GT=$D/gt.tum; [ -f "$GT" ] || GT=datasets/data/minisim/${SEQ}_gt.tum
  UV_NO_SYNC=1 PYTHONPATH=$PWD uv run python -m bench.evaluate "$GT" "$R/est.tum" --json "$R/ate.json" || echo "$SEQ: evaluate failed"
  UV_NO_SYNC=1 PYTHONPATH=$PWD uv run python -m bench.minisim.map_eval "$R/map.npz" "$SCENE" --gt "$GT" --out "$R/map_metrics.json" || echo "$SEQ: map_eval failed"
  echo "=== $SEQ complete ==="
done
