#!/bin/bash
# Serialized minisim experiment matrix: render -> bag -> podslam -> ATE + map eval.
#   bench/minisim/run_matrix.sh <rig.yaml> <scene> <condition> [condition...]
# Env: DUR (110), FPS (20), TRAJ (optional TUM trajectory, e.g. PX4 flight),
#      CLEAN=1 (delete frame dir after bag), SKIP_RENDER=1, FRONTEND (klt),
#      PODSLAM_IMG (fisheye/podslam), RENDER_COOLDOWN (90 s between render
#      attempts — thermal headroom on laptops; a desktop can use 10),
#      SEQ_SUFFIX (appended to the sequence name, e.g. the TRAJ identity:
#      COND=day SEQ_SUFFIX=px4b -> skydio6_indoor_day_px4b).
# Fully containerized: every python step runs in $PODSLAM_IMG with the repo
# mounted at its host path (host needs only bash + docker + awk).
set -uo pipefail
cd "$(dirname "$0")/../.."
RIG=$1; SCENE=$2; shift 2
RIGNAME=$(basename "$RIG" .yaml)
DUR=${DUR:-110}; FPS=${FPS:-20}
IMG=${PODSLAM_IMG:-fisheye/podslam}
COOL=${RENDER_COOLDOWN:-90}
# Unguarded light steps (bag conversion, evaluation) — sequential, IO-bound.
# The heavy steps (render, track) go through guard.sh with the GUARD_DOCKER_ARGS
# expansion deferred into the guarded child (guard exports it after preflight).
PY="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=$PWD -v $PWD:$PWD -w $PWD $IMG python3"
NF=$(awk "BEGIN{print int($DUR*$FPS)}")
LASTCAM=$($PY -c "import yaml; print(yaml.safe_load(open('$RIG'))['cameras'][-1]['name'])")
for COND in "$@"; do
  SEQ=${RIGNAME}_${SCENE}_${COND}${SEQ_SUFFIX:+_${SEQ_SUFFIX}}
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
        sleep "$COOL"
      done
      n=$(ls "$D/$LASTCAM"/*.png 2>/dev/null | wc -l)
      [ "$n" -lt "$NF" ] && { echo "$SEQ: render incomplete ($n/$NF), skipping"; continue; }
    fi
    $PY -m bench.minisim.flightbag "$D" "$RIG" "$D/flight.bag" && \
    $PY -m bench.frames2bag "$D" "$D/flight.bag" "$RIG" "$BAG" || { echo "$SEQ: bag failed"; continue; }
    [ "${CLEAN:-0}" = 1 ] && rm -rf "$D/cam"*
  fi
  mkdir -p "$R"
  if [ ! -f "$R/est.tum" ]; then
    TRACK="exec docker run --rm \${GUARD_DOCKER_ARGS:-} --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=$PWD -v $PWD:$PWD -w $PWD $IMG python3 -m podslam.track_bag $BAG $R --rig $RIG --map-out $R/map ${FRONTEND:+--frontend $FRONTEND}"
    bash bench/guard.sh --cpus 6 --mem 10G --temp-max 97 --temp-hold 3 -- bash -c "$TRACK" || \
    { echo "$SEQ: track killed, cool + retry"; sleep 120; \
      bash bench/guard.sh --cpus 6 --mem 10G --temp-max 97 --temp-hold 3 -- bash -c "$TRACK"; } || { echo "$SEQ: track failed"; continue; }
  fi
  GT=$D/gt.tum; [ -f "$GT" ] || GT=datasets/data/minisim/${SEQ}_gt.tum
  $PY -m bench.evaluate "$GT" "$R/est.tum" --json "$R/ate.json" || echo "$SEQ: evaluate failed"
  $PY -m bench.minisim.map_eval "$R/map.npz" "$SCENE" --gt "$GT" --est "$R/est.tum" --out "$R/map_metrics.json" || echo "$SEQ: map_eval failed"
  echo "=== $SEQ complete ==="
done
