#!/usr/bin/env bash
# Run a candidate on a SIM benchmark bag with a config generated from the rig
# yaml, into a run dir that bench/collect_results.py understands.
#
# Usage: ./run_sim_candidate.sh openvins <rig.yaml> <bag> <run_dir>
#   <rig.yaml>  the SINGLE rig the bag follows — for a composite recording,
#               the member's own yaml after bench/split_bag.py (e.g.
#               rigs/oakdpro.yaml for combo_oakd.bag)
#   <bag>       if <bag%.bag>.gt.txt exists next to it (written by split_bag /
#               gt_extract) it is copied to <run_dir>/gt.txt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Heavy job: always run under the resource guard (bench/guard.sh) unless a
# guard is already active (GUARD_ID set) — see its header for the limits.
if [ -z "${GUARD_ID:-}" ]; then
    exec "$ROOT/bench/guard.sh" ${GUARD_OPTS:-} -- "$(realpath "$0")" "$@"
fi
CAND="${1:?usage: run_sim_candidate.sh openvins <rig.yaml> <bag> <run_dir>}"
RIG="$(realpath "${2:?rig yaml required}")"
BAG="$(realpath "${3:?bag required}")"
RUN="${4:?run dir required}"
mkdir -p "$RUN" && RUN="$(realpath "$RUN")"

GT="${BAG%.bag}.gt.txt"
[ -f "$GT" ] && cp "$GT" "$RUN/gt.txt"

OV_PARAMS="_save_total_state:=true _filepath_est:=/out/state_estimate.txt \
    _filepath_std:=/out/state_std.txt _filepath_gt:=/out/state_gt.txt"
NCAM=$(cd "$ROOT" && uv run python -c "import sys; from bench.rigdef import load_rig; print(len(load_rig(sys.argv[1])['cameras']))" "$RIG")

case "$CAND" in
openvins)
    (cd "$ROOT" && uv run python -m bench.gen_openvins_config "$RIG" "$RUN/config")
    if [ "$NCAM" -le 2 ]; then
        # 1-2 cameras: deterministic serial bag reader.
        INNER="python3 /timer_wrap.py rosrun ov_msckf ros1_serial_msckf \
            /config/estimator_config.yaml _path_bag:=/data/input.bag $OV_PARAMS \
            > /out/run.log 2>&1"
    else
        # >2 cameras: ros1_serial_msckf only supports 1-2 cams, so run the live
        # node and play the bag into it at BAG_RATE x real time (default 1.0 —
        # the node drops frames it cannot keep up with; timing is not
        # bit-reproducible like the serial reader).
        INNER="rosparam set use_sim_time true && \
            (python3 /timer_wrap.py rosrun ov_msckf run_subscribe_msckf \
                /config/estimator_config.yaml $OV_PARAMS > /out/run.log 2>&1 &) && \
            for i in \$(seq 60); do rosnode list 2>/dev/null | grep -q subscribe_msckf && break; sleep 1; done && \
            rosbag play --clock -d 3 -r ${BAG_RATE:-1.0} /data/input.bag > /out/play.log 2>&1; \
            sleep 3; pkill -INT -f lib/ov_msckf/run_subscribe_msckf || true; \
            for i in \$(seq 30); do pgrep -f lib/ov_msckf/run_subscribe_msckf >/dev/null || break; sleep 1; done"
    fi
    docker run --rm ${GUARD_DOCKER_ARGS:-} \
        -v "$BAG:/data/input.bag:ro" \
        -v "$RUN/config:/config:ro" \
        -v "$ROOT/bench/timer_wrap.py:/timer_wrap.py:ro" \
        -v "$RUN:/out" \
        3dfe/openvins \
        bash -lc "source /catkin_ws/devel/setup.bash && \
            (roscore >/dev/null 2>&1 &) && sleep 3 && $INNER; tail -40 /out/run.log"
    ;;
*)  echo "unknown candidate '$CAND' (openvins)"; exit 1 ;;
esac

echo "=== outputs in $RUN:"
ls -la "$RUN"
