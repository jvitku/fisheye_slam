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
CAND="${1:?usage: run_sim_candidate.sh openvins <rig.yaml> <bag> <run_dir>}"
RIG="$(realpath "${2:?rig yaml required}")"
BAG="$(realpath "${3:?bag required}")"
RUN="${4:?run dir required}"
mkdir -p "$RUN" && RUN="$(realpath "$RUN")"

GT="${BAG%.bag}.gt.txt"
[ -f "$GT" ] && cp "$GT" "$RUN/gt.txt"

case "$CAND" in
openvins)
    (cd "$ROOT" && uv run python -m bench.gen_openvins_config "$RIG" "$RUN/config")
    docker run --rm \
        -v "$BAG:/data/input.bag:ro" \
        -v "$RUN/config:/config:ro" \
        -v "$ROOT/bench/timer_wrap.py:/timer_wrap.py:ro" \
        -v "$RUN:/out" \
        3dfe/openvins \
        bash -lc "source /catkin_ws/devel/setup.bash && \
            (roscore >/dev/null 2>&1 &) && sleep 3 && \
            python3 /timer_wrap.py rosrun ov_msckf ros1_serial_msckf \
                /config/estimator_config.yaml _path_bag:=/data/input.bag \
                _save_total_state:=true \
                _filepath_est:=/out/state_estimate.txt \
                _filepath_std:=/out/state_std.txt \
                _filepath_gt:=/out/state_gt.txt \
                2>&1 | tail -40" 2>&1 | tee "$RUN/run.log"
    ;;
*)  echo "unknown candidate '$CAND' (openvins)"; exit 1 ;;
esac

echo "=== outputs in $RUN:"
ls -la "$RUN"
