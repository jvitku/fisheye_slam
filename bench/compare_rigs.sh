#!/usr/bin/env bash
# Side-by-side comparison on ONE recording of a composite rig
# (rigs/pod3_oakdpro.yaml): split the bag per member pod, run the SAME
# estimator (OpenVINS) on every pod so the rig is the only variable, run the
# OAK-D native stack (cuVSLAM) where its image exists, evaluate everything
# against the simulator ground truth expressed in each pod's own IMU frame,
# and print the comparison table.
#
# Usage: ./compare_rigs.sh <combo.bag> <composite_rig.yaml> [results_name]
#   results land in bench/results/<results_name>/ (default: the bag's stem)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BAG="$(realpath "${1:?usage: compare_rigs.sh <combo.bag> <composite_rig.yaml> [results_name]}")"
RIG="$(realpath "${2:?composite rig yaml required}")"
NAME="${3:-$(basename "${BAG%.bag}")}"
RES="$ROOT/bench/results/$NAME"
mkdir -p "$RES"

echo "=== 1/3 split $BAG per pod"
(cd "$ROOT" && uv run python -m bench.split_bag "$BAG" --rig "$RIG")

echo "=== 2/3 run candidates"
while IFS=$'\t' read -r NS SUBRIG SUBNAME; do
    PODBAG="${BAG%.bag}_${NS}.bag"
    "$ROOT/bench/run_sim_candidate.sh" openvins "$SUBRIG" "$PODBAG" "$RES/openvins_${NS}"
    if [ "$SUBNAME" = "oakdpro" ]; then
        if docker image inspect 3dfe/cuvslam >/dev/null 2>&1; then
            OUT_DIR="$RES/cuvslam_${NS}" \
                "$ROOT/experiments/08_oakdpro_slam/run_cuvslam.sh" "$PODBAG"
            cp "${PODBAG%.bag}.gt.txt" "$RES/cuvslam_${NS}/gt.txt"
        else
            echo "(skipping cuVSLAM on $NS: docker image 3dfe/cuvslam not built — docker/cuvslam)"
        fi
    fi
done < <(cd "$ROOT" && uv run python -c "
import sys
from bench.rigdef import load_rig
rig = load_rig(sys.argv[1])
for p in rig['pods']:
    print(p['ns'], p['rig'], p['name'], sep='\t')
" "$RIG")

echo "=== 3/3 results ($RES)"
(cd "$ROOT" && uv run python -m bench.collect_results "$RES" --json "$RES/summary.json")
