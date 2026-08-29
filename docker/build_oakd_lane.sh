#!/usr/bin/env bash
# Build the OAK-D Pro lane images (docs/oak_d_pro_slam.md) one at a time under
# the resource guard, so a build cannot exhaust the disk or cook the host.
#
# Usage: docker/build_oakd_lane.sh [image ...]
#   default order: spectacularai rtabmap cuvslam nvblox  (light -> heavy)
#   env: BUILD_MEM (12G) BUILD_CPUS (8) BUILD_DISK_FLOOR (25G) BUILD_TIMEOUT (90m)
#        TEMP_MAX (90) BUILD_LOG_DIR (bench/results/guard, gitignored)
# Note: `docker build` executes inside the docker daemon, outside the guard's
# cgroup — the disk/RAM/thermal watchdog and the timeout still apply (killing
# the client cancels the BuildKit job).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGES=("$@"); [ ${#IMAGES[@]} -gt 0 ] || IMAGES=(spectacularai rtabmap cuvslam nvblox)
LOGDIR="${BUILD_LOG_DIR:-$ROOT/bench/results/guard}"; mkdir -p "$LOGDIR"
status=0
for img in "${IMAGES[@]}"; do
    echo "=== building 3dfe/$img  ($(date +%H:%M:%S), free disk $(df -h --output=avail / | tail -1 | tr -d ' '))"
    if "$ROOT/bench/guard.sh" --mem "${BUILD_MEM:-12G}" --cpus "${BUILD_CPUS:-8}" \
            --disk-floor "${BUILD_DISK_FLOOR:-25G}" --timeout "${BUILD_TIMEOUT:-90m}" \
            --temp-max "${TEMP_MAX:-90}" \
            -- docker build --progress=plain -t "3dfe/$img" "$ROOT/docker/$img" \
            > "$LOGDIR/build-$img.log" 2>&1; then
        echo "OK   3dfe/$img  size=$(docker image inspect "3dfe/$img" --format '{{.Size}}' | numfmt --to=iec)  ($(date +%H:%M:%S))"
    else
        rc=$?
        echo "FAIL 3dfe/$img  rc=$rc  — $LOGDIR/build-$img.log, last lines:"
        grep -v "^#[0-9]* \(DONE\|CACHED\|sha256\|extracting\|resolve\)" "$LOGDIR/build-$img.log" | tail -25
        status=1
    fi
done
echo "=== done: status=$status, free disk $(df -h --output=avail / | tail -1 | tr -d ' ')"
exit $status
