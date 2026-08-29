#!/usr/bin/env bash
# Resource guard for heavy benchmark jobs (docker builds, Isaac Sim, SLAM runs)
# so a runaway job cannot take the machine down. Every runner in bench/ and
# experiments/ is guard-aware: run them through this wrapper (compare_rigs.sh
# and run_sim_candidate.sh re-exec themselves under it automatically).
#
#   bench/guard.sh [options] -- <command> [args...]
#   bench/guard.sh --check [options]            preflight only, exit 0/1
#
# Preflight — refuses to start unless ALL hold:
#   free disk (repo + docker root)   >= --disk-floor   (15G)
#   available RAM                    >= --mem          (12G)
#   free VRAM                        >= --vram-need    (0; e.g. 7G for Isaac)
#   CPU package temperature          <  --temp-max     (90 C)
#   1-min load average               <  --load-max     (ncpu)
# Limits while running:
#   host processes  : systemd user scope, MemoryMax=--mem, no swap, pinned to
#                     the first --cpus cores (taskset: the user cgroup has no
#                     cpu controller), nice 10, ionice best-effort/7
#   docker containers: runners append $GUARD_DOCKER_ARGS
#                     (--memory --memory-swap --cpus --pids-limit
#                      --label fisheye_guard=<id>) so the guard can find and
#                     kill them
# Watchdog every --interval s (5): kills the job (scope + labelled containers)
#   when available RAM < --ram-floor (2G), disk < --disk-floor,
#   free VRAM < --vram-floor (512M), package temp >= --temp-max, or
#   runtime > --timeout (4h). Samples + reason: bench/results/guard/<id>.log
#
# Sizes accept K/M/G/T suffixes. Exit code = the command's, 137 when killed.
set -euo pipefail
export LC_ALL=C   # integer arithmetic + numfmt/awk output must not be localized
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MEM=12G; CPUS=$(( $(nproc) / 2 )); DISK_FLOOR=15G; RAM_FLOOR=2G
VRAM_NEED=0; VRAM_FLOOR=512M; TEMP_MAX=90; LOAD_MAX=$(nproc)
TIMEOUT=4h; INTERVAL=5; CHECK_ONLY=0

to_bytes() {   # 12G / 512M / 4096 -> bytes
    local v="${1^^}" n
    n="${v%[KMGT]}"
    case "$v" in
        *K) echo $(( n * 1024 ));;      *M) echo $(( n * 1024 ** 2 ));;
        *G) echo $(( n * 1024 ** 3 ));; *T) echo $(( n * 1024 ** 4 ));;
        *)  echo "$v";;
    esac
}
to_seconds() {  # 4h / 30m / 90s / 120 -> seconds
    local v="$1"
    case "$v" in
        *h) echo $(( ${v%h} * 3600 ));; *m) echo $(( ${v%m} * 60 ));;
        *s) echo "${v%s}";;             *)  echo "$v";;
    esac
}
human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1"; }

usage() { sed -n '2,30p' "$0"; exit "${1:-1}"; }
while [ $# -gt 0 ]; do
    case "$1" in
        --mem) MEM="$2"; shift 2;;              --cpus) CPUS="$2"; shift 2;;
        --disk-floor) DISK_FLOOR="$2"; shift 2;; --ram-floor) RAM_FLOOR="$2"; shift 2;;
        --vram-need) VRAM_NEED="$2"; shift 2;;  --vram-floor) VRAM_FLOOR="$2"; shift 2;;
        --temp-max) TEMP_MAX="$2"; shift 2;;    --load-max) LOAD_MAX="$2"; shift 2;;
        --timeout) TIMEOUT="$2"; shift 2;;      --interval) INTERVAL="$2"; shift 2;;
        --check) CHECK_ONLY=1; shift;;
        -h|--help) usage 0;;
        --) shift; break;;
        *) echo "guard: unknown option $1" >&2; usage;;
    esac
done
[ "$CHECK_ONLY" = 1 ] || [ $# -gt 0 ] || usage
[ "$CPUS" -ge 1 ] || CPUS=1
[ "$CPUS" -le "$(nproc)" ] || CPUS=$(nproc)

MEM_B=$(to_bytes "$MEM"); DISK_FLOOR_B=$(to_bytes "$DISK_FLOOR"); RAM_FLOOR_B=$(to_bytes "$RAM_FLOOR")
VRAM_NEED_B=$(to_bytes "$VRAM_NEED"); VRAM_FLOOR_B=$(to_bytes "$VRAM_FLOOR"); TIMEOUT_S=$(to_seconds "$TIMEOUT")
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)

# --- samplers -----------------------------------------------------------------
mem_available_b() { echo $(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) * 1024 )); }
disk_free_b()     { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' '; }
min_disk_free_b() {
    local a b; a=$(disk_free_b "$ROOT"); b=$(disk_free_b "$DOCKER_ROOT" || echo "$a")
    [ "${b:-$a}" -lt "$a" ] && echo "$b" || echo "$a"
}
vram_free_b()  {
    local m=""
    command -v nvidia-smi >/dev/null && m=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$m" ] && echo $(( m * 1024 * 1024 )) || echo ""
}
pkg_temp_c()   {
    local z
    for z in /sys/class/thermal/thermal_zone*; do
        [ "$(cat "$z/type" 2>/dev/null)" = "x86_pkg_temp" ] && { awk '{print int($1/1000)}' "$z/temp"; return; }
    done
    echo ""
}
gpu_temp_c()   { command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo ""; }
load1()        { cut -d' ' -f1 /proc/loadavg; }

sample_line() {
    printf 'ram_avail=%s disk_free=%s vram_free=%s pkg_temp=%sC gpu_temp=%sC load=%s' \
        "$(human "$(mem_available_b)")" "$(human "$(min_disk_free_b)")" \
        "$(human "${1:-$(vram_free_b)}")" "$(pkg_temp_c)" "$(gpu_temp_c)" "$(load1)"
}

# --- preflight ----------------------------------------------------------------
preflight() {
    local ok=1 v
    v=$(min_disk_free_b)
    [ "$v" -ge "$DISK_FLOOR_B" ] || { echo "guard: free disk $(human "$v") < floor $(human "$DISK_FLOOR_B") ($ROOT / $DOCKER_ROOT)"; ok=0; }
    v=$(mem_available_b)
    [ "$v" -ge "$MEM_B" ] || { echo "guard: available RAM $(human "$v") < requested --mem $(human "$MEM_B")"; ok=0; }
    v=$(vram_free_b)
    if [ "$VRAM_NEED_B" -gt 0 ]; then
        [ -n "$v" ] || { echo "guard: --vram-need set but nvidia-smi unavailable"; ok=0; }
        [ -z "$v" ] || [ "$v" -ge "$VRAM_NEED_B" ] || { echo "guard: free VRAM $(human "$v") < needed $(human "$VRAM_NEED_B")"; ok=0; }
    fi
    v=$(pkg_temp_c)
    [ -z "$v" ] || [ "$v" -lt "$TEMP_MAX" ] || { echo "guard: CPU package ${v}C >= ${TEMP_MAX}C — let the machine cool down"; ok=0; }
    v=$(load1)
    awk -v l="$v" -v m="$LOAD_MAX" 'BEGIN{exit !(l < m)}' || { echo "guard: load average $v >= $LOAD_MAX — something else is busy"; ok=0; }
    echo "guard: preflight $( [ $ok = 1 ] && echo OK || echo FAILED ) — $(sample_line)" >&2
    [ $ok = 1 ]
}

reap_orphans() {   # containers labelled by a guard whose PID no longer exists
    local line cid label pid
    while read -r cid label; do
        [ -n "$cid" ] || continue
        pid="${label##*-}"
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "guard: killing orphaned container $cid (guard $label is gone)" >&2
            docker kill "$cid" >/dev/null 2>&1 || true
        fi
    done < <(docker ps --filter "label=fisheye_guard" --format '{{.ID}} {{.Label "fisheye_guard"}}' 2>/dev/null)
}
reap_orphans
preflight || { echo "guard: refusing to start (override thresholds with --mem/--disk-floor/... if you really mean it)" >&2; exit 1; }
[ "$CHECK_ONLY" = 1 ] && exit 0

# --- launch under limits ------------------------------------------------------
GUARD_ID="$(date +%Y%m%d-%H%M%S)-$$"
LOGDIR="$ROOT/bench/results/guard"; mkdir -p "$LOGDIR"; LOG="$LOGDIR/$GUARD_ID.log"
export GUARD_ID
export GUARD_DOCKER_ARGS="--memory=${MEM} --memory-swap=${MEM} --cpus=${CPUS} --pids-limit=4096 --label fisheye_guard=${GUARD_ID}"
SCOPE="fisheye-guard-${GUARD_ID}.scope"
CPUSET="0-$(( CPUS - 1 ))"

echo "guard[$GUARD_ID]: mem=$MEM cpus=$CPUS (cores $CPUSET) timeout=$TIMEOUT floors: ram=$RAM_FLOOR disk=$DISK_FLOOR vram=$VRAM_FLOOR temp<${TEMP_MAX}C" | tee -a "$LOG" >&2
echo "guard[$GUARD_ID]: $*" >> "$LOG"

if command -v systemd-run >/dev/null && systemctl --user status >/dev/null 2>&1; then
    systemd-run --user --scope -q --unit="$SCOPE" -p MemoryMax="$MEM" -p MemorySwapMax=0 \
        -- taskset -c "$CPUSET" nice -n 10 ionice -c2 -n7 "$@" &
else
    echo "guard: no user systemd — running without the memory cgroup (docker limits + watchdog only)" >&2
    taskset -c "$CPUSET" nice -n 10 ionice -c2 -n7 "$@" &
fi
JOB=$!

kill_job() {   # $1 = reason
    echo "guard[$GUARD_ID]: KILLING job — $1" | tee -a "$LOG" >&2
    docker ps -q --filter "label=fisheye_guard=$GUARD_ID" 2>/dev/null | xargs -r docker kill >/dev/null 2>&1 || true
    systemctl --user kill --signal=SIGTERM "$SCOPE" 2>/dev/null || kill -TERM "$JOB" 2>/dev/null || true
    sleep 5
    systemctl --user kill --signal=SIGKILL "$SCOPE" 2>/dev/null || kill -KILL "$JOB" 2>/dev/null || true
    docker ps -q --filter "label=fisheye_guard=$GUARD_ID" 2>/dev/null | xargs -r docker kill >/dev/null 2>&1 || true
}
trap 'kill_job "interrupted"; exit 130' INT TERM

START=$(date +%s); KILLED=""
while kill -0 "$JOB" 2>/dev/null; do
    sleep "$INTERVAL"
    kill -0 "$JOB" 2>/dev/null || break
    ram=$(mem_available_b); disk=$(min_disk_free_b); vram=$(vram_free_b); temp=$(pkg_temp_c)
    echo "$(date +%H:%M:%S) $(sample_line "$vram")" >> "$LOG"
    reason=""
    [ "$ram" -ge "$RAM_FLOOR_B" ] || reason="available RAM $(human "$ram") < $(human "$RAM_FLOOR_B")"
    [ "$disk" -ge "$DISK_FLOOR_B" ] || reason="free disk $(human "$disk") < $(human "$DISK_FLOOR_B")"
    [ -z "$vram" ] || [ "$vram" -ge "$VRAM_FLOOR_B" ] || reason="free VRAM $(human "$vram") < $(human "$VRAM_FLOOR_B")"
    [ -z "$temp" ] || [ "$temp" -lt "$TEMP_MAX" ] || reason="CPU package ${temp}C >= ${TEMP_MAX}C"
    [ $(( $(date +%s) - START )) -le "$TIMEOUT_S" ] || reason="runtime exceeded $TIMEOUT"
    if [ -n "$reason" ]; then KILLED="$reason"; kill_job "$reason"; break; fi
done
wait "$JOB" 2>/dev/null && RC=0 || RC=$?
[ -z "$KILLED" ] || RC=137
echo "guard[$GUARD_ID]: done rc=$RC after $(( $(date +%s) - START ))s — $(sample_line)" | tee -a "$LOG" >&2
exit "$RC"
