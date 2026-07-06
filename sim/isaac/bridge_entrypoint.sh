#!/bin/bash

# Watchdog entrypoint for ros1_bridge.
# Automatically (re)starts the bridge when roscore becomes available and
# kills it when roscore disappears, so a roscore restart on the host is
# handled without touching the container.

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Source ros1_bridge workspace (ROS1 upstream libs are in system paths)
source /bridge_ws/install/local_setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "=== ros1_bridge watchdog ==="
echo "  ROS_MASTER_URI: ${ROS_MASTER_URI}"
echo "  RMW:            ${RMW_IMPLEMENTATION}"
echo ""

BRIDGE_PID=0
POLL_INTERVAL=5
BRIDGE_LOG="/tmp/bridge_output.log"
HEALTH_CHECK_DELAY=30
MAX_RETRIES=3
RETRY_COUNT=0

# --- helpers ---------------------------------------------------------------

roscore_reachable() {
    python3 -c "
import xmlrpc.client, socket
try:
    proxy = xmlrpc.client.ServerProxy('${ROS_MASTER_URI}')
    proxy.getSystemState('/bridge_watchdog')
    exit(0)
except (ConnectionRefusedError, socket.error, xmlrpc.client.Fault):
    exit(1)
" 2>/dev/null
}

# Interruptible sleep — allows SIGTERM/SIGINT to break out immediately.
isleep() {
    sleep "$1" &
    wait $!
}

# --- signal handling -------------------------------------------------------

cleanup() {
    echo "[watchdog] Caught signal, shutting down..."
    if [ "$BRIDGE_PID" -ne 0 ] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill "$BRIDGE_PID" 2>/dev/null
        wait "$BRIDGE_PID" 2>/dev/null
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT

# --- main loop -------------------------------------------------------------

while true; do
    # Wait for roscore to become reachable
    echo "[watchdog] Waiting for roscore at ${ROS_MASTER_URI} ..."
    while ! roscore_reachable; do
        isleep "$POLL_INTERVAL"
    done
    echo "[watchdog] roscore is reachable, starting bridge."

    # Start bridge in background, capturing output to log for health checks
    > "$BRIDGE_LOG"
    ros2 run ros1_bridge dynamic_bridge --bridge-all-topics 2>&1 \
        | tee "$BRIDGE_LOG" \
        | grep -v -e "failed to create.*bridge for topic '/rosout'" \
                  -e "check the list of supported pairs" &
    BRIDGE_PID=$!
    echo "[watchdog] Bridge started (PID ${BRIDGE_PID})."

    SECONDS_SINCE_START=0
    HEALTH_CHECKED=false

    # Monitor: poll roscore, bridge health, and 2to1 bridge creation
    while true; do
        isleep "$POLL_INTERVAL"
        SECONDS_SINCE_START=$((SECONDS_SINCE_START + POLL_INTERVAL))

        # Bridge process died on its own
        if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
            echo "[watchdog] Bridge process exited, will restart."
            wait "$BRIDGE_PID" 2>/dev/null
            BRIDGE_PID=0
            break
        fi

        # Roscore unreachable — kill stale bridge
        if ! roscore_reachable; then
            echo "[watchdog] roscore unreachable, killing stale bridge (PID ${BRIDGE_PID})."
            kill "$BRIDGE_PID" 2>/dev/null
            wait "$BRIDGE_PID" 2>/dev/null
            BRIDGE_PID=0
            break
        fi

        # Health check: verify 2to1 bridges were created
        if [ "$HEALTH_CHECKED" = false ] && [ "$SECONDS_SINCE_START" -ge "$HEALTH_CHECK_DELAY" ]; then
            HEALTH_CHECKED=true
            if grep -q "created 2to1 bridge" "$BRIDGE_LOG"; then
                echo "[watchdog] Health check passed: 2to1 bridges detected."
                RETRY_COUNT=0
            else
                RETRY_COUNT=$((RETRY_COUNT + 1))
                if [ "$RETRY_COUNT" -le "$MAX_RETRIES" ]; then
                    echo "[watchdog] No 2to1 bridges after ${HEALTH_CHECK_DELAY}s (attempt ${RETRY_COUNT}/${MAX_RETRIES}), restarting bridge..."
                    kill "$BRIDGE_PID" 2>/dev/null
                    wait "$BRIDGE_PID" 2>/dev/null
                    BRIDGE_PID=0
                    break
                else
                    echo "[watchdog] No 2to1 bridges after ${MAX_RETRIES} retries, giving up health checks."
                fi
            fi
        fi
    done
done
