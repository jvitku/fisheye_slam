#!/usr/bin/env bash
set -euo pipefail

# Bridge PX4 MAVLink UDP ports to TCP so they can be SSH-tunneled.
#
# On the remote host, run this script, then SSH tunnel from your local machine:
#   ssh -L 15550:localhost:15550 -L 15540:localhost:15540 user@remote-host
#
# Then connect QGroundControl / MAVROS to localhost:15550 (TCP).
#
# UDP 14550 (GCS)      -> TCP 15550
# UDP 14540 (offboard) -> TCP 15540

echo "Bridging MAVLink UDP -> TCP..."
echo "  UDP 14550 (GCS)      -> TCP 15550"
echo "  UDP 14540 (offboard) -> TCP 15540"
echo ""
echo "SSH tunnel with:"
echo "  ssh -L 15550:localhost:15550 -L 15540:localhost:15540 user@\$(hostname)"
echo ""
echo "Press Ctrl+C to stop."

socat TCP-LISTEN:15550,fork,reuseaddr UDP4-RECVFROM:14550,fork &
PID_GCS=$!

socat TCP-LISTEN:15540,fork,reuseaddr UDP4-RECVFROM:14540,fork &
PID_OFF=$!

trap 'kill $PID_GCS $PID_OFF 2>/dev/null; exit' INT TERM

wait
