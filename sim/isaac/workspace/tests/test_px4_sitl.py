"""Tier 1 — PX4 SITL-only tests (no GPU required).

PX4 MAVLink port layout (SITL default):
  - GCS:      PX4 binds 18570, sends TO 14550
  - Offboard: PX4 binds 14580, sends TO 14540

Tests listen on the remote (destination) ports: 14550 and 14540.

Run with:
    python3 -m pytest /workspace/tests/test_px4_sitl.py -v
"""

import socket
import time

import pytest

pymavlink = pytest.importorskip("pymavlink")
from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import common as mavlink2  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recv_heartbeat(udp_port: int, timeout: float = 10.0):
    """Listen for a MAVLink heartbeat on a UDP port."""
    conn = mavutil.mavlink_connection(
        f"udpin:0.0.0.0:{udp_port}", dialect="common"
    )
    msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout)
    conn.close()
    return msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPX4SITL:
    """Basic PX4 SITL health checks."""

    def test_px4_starts(self, px4_sitl):
        """PX4 process is alive."""
        assert px4_sitl.poll() is None, "PX4 process exited prematurely"

    def test_tcp_4560_connected(self, px4_sitl):
        """Mock simulator TCP port 4560 is serving (PX4 connects here)."""
        with socket.create_connection(("127.0.0.1", 4560), timeout=5):
            pass  # connection succeeded

    def test_gcs_heartbeat(self, px4_sitl):
        """GCS MAVLink heartbeat received on UDP 14550."""
        msg = _recv_heartbeat(14550, timeout=15)
        assert msg is not None, "No heartbeat on UDP 14550 (GCS)"

    def test_offboard_heartbeat(self, px4_sitl):
        """Offboard MAVLink heartbeat received on UDP 14540."""
        msg = _recv_heartbeat(14540, timeout=15)
        assert msg is not None, "No heartbeat on UDP 14540 (offboard)"

    def test_px4_reports_vehicle_type(self, px4_sitl):
        """Heartbeat indicates MAV_TYPE_QUADROTOR (2)."""
        msg = _recv_heartbeat(14550, timeout=15)
        assert msg is not None, "No heartbeat received"
        assert msg.type == mavlink2.MAV_TYPE_QUADROTOR, (
            f"Expected MAV_TYPE_QUADROTOR (2), got {msg.type}"
        )

    def test_arm_accepted(self, px4_sitl):
        """Arm command gets a COMMAND_ACK response."""
        conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550", dialect="common")
        conn.wait_heartbeat(timeout=15)

        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            mavlink2.MAV_CMD_COMPONENT_ARM_DISARM,
            0,   # confirmation
            1,   # arm
            0, 0, 0, 0, 0, 0,
        )

        ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=10)
        conn.close()
        assert ack is not None, "No COMMAND_ACK received for arm command"
        assert ack.command == mavlink2.MAV_CMD_COMPONENT_ARM_DISARM
