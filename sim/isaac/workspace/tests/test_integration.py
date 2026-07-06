"""Tier 2 — Full integration tests (GPU required).

Launches px4_drone.py via Isaac Sim and validates MAVLink communication.

Run with:
    /isaac-sim/python.sh -m pytest /workspace/tests/test_integration.py -v
"""

import os
import signal
import subprocess
import time

import pytest

pymavlink = pytest.importorskip("pymavlink")
from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import common as mavlink2  # noqa: E402

ISAAC_PYTHON = "/isaac-sim/python.sh"
DRONE_SCRIPT = "/workspace/px4_drone.py"
STARTUP_TIMEOUT = 60  # Isaac Sim takes a while to load


@pytest.fixture(scope="module")
def sim_process():
    """Launch px4_drone.py under Isaac Sim and yield the process."""
    env = os.environ.copy()
    env["HEADLESS"] = "1"

    proc = subprocess.Popen(
        [ISAAC_PYTHON, DRONE_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for MAVLink heartbeat as a proxy for "sim is ready"
    deadline = time.monotonic() + STARTUP_TIMEOUT
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, _ = proc.communicate(timeout=5)
            pytest.fail(
                f"Sim process exited early (code {proc.returncode}).\n"
                f"Output:\n{stdout.decode(errors='replace')}"
            )
        try:
            conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550", dialect="common")
            msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
            conn.close()
            if msg is not None:
                ready = True
                break
        except Exception:
            time.sleep(2)

    if not ready:
        proc.kill()
        stdout, _ = proc.communicate(timeout=5)
        pytest.fail(
            f"No MAVLink heartbeat within {STARTUP_TIMEOUT}s.\n"
            f"Output:\n{stdout.decode(errors='replace')}"
        )

    yield proc

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


class TestIntegration:
    """Full-stack Isaac Sim + PX4 tests."""

    def test_simulation_running(self, sim_process):
        """Simulation process is alive."""
        assert sim_process.poll() is None, "Simulation exited prematurely"

    def test_mavlink_heartbeat_during_sim(self, sim_process):
        """MAVLink heartbeats are flowing while sim runs."""
        conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550", dialect="common")
        msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=15)
        conn.close()
        assert msg is not None, "No heartbeat during simulation"

    def test_arm_and_takeoff(self, sim_process):
        """Arm the drone and command takeoff to 5 m; verify altitude in NED."""
        conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550", dialect="common")
        conn.wait_heartbeat(timeout=15)

        # Arm
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            mavlink2.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=10)
        assert ack is not None, "No ACK for arm"

        # Takeoff to 5 m
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            mavlink2.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0,
            5.0,  # altitude in metres
        )
        ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=10)
        assert ack is not None, "No ACK for takeoff"

        # Wait for the drone to climb — check LOCAL_POSITION_NED
        deadline = time.monotonic() + 30
        reached = False
        while time.monotonic() < deadline:
            msg = conn.recv_match(
                type="LOCAL_POSITION_NED", blocking=True, timeout=2
            )
            if msg is not None and msg.z < -3.0:  # NED: z negative = up
                reached = True
                break

        conn.close()
        assert reached, "Drone did not reach 3 m altitude within 30 s"
