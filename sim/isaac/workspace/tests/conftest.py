"""Shared fixtures for PX4 SITL and integration tests."""

import os
import signal
import socket
import struct
import subprocess
import threading
import time

import pytest
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2

PX4_DIR = "/opt/PX4-Autopilot"
PX4_BIN = f"{PX4_DIR}/build/px4_sitl_default/bin/px4"
PX4_ROMFS = f"{PX4_DIR}/ROMFS/px4fmu_common"
PX4_RCS = f"{PX4_ROMFS}/init.d-posix/rcS"


class MockSimulator:
    """Minimal TCP server on port 4560 that feeds HIL sensor data to PX4.

    PX4 SITL connects to a simulator on TCP 4560 and expects HIL_SENSOR
    and HIL_GPS messages at ~250 Hz to proceed with initialization.
    """

    def __init__(self, port: int = 4560):
        self.port = port
        self._stop = threading.Event()
        self._server_sock = None
        self._thread = None

    def start(self):
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("0.0.0.0", self.port))
        self._server_sock.listen(1)
        self._server_sock.settimeout(1.0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._server_sock:
            self._server_sock.close()

    def _run(self):
        conn = None
        try:
            # Wait for PX4 to connect
            while not self._stop.is_set():
                try:
                    conn, _ = self._server_sock.accept()
                    break
                except socket.timeout:
                    continue

            if conn is None:
                return

            # Create a MAVLink encoder
            mav = mavlink2.MAVLink(None, srcSystem=1, srcComponent=1)
            t0 = time.monotonic()

            while not self._stop.is_set():
                now_us = int((time.monotonic() - t0) * 1e6)

                # HIL_SENSOR: accelerometer reads ~9.81 m/s² on Z,
                # pressure at sea level, fields_updated=0x1FFF (all)
                sensor_msg = mav.hil_sensor_encode(
                    time_usec=now_us,
                    xacc=0.0, yacc=0.0, zacc=-9.81,
                    xgyro=0.0, ygyro=0.0, zgyro=0.0,
                    xmag=0.2, ymag=0.0, zmag=0.4,
                    abs_pressure=1013.25,
                    diff_pressure=0.0,
                    pressure_alt=0.0,
                    temperature=25.0,
                    fields_updated=0x1FFF,
                )

                # HIL_GPS: stationary at lat=0, lon=0, alt=0
                gps_msg = mav.hil_gps_encode(
                    time_usec=now_us,
                    fix_type=3,  # 3D fix
                    lat=0, lon=0, alt=0,
                    eph=100, epv=100,
                    vel=0, vn=0, ve=0, vd=0,
                    cog=0,
                    satellites_visible=10,
                )

                try:
                    conn.sendall(sensor_msg.pack(mav))
                    conn.sendall(gps_msg.pack(mav))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break

                # ~250 Hz
                time.sleep(0.004)

        finally:
            if conn:
                conn.close()


def _wait_for_udp_heartbeat(port: int, timeout: float = 20.0) -> bool:
    """Block until a MAVLink heartbeat is received on a UDP port."""
    try:
        conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{port}", dialect="common")
        msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout)
        conn.close()
        return msg is not None
    except Exception:
        return False


@pytest.fixture(scope="session")
def px4_sitl():
    """Start mock simulator + PX4 SITL; yield the PX4 process; clean up both."""
    mock = MockSimulator(port=4560)
    mock.start()

    env = os.environ.copy()
    env["PX4_SIM_MODEL"] = "gazebo-classic_iris"

    proc = subprocess.Popen(
        [PX4_BIN, PX4_ROMFS, "-s", PX4_RCS, "-d"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for PX4 to be ready — PX4 sends heartbeats TO port 14550 (GCS)
    if not _wait_for_udp_heartbeat(14550, timeout=30.0):
        proc.kill()
        stdout, _ = proc.communicate(timeout=5)
        mock.stop()
        pytest.fail(
            f"PX4 did not produce heartbeats within 30s.\n"
            f"Exit code: {proc.returncode}\n"
            f"Output:\n{stdout.decode(errors='replace')}"
        )

    yield proc

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    mock.stop()
