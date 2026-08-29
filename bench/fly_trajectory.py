"""Fly the benchmark trajectory in the Isaac/PX4-SITL sim over MAVLink offboard.

Deterministic and identical every run (same commanded setpoints; PX4 tracking
adds small run-to-run differences, which is why the ground truth is recorded
rather than assumed): take off, stream LOCAL_NED position + yaw setpoints at
20 Hz along a time-parametrized path, land.

Patterns (bench/README.md "The matrix"):
  slow_scan  lawn-mower survey at 0.6 m/s, 90-degree turns, yaw follows the
             leg direction — the mapping-flight case
  fast_yaw   figure-8 at 1.5 m/s with continuous yaw — the fast-rotation stress
             case (the fisheye FOV argument)

Connection: PX4 SITL sends its offboard link to UDP 14540 on the host
(sim/isaac/workspace/px4-rc.mavlink; the isaac container runs on the host
network), so we LISTEN there: udpin:0.0.0.0:14540. The GCS link (14550) is left
free for QGroundControl.

Usage (host):
    bench/record.sh rigs/pod3_oakdpro.yaml combo_day 150 &     # start recording
    uv run python -m bench.fly_trajectory --pattern slow_scan --duration 120

VERIFY-IN-SIM: offboard mode switch + arming sequence against this PX4 build
(v1.15.2) on first bring-up; the message set is standard MAVLink v2.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

# PX4 custom modes (px4_custom_mode.h)
PX4_MAIN_OFFBOARD = 6
PX4_MAIN_AUTO = 4
PX4_SUB_AUTO_LAND = 6
# SET_POSITION_TARGET_LOCAL_NED type_mask: use x, y, z, yaw; ignore the rest
TYPE_MASK_POS_YAW = (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 11)


def slow_scan(t: float, *, speed=0.6, leg=6.0, spacing=1.5, rows=5, alt=1.5):
    """Lawn-mower survey: legs along x, stepping in y; after `rows` rows the
    serpentine bounces back so the footprint stays leg x (rows-1)*spacing for
    any duration. Returns (x, y, z_up, yaw); yaw follows the motion."""
    leg_time = leg / speed
    step_time = spacing / speed
    period = leg_time + step_time
    i = int(t // period)
    tau = t - i * period
    cycle = 2 * (rows - 1)
    j = i % cycle
    row = j if j < rows else cycle - j
    row_next = (j + 1) if (j + 1) < rows else cycle - (j + 1)
    direction = 1.0 if i % 2 == 0 else -1.0
    x0 = 0.0 if direction > 0 else leg
    y0 = row * spacing
    if tau <= leg_time:                # along the leg
        x, y = x0 + direction * speed * tau, y0
        yaw = 0.0 if direction > 0 else math.pi
    else:                              # step to the next row
        sign = 1.0 if row_next > row else -1.0
        x, y = x0 + direction * leg, y0 + sign * speed * (tau - leg_time)
        yaw = sign * math.pi / 2
    return x, y, alt, yaw


def _lemniscate_unit_perimeter(n=20000) -> float:
    """Path length of x=sin(s), y=sin(s)cos(s) over one period (a = 1)."""
    s = np.linspace(0.0, 2 * np.pi, n)
    x, y = np.sin(s), np.sin(s) * np.cos(s)
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


_LEMNISCATE_P1 = _lemniscate_unit_perimeter()


def fast_yaw(t: float, *, speed=1.5, a=3.0, alt=1.5):
    """Figure-8 (lemniscate of Gerono) at mean speed `speed`; yaw follows the
    velocity direction, so the heading sweeps continuously."""
    period = a * _LEMNISCATE_P1 / speed
    w = 2 * math.pi / period
    x = a * math.sin(w * t)
    y = a * math.sin(w * t) * math.cos(w * t)
    vx = a * w * math.cos(w * t)
    vy = a * w * math.cos(2 * w * t)
    return x, y, alt, math.atan2(vy, vx)


PATTERNS = {"slow_scan": slow_scan, "fast_yaw": fast_yaw}


def setpoints(pattern: str, duration: float, rate: float = 20.0, ramp: float = 4.0):
    """Yield (t, x, y, z_up, yaw) for the whole flight.

    The path clock is time-warped for the first `ramp` seconds so the commanded
    speed rises linearly from 0 to nominal (tau = t^2 / 2 ramp, then t - ramp/2):
    no jerk at takeoff, never faster than the pattern's nominal speed.
    """
    f = PATTERNS[pattern]
    n = int(duration * rate)
    for k in range(n):
        t = k / rate
        tau = t * t / (2.0 * ramp) if t < ramp else t - ramp / 2.0
        x, y, z, yaw = f(tau)
        yield t, x, y, z, yaw


class Offboard:
    """Minimal PX4 offboard driver over pymavlink."""

    def __init__(self, url: str):
        from pymavlink import mavutil   # imported here: optional dependency
        self.mavutil = mavutil
        self.conn = mavutil.mavlink_connection(url, dialect="common", source_system=255)
        print(f"waiting for PX4 heartbeat on {url} ...")
        self.conn.wait_heartbeat(timeout=60)
        self.sys, self.comp = self.conn.target_system, self.conn.target_component
        print(f"connected: system {self.sys} component {self.comp}")

    def send_setpoint(self, x, y, z_up, yaw):
        self.conn.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF, self.sys, self.comp,
            self.mavutil.mavlink.MAV_FRAME_LOCAL_NED, TYPE_MASK_POS_YAW,
            float(x), float(-y), float(-z_up),      # ENU/FLU path -> NED
            0, 0, 0, 0, 0, 0, float(-yaw), 0)

    def set_mode(self, main: int, sub: int = 0):
        custom = (main << 16) | (sub << 24)
        self.conn.mav.command_long_send(
            self.sys, self.comp, self.mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            self.mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, custom, 0, 0, 0, 0, 0)

    def arm(self, arm: bool = True):
        self.conn.mav.command_long_send(
            self.sys, self.comp, self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1.0 if arm else 0.0, 0, 0, 0, 0, 0, 0)

    def local_position(self):
        m = self.conn.recv_match(type="LOCAL_POSITION_NED", blocking=False)
        return None if m is None else (m.x, -m.y, -m.z)

    def fly(self, pattern: str, duration: float, rate: float = 20.0, takeoff_time: float = 6.0):
        x0, y0, z0, yaw0 = PATTERNS[pattern](0.0)
        dt = 1.0 / rate
        # PX4 requires a setpoint stream BEFORE offboard is accepted
        for _ in range(int(2 * rate)):
            self.send_setpoint(x0, y0, 0.0, yaw0)
            time.sleep(dt)
        self.set_mode(PX4_MAIN_OFFBOARD)
        self.arm(True)
        print(f"offboard + armed; takeoff to {z0} m")
        t_end = time.monotonic() + takeoff_time
        while time.monotonic() < t_end:
            self.send_setpoint(x0, y0, z0, yaw0)
            time.sleep(dt)
        print(f"flying '{pattern}' for {duration:.0f} s")
        t_start = time.monotonic()
        for t, x, y, z, yaw in setpoints(pattern, duration, rate):
            self.send_setpoint(x, y, z, yaw)
            pos = self.local_position()
            if pos is not None and int(t) % 10 == 0 and abs(t - int(t)) < dt / 2:
                print(f"  t={t:5.1f}s  cmd=({x:5.2f},{y:5.2f},{z:4.2f})  pos=({pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:4.2f})")
            time.sleep(max(0.0, t_start + t + dt - time.monotonic()))
        print("landing")
        self.set_mode(PX4_MAIN_AUTO, PX4_SUB_AUTO_LAND)
        time.sleep(8.0)
        self.arm(False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", choices=sorted(PATTERNS), default="slow_scan")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds on the path")
    ap.add_argument("--rate", type=float, default=20.0, help="setpoint rate [Hz]")
    ap.add_argument("--url", default="udpin:0.0.0.0:14540", help="PX4 offboard link")
    ap.add_argument("--dry-run", action="store_true", help="print the path summary, no MAVLink")
    args = ap.parse_args(argv)

    if args.dry_run:
        pts = np.array([[x, y, z, yaw] for _, x, y, z, yaw in setpoints(args.pattern, args.duration, args.rate)])
        d = np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1)
        print(f"{args.pattern}: {len(pts)} setpoints, path {d.sum():.1f} m, "
              f"mean speed {d.sum() / args.duration:.2f} m/s, bbox x[{pts[:, 0].min():.1f},{pts[:, 0].max():.1f}] "
              f"y[{pts[:, 1].min():.1f},{pts[:, 1].max():.1f}] z={pts[0, 2]:.1f}")
        return 0
    Offboard(args.url).fly(args.pattern, args.duration, args.rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
