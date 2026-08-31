"""Fly a PX4 SITL (SIH quadrotor, lockstep) mission and extract the flown
trajectory as a TUM file for the minisim renderers.

The same flown trajectory is reused for every rig and condition of a scene, so
comparisons are over identical, dynamically-real paths — real flight stack,
real quad dynamics, controller-induced motion (not analytic splines).

    python -m bench.minisim.px4_fly indoor trajs/indoor_px4.tum [--px4 ~/tools/px4]

Paths (offboard position setpoints, 10 Hz):
    indoor  figure-8, +-3 x +-1.9 m, alt 1.5-2.0 m, yaw sweeps  (fits the room)
    forest  ellipse 13 x 7 m at ~2.2 m/s, alt 2-2.6 m
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

R_EN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # NED vec -> ENU vec
R_FL = np.diag([1.0, -1.0, -1.0])                                        # FLU -> FRD (and back)


def path_indoor(t: float):
    ramp = min(t / 8.0, 1.0)
    x = 3.0 * np.sin(2 * np.pi * t / 40.0) * ramp
    y = 1.9 * np.sin(2 * np.pi * t / 23.0 + 1.0) * ramp
    z = 1.7 + 0.35 * np.sin(2 * np.pi * t / 17.0) * ramp
    yaw = 55.0 * np.sin(2 * np.pi * t / 31.0) * ramp
    for t0 in (45.0, 80.0):
        if t0 <= t < t0 + 4.0:
            yaw += np.degrees(np.pi * (1 - np.cos(np.pi * (t - t0) / 4.0)) / 2.0 * 1.4)
        elif t >= t0 + 4.0:
            yaw += np.degrees(1.4 * np.pi)
    return x, y, z, yaw


def path_forest(t: float):
    ramp = min(max(t - 1.0, 0.0) / 7.0, 1.0)
    th = 2 * np.pi * (t * ramp * 0.5) / 55.0
    x = 13.0 * np.sin(th)
    y = -7.0 * np.cos(th) + 7.0
    z = 2.2 + 0.4 * np.sin(2 * np.pi * t / 11.0) * ramp
    yaw = np.degrees(np.arctan2(7.0 * np.sin(th), 13.0 * np.cos(th))) if ramp > 0 else 0.0
    yaw += 20.0 * np.sin(2 * np.pi * t / 8.0) * ramp
    return x, y, z, yaw


PATHS = {"indoor": path_indoor, "forest": path_forest}


async def fly(scene: str, duration: float):
    from mavsdk import System
    from mavsdk.offboard import OffboardError, PositionNedYaw
    drone = System()
    await drone.connect(system_address="udp://:14540")
    print("waiting for connection...", flush=True)
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    async for h in drone.telemetry.health():
        if h.is_global_position_ok and h.is_home_position_ok:
            break
    print("armed + offboard", flush=True)
    await drone.action.arm()
    path = PATHS[scene]
    x0, y0, z0, yaw0 = path(0.0)
    await drone.offboard.set_position_ned(PositionNedYaw(y0, x0, -z0, -yaw0 + 90.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"offboard failed: {e}", flush=True)
        raise
    t_start = time.time()
    while True:
        t = time.time() - t_start
        if t > duration:
            break
        x, y, z, yaw = path(t)
        # world ENU (x east-ish = our x) -> NED: north = y? Use ENU->NED: n=y, e=x, d=-z
        await drone.offboard.set_position_ned(PositionNedYaw(y, x, -z, -yaw + 90.0))
        await asyncio.sleep(0.1)
    await drone.offboard.stop()
    await drone.action.land()
    for _ in range(60):
        armed = None
        async for a in drone.telemetry.armed():
            armed = a
            break
        if not armed:
            break
        await asyncio.sleep(1.0)
    print("flight done", flush=True)


def extract_ulog(px4_dir: Path, out_tum: Path, t_min_s: float = 0.0):
    """Newest .ulg -> TUM (world ENU-ish frame, FLU body), position+attitude GT."""
    from pyulog import ULog
    from scipy.spatial.transform import Rotation, Slerp
    logs = sorted(glob.glob(str(px4_dir / "build/px4_sitl_default/rootfs/log/*/*.ulg")),
                  key=os.path.getmtime)
    if not logs:
        logs = sorted(glob.glob(str(px4_dir / "build/px4_sitl_default/**/*.ulg"), recursive=True),
                      key=os.path.getmtime)
    ulog = ULog(logs[-1])
    print(f"ulog: {logs[-1]}", flush=True)
    pos = ulog.get_dataset("vehicle_local_position_groundtruth").data
    att = ulog.get_dataset("vehicle_attitude_groundtruth").data
    tq = att["timestamp"] / 1e6
    q_wxyz = np.stack([att["q[0]"], att["q[1]"], att["q[2]"], att["q[3]"]], 1)
    rot_ned_frd = Rotation.from_quat(np.roll(q_wxyz, -1, axis=1))     # -> xyzw
    slerp = Slerp(tq, rot_ned_frd)
    tp = pos["timestamp"] / 1e6
    keep = (tp >= tq[0]) & (tp <= tq[-1]) & (tp >= t_min_s)
    tp = tp[keep]
    p_ned = np.stack([pos["x"], pos["y"], pos["z"]], 1)[keep]
    R_nf = slerp(tp).as_matrix()
    p_enu = (R_EN @ p_ned.T).T
    R_wb = np.einsum("ij,njk,kl->nil", R_EN, R_nf, R_FL)              # world ENU <- body FLU
    q = Rotation.from_matrix(R_wb).as_quat()
    out_tum.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tum, "w") as f:
        for i in range(len(tp)):
            f.write(f"{tp[i]:.6f} {p_enu[i,0]:.6f} {p_enu[i,1]:.6f} {p_enu[i,2]:.6f} "
                    f"{q[i,0]:.9f} {q[i,1]:.9f} {q[i,2]:.9f} {q[i,3]:.9f}\n")
    print(f"{out_tum}: {len(tp)} poses, {tp[-1]-tp[0]:.1f} s, "
          f"alt {-p_ned[:,2].min():.1f}..{-p_ned[:,2].max():.1f} m", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", choices=list(PATHS))
    ap.add_argument("out_tum")
    ap.add_argument("--px4", default=os.path.expanduser("~/tools/px4"))
    ap.add_argument("--duration", type=float, default=110.0)
    a = ap.parse_args(argv)
    px4 = Path(a.px4)
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env.update(PX4_SYS_AUTOSTART="10040", PX4_SIM_MODEL="sihsim_quadx", HEADLESS="1")
    proc = subprocess.Popen(["make", "px4_sitl", "sihsim_quadx"], cwd=px4, env=env,
                            stdin=subprocess.PIPE, stdout=open("/tmp/px4_sitl.log", "wb"),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    try:
        time.sleep(12)                                   # boot + EKF settle
        asyncio.run(fly(a.scene, a.duration))
        time.sleep(2)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    extract_ulog(px4, Path(a.out_tum))
    return 0


if __name__ == "__main__":
    sys.exit(main())
