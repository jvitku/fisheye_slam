"""PyCuVSLAM stereo(+IMU) tracking on a bag-contract ROS1 bag -> TUM + depth dump.

Feeds the synced stereo pair (and IMU, when supported by the installed
pycuvslam build) to cuVSLAM and writes:
    <out>/est.tum            time x y z qx qy qz qw
    <out>/depth/NNNNNN.npy   (optional --dump-depth) depth [m] + poses.txt —
                             the nvblox fuser input (see run_cuvslam.sh)

Written against the PyCuVSLAM EuRoC example (github.com/NVlabs/PyCuVSLAM)
— the API surface moved between releases; VERIFY-ON-FIRST-RUN against the
pinned wheel (docs/oak_d_pro_slam.md §6).

Usage (inside 3dfe/cuvslam, --gpus all):
    python3 cuvslam_track.py input.bag out/ --rig /rigs/oakdpro.yaml [--dump-depth]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

import cuvslam

TS = get_typestore(Stores.ROS1_NOETIC)

LEFT, RIGHT = "/uav1/cam0/color/image_raw", "/uav1/cam1/color/image_raw"
DEPTH = "/uav1/cam0/depth/image_raw"
IMU = "/uav1/sensor_pod/imu"


def make_rig(rig_yaml: str):
    """rigs/oakdpro.yaml -> cuvslam.Rig (pinhole stereo)."""
    rig = yaml.safe_load(open(rig_yaml))
    cams = []
    for cam in rig["cameras"]:
        intr = cam["intrinsics"]
        c = cuvslam.Camera()
        c.size = tuple(cam["resolution"])
        c.focal = (intr["fx"], intr["fy"])
        c.principal = (intr["cx"], intr["cy"])
        # camera pose in the rig frame: FLU mount -> optical (z fwd, x right)
        y = cam["mount"]["position"][1]
        c.rig_from_camera = cuvslam.Pose(
            rotation=[0.5, -0.5, 0.5, -0.5],   # FLU->optical as quaternion xyzw
            translation=[-y, 0.0, 0.0],        # optical x = -FLU y (baseline)
        )
        cams.append(c)
    r = cuvslam.Rig()
    r.cameras = cams
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--dump-depth", action="store_true",
                    help="also dump depth frames + tracked poses for nvblox")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = cuvslam.Tracker.OdometryConfig()
    cfg.async_sba = False          # deterministic offline replay
    tracker = cuvslam.Tracker(make_rig(args.rig), cfg)

    tum = (args.out / "est.tum").open("w")
    depth_dir = args.out / "depth"
    poses_txt = None
    if args.dump_depth:
        depth_dir.mkdir(exist_ok=True)
        poses_txt = (depth_dir / "poses.txt").open("w")

    pending: dict[int, dict[str, np.ndarray]] = {}
    n_frames = n_valid = 0

    with Reader(args.bag) as reader:
        conns = [c for c in reader.connections if c.topic in (LEFT, RIGHT, DEPTH, IMU)]
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = TS.deserialize_ros1(raw, conn.msgtype)
            if conn.topic == IMU:
                # IMU support depends on the pycuvslam build; skip if absent.
                if hasattr(tracker, "register_imu_measurement"):
                    m = cuvslam.ImuMeasurement()
                    m.timestamp_ns = t_ns
                    g, a = msg.angular_velocity, msg.linear_acceleration
                    m.angular_velocities = [g.x, g.y, g.z]
                    m.linear_accelerations = [a.x, a.y, a.z]
                    tracker.register_imu_measurement(0, m)
                continue

            stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if conn.topic == DEPTH:
                if args.dump_depth:
                    d = np.frombuffer(msg.data, dtype=np.float32).reshape(
                        msg.height, msg.width)
                    pending.setdefault(stamp_ns, {})["depth"] = d
                continue
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            pending.setdefault(stamp_ns, {})["l" if conn.topic == LEFT else "r"] = img

            group = pending[stamp_ns]
            if "l" not in group or "r" not in group:
                continue
            odom = tracker.track(stamp_ns, (group["l"], group["r"]))
            n_frames += 1
            if getattr(odom, "is_valid", True):
                n_valid += 1
                t = stamp_ns / 1e9
                x, y, z = odom.pose.translation
                qx, qy, qz, qw = odom.pose.rotation
                tum.write(f"{t:.6f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                if args.dump_depth and "depth" in group:
                    np.save(depth_dir / f"{n_frames:06d}.npy", group["depth"])
                    poses_txt.write(
                        f"{n_frames:06d} {t:.6f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
            del pending[stamp_ns]

    tum.close()
    if poses_txt:
        poses_txt.close()
    print(f"tracked {n_valid}/{n_frames} stereo frames -> {args.out / 'est.tum'}")


if __name__ == "__main__":
    main()
