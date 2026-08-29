"""Extract the sim ground truth (/uav1/ground_truth, nav_msgs/Odometry) to TUM.

By default the pose is the drone BODY, which is what Pegasus reports. A VIO
estimate is the pose of the IMU it runs on; for a pod mounted off body
center the two differ by a lever arm that the SE3 alignment in
bench/evaluate.py cannot absorb (0.12 m above center x 15 deg of pitch is
~3 cm — the size of the ATE numbers being compared). So pass --rig (and --pod
for composite rigs) to express the ground truth at that pod's IMU frame:

    T_world_pod = T_world_body . T_body_pod          (T_body_pod from the rig yaml)

Usage:
    python -m bench.gt_extract bag.bag gt.txt
        [--rig rigs/pod3_oakdpro.yaml --pod oakd] [--topic /uav1/ground_truth]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from bench.rigdef import R_to_quat_xyzw, load_rig, mount_to_T, pod_imu, quat_xyzw_to_R

DEFAULT_TOPIC = "/uav1/ground_truth"


def body_to_sensor_T(rig: dict, ns: str | None = None) -> np.ndarray:
    """4x4 transform of a pod's IMU frame in the drone body frame."""
    return mount_to_T(pod_imu(rig, ns)["body_mount"])


def read_odometry(bag: str, topic: str = DEFAULT_TOPIC) -> np.ndarray:
    """Odometry messages on `topic` -> (N, 8) [t x y z qx qy qz qw], header time, sorted."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    rows = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise ValueError(f"{bag}: no messages on {topic}")
        for conn, _, raw in reader.messages(connections=conns):
            msg = ts.deserialize_ros1(raw, conn.msgtype)
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            rows.append([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
    arr = np.asarray(rows, dtype=np.float64)
    return arr[np.argsort(arr[:, 0], kind="stable")]


def transform_poses(poses: np.ndarray, T_bs: np.ndarray) -> np.ndarray:
    """Re-express body poses (N, 8) at a rigidly attached sensor frame."""
    R_bs, t_bs = T_bs[:3, :3], T_bs[:3, 3]
    out = poses.copy()
    for i, row in enumerate(poses):
        R_wb = quat_xyzw_to_R(row[4:8])
        out[i, 1:4] = row[1:4] + R_wb @ t_bs
        out[i, 4:8] = R_to_quat_xyzw(R_wb @ R_bs)
    return out


def extract(bag: str, out: str, topic: str = DEFAULT_TOPIC,
            T_body_sensor: np.ndarray | None = None) -> int:
    poses = read_odometry(bag, topic)
    if T_body_sensor is not None:
        poses = transform_poses(poses, T_body_sensor)
    np.savetxt(out, poses, fmt="%.9f",
               header="t x y z qx qy qz qw (sim ground truth)")
    return len(poses)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--topic", default=DEFAULT_TOPIC)
    ap.add_argument("--rig", help="rig yaml: express GT at the pod IMU instead of the body")
    ap.add_argument("--pod", help="member ns for composite rigs (e.g. oakd)")
    args = ap.parse_args(argv)

    T = None
    if args.rig:
        T = body_to_sensor_T(load_rig(args.rig), args.pod)
    elif args.pod:
        raise SystemExit("--pod needs --rig")
    n = extract(args.bag, args.out, args.topic, T)
    frame = f"pod IMU ({args.pod or 'single pod'})" if T is not None else "drone body"
    print(f"{n} poses ({frame}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
