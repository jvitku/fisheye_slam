"""Build the "flight bag" (IMU + ground truth) that frames2bag merges with
rendered frames — from minisim's imu.csv + gt.tum (or a PX4 SITL flight's).

    python -m bench.minisim.flightbag out_frames/ rigs/skydio3.yaml flight.bag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rosbags.rosbag1 import Writer                      # noqa: E402
from rosbags.typesys import Stores, get_typestore       # noqa: E402

from bench.gt_extract import DEFAULT_TOPIC as GT_TOPIC  # noqa: E402
from bench.rigdef import load_rig                       # noqa: E402

TS = get_typestore(Stores.ROS1_NOETIC)
NS = 1_000_000_000


def _header(t_ns: int, frame_id: str):
    Header = TS.types["std_msgs/msg/Header"]
    Time = TS.types["builtin_interfaces/msg/Time"]
    kw = {"stamp": Time(sec=t_ns // NS, nanosec=t_ns % NS), "frame_id": frame_id}
    if "seq" in Header.__dataclass_fields__:
        kw["seq"] = 0
    return Header(**kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames_dir"); ap.add_argument("rig"); ap.add_argument("out_bag")
    a = ap.parse_args(argv)
    d = Path(a.frames_dir)
    rig = load_rig(a.rig)
    imu_topic = rig["imus"][0]["topic"]

    imu = np.loadtxt(d / "imu.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(d / "gt.tum")
    V3 = TS.types["geometry_msgs/msg/Vector3"]
    Q = TS.types["geometry_msgs/msg/Quaternion"]
    P = TS.types["geometry_msgs/msg/Point"]
    Imu = TS.types["sensor_msgs/msg/Imu"]
    Odo = TS.types["nav_msgs/msg/Odometry"]
    Pose = TS.types["geometry_msgs/msg/Pose"]
    PoseC = TS.types["geometry_msgs/msg/PoseWithCovariance"]
    Twist = TS.types["geometry_msgs/msg/Twist"]
    TwistC = TS.types["geometry_msgs/msg/TwistWithCovariance"]
    z9, z36 = np.zeros(9), np.zeros(36)

    def imu_raw(row):
        t_ns = int(row[0])
        m = Imu(header=_header(t_ns, "imu"), orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=z9,
                angular_velocity=V3(x=row[1], y=row[2], z=row[3]), angular_velocity_covariance=z9,
                linear_acceleration=V3(x=row[4], y=row[5], z=row[6]), linear_acceleration_covariance=z9)
        return t_ns, TS.serialize_ros1(m, "sensor_msgs/msg/Imu")

    def gt_raw(row):
        t_ns = int(round(row[0] * NS))
        pose = Pose(position=P(x=row[1], y=row[2], z=row[3]),
                    orientation=Q(x=row[4], y=row[5], z=row[6], w=row[7]))
        m = Odo(header=_header(t_ns, "map"), child_frame_id="base_link",
                pose=PoseC(pose=pose, covariance=z36),
                twist=TwistC(twist=Twist(linear=V3(x=0.0, y=0.0, z=0.0),
                                         angular=V3(x=0.0, y=0.0, z=0.0)), covariance=z36))
        return t_ns, TS.serialize_ros1(m, "nav_msgs/msg/Odometry")

    out = Path(a.out_bag); out.unlink(missing_ok=True)
    with Writer(out) as w:
        ci = w.add_connection(imu_topic, "sensor_msgs/msg/Imu", typestore=TS)
        cg = w.add_connection(GT_TOPIC, "nav_msgs/msg/Odometry", typestore=TS)
        events = [(int(r[0]), 0, r) for r in imu] + [(int(round(r[0] * NS)), 1, r) for r in gt]
        events.sort(key=lambda e: (e[0], e[1]))
        n = 0
        for t_ns, kind, row in events:
            if kind == 0:
                t, raw = imu_raw(row)
                w.write(ci, t, raw)
            else:
                t, raw = gt_raw(row)
                w.write(cg, t, raw)
            n += 1
    print(f"{out}: {n} messages ({len(imu)} imu @ {imu_topic}, {len(gt)} gt @ {GT_TOPIC})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
