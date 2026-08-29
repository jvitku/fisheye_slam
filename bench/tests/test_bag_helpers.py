"""Shared synthetic-bag builders for the split / ground-truth tests."""

import numpy as np
import pytest

pytest.importorskip("rosbags")

from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS1_NOETIC)
NS = 1_000_000_000


def header(t_ns, frame="uav1"):
    Header = TS.types["std_msgs/msg/Header"]
    Time = TS.types["builtin_interfaces/msg/Time"]
    kw = {"stamp": Time(sec=t_ns // NS, nanosec=t_ns % NS), "frame_id": frame}
    if "seq" in Header.__dataclass_fields__:
        kw["seq"] = 0
    return Header(**kw)


def odometry(t_ns, p, q):
    Point = TS.types["geometry_msgs/msg/Point"]
    Quat = TS.types["geometry_msgs/msg/Quaternion"]
    Vec3 = TS.types["geometry_msgs/msg/Vector3"]
    Pose = TS.types["geometry_msgs/msg/Pose"]
    Twist = TS.types["geometry_msgs/msg/Twist"]
    return TS.types["nav_msgs/msg/Odometry"](
        header=header(t_ns, "world"), child_frame_id="uav1",
        pose=TS.types["geometry_msgs/msg/PoseWithCovariance"](
            pose=Pose(position=Point(x=p[0], y=p[1], z=p[2]),
                      orientation=Quat(x=q[0], y=q[1], z=q[2], w=q[3])),
            covariance=np.zeros(36)),
        twist=TS.types["geometry_msgs/msg/TwistWithCovariance"](
            twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
            covariance=np.zeros(36)),
    )


def image(t_ns, value, encoding="mono8", frame="cam"):
    if encoding == "32FC1":
        data = np.full(16, float(value), dtype=np.float32).view(np.uint8)
        step = 16
    else:
        data = np.full(16, value, dtype=np.uint8)
        step = 4
    return TS.types["sensor_msgs/msg/Image"](
        header=header(t_ns, frame), height=4, width=4, encoding=encoding,
        is_bigendian=0, step=step, data=data)


def imu(t_ns, frame="imu"):
    Quat = TS.types["geometry_msgs/msg/Quaternion"]
    Vec3 = TS.types["geometry_msgs/msg/Vector3"]
    return TS.types["sensor_msgs/msg/Imu"](
        header=header(t_ns, frame),
        orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0), orientation_covariance=np.zeros(9),
        angular_velocity=Vec3(x=0.0, y=0.0, z=0.0), angular_velocity_covariance=np.zeros(9),
        linear_acceleration=Vec3(x=0.0, y=0.0, z=9.81), linear_acceleration_covariance=np.zeros(9),
    )


def circle_trajectory(n, radius=2.0, z=1.0):
    """Body poses on a circle, yaw = heading -> (t_s, p, q_xyzw) tuples."""
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        p = np.array([radius * np.cos(a), radius * np.sin(a), z])
        yaw = a + np.pi / 2
        q = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        out.append((1000.0 + i * 0.05, p, q))
    return out


def write_combo_bag(path, n_frames=10):
    """A tiny recording of rigs/pod3_oakdpro.yaml: 5 cams, oakd depth, 2 IMUs, GT."""
    cams = ["pod_cam0", "pod_cam1", "pod_cam2", "oakd_cam0", "oakd_cam1"]
    with Writer(path) as w:
        c_img = {c: w.add_connection(f"/uav1/{c}/color/image_raw", "sensor_msgs/msg/Image",
                                     typestore=TS) for c in cams}
        c_depth = w.add_connection("/uav1/oakd_cam0/depth/image_raw", "sensor_msgs/msg/Image",
                                   typestore=TS)
        c_imu = {ns: w.add_connection(f"/uav1/{ns}/imu", "sensor_msgs/msg/Imu", typestore=TS)
                 for ns in ("pod", "oakd")}
        c_gt = w.add_connection("/uav1/ground_truth", "nav_msgs/msg/Odometry", typestore=TS)
        for k, (t, p, q) in enumerate(circle_trajectory(n_frames)):
            t_ns = int(round(t * NS))
            w.write(c_gt, t_ns, TS.serialize_ros1(odometry(t_ns, p, q), "nav_msgs/msg/Odometry"))
            for c in cams:
                w.write(c_img[c], t_ns + 1, TS.serialize_ros1(image(t_ns, k), "sensor_msgs/msg/Image"))
            w.write(c_depth, t_ns + 2, TS.serialize_ros1(image(t_ns, 1.5, "32FC1"), "sensor_msgs/msg/Image"))
            for j in range(4):   # IMUs at 4x the camera rate
                tj = t_ns + j * (NS // 80)
                for ns in ("pod", "oakd"):
                    w.write(c_imu[ns], tj, TS.serialize_ros1(imu(tj), "sensor_msgs/msg/Imu"))
