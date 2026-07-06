"""Tests for bench.skew_bag — write a tiny ROS1 bag, skew it, verify shifts."""

import numpy as np
import pytest

pytest.importorskip("rosbags")

from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

from bench.skew_bag import skew_bag

TS = get_typestore(Stores.ROS1_NOETIC)
NS = 1_000_000_000


def make_imu(sec, nsec, frame="imu"):
    Header = TS.types["std_msgs/msg/Header"]
    Time = TS.types["builtin_interfaces/msg/Time"]
    Quat = TS.types["geometry_msgs/msg/Quaternion"]
    Vec3 = TS.types["geometry_msgs/msg/Vector3"]
    hdr_kwargs = {"stamp": Time(sec=sec, nanosec=nsec), "frame_id": frame}
    if "seq" in Header.__dataclass_fields__:
        hdr_kwargs["seq"] = 0
    return TS.types["sensor_msgs/msg/Imu"](
        header=Header(**hdr_kwargs),
        orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=np.zeros(9),
        angular_velocity=Vec3(x=0.0, y=0.0, z=0.0),
        angular_velocity_covariance=np.zeros(9),
        linear_acceleration=Vec3(x=0.0, y=0.0, z=9.81),
        linear_acceleration_covariance=np.zeros(9),
    )


def write_source_bag(path, n=50, dt_ns=NS // 10):
    with Writer(path) as w:
        c_imu = w.add_connection("/uav1/imu", "sensor_msgs/msg/Imu", typestore=TS)
        c_cam = w.add_connection("/uav1/cam1/data", "sensor_msgs/msg/Imu", typestore=TS)
        t0 = 1000 * NS
        for i in range(n):
            t = t0 + i * dt_ns
            raw = TS.serialize_ros1(make_imu(t // NS, t % NS), "sensor_msgs/msg/Imu")
            w.write(c_imu, t, raw)
            w.write(c_cam, t, raw)


def read_stamps(path):
    """Returns {topic: [(bag_time_ns, header_time_ns), ...]}."""
    out = {}
    with Reader(path) as r:
        for conn, timestamp, raw in r.messages():
            msg = TS.deserialize_ros1(raw, conn.msgtype)
            hdr_ns = msg.header.stamp.sec * NS + msg.header.stamp.nanosec
            out.setdefault(conn.topic, []).append((timestamp, hdr_ns))
    return out


def test_offset_shifts_only_target_topic(tmp_path):
    src, dst = str(tmp_path / "in.bag"), str(tmp_path / "out.bag")
    write_source_bag(src)
    counts = skew_bag(src, dst, offsets={"/uav1/cam1/data": 0.015}, jitters={})
    assert counts == {"/uav1/imu": 50, "/uav1/cam1/data": 50}

    before, after = read_stamps(src), read_stamps(dst)
    # untouched topic identical
    assert after["/uav1/imu"] == before["/uav1/imu"]
    # skewed topic: bag time AND header shifted by exactly 15 ms
    shift = 15_000_000
    for (b0, h0), (b1, h1) in zip(before["/uav1/cam1/data"], after["/uav1/cam1/data"]):
        assert b1 - b0 == shift
        assert h1 - h0 == shift


def test_negative_offset_keeps_chronological_order(tmp_path):
    src, dst = str(tmp_path / "in.bag"), str(tmp_path / "out.bag")
    write_source_bag(src)
    skew_bag(src, dst, offsets={"/uav1/cam1/data": -0.25}, jitters={})
    with Reader(dst) as r:
        times = [t for _, t, _ in r.messages()]
    assert times == sorted(times)


def test_jitter_is_deterministic_and_zero_mean(tmp_path):
    src = str(tmp_path / "in.bag")
    write_source_bag(src, n=200)
    d1, d2 = str(tmp_path / "a.bag"), str(tmp_path / "b.bag")
    skew_bag(src, d1, offsets={}, jitters={"/uav1/cam1/data": 0.002}, seed=3)
    skew_bag(src, d2, offsets={}, jitters={"/uav1/cam1/data": 0.002}, seed=3)
    s1, s2 = read_stamps(d1), read_stamps(d2)
    assert s1 == s2  # same seed -> identical output
    src_stamps = read_stamps(src)
    diffs = np.array([
        a[0] - b[0] for a, b in zip(s1["/uav1/cam1/data"], src_stamps["/uav1/cam1/data"])
    ])
    assert diffs.std() == pytest.approx(0.002 * NS, rel=0.25)
    assert abs(diffs.mean()) < 0.001 * NS
