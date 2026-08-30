"""Tests for bench.frames2bag — rendered frames + flight bag -> contract bag."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rosbags")

from PIL import Image
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

from bench.frames2bag import frames2bag
from bench.split_bag import split_bag
from bench.tests.test_bag_helpers import NS, circle_trajectory, imu, odometry

ROOT = Path(__file__).resolve().parents[2]
COMBO = str(ROOT / "rigs" / "pod3_oakdpro.yaml")
TS = get_typestore(Stores.ROS1_NOETIC)


def write_flight_bag(path, n=8):
    """IMU (two pods) + ground truth only — what the flight pass records."""
    with Writer(path) as w:
        c_imu = {ns: w.add_connection(f"/uav1/{ns}/imu", "sensor_msgs/msg/Imu", typestore=TS)
                 for ns in ("pod", "oakd")}
        c_gt = w.add_connection("/uav1/ground_truth", "nav_msgs/msg/Odometry", typestore=TS)
        for t, p, q in circle_trajectory(n):
            t_ns = int(round(t * NS))
            w.write(c_gt, t_ns, TS.serialize_ros1(odometry(t_ns, p, q), "nav_msgs/msg/Odometry"))
            for j in range(4):
                tj = t_ns + j * (NS // 80)
                for ns in ("pod", "oakd"):
                    w.write(c_imu[ns], tj, TS.serialize_ros1(imu(tj), "sensor_msgs/msg/Imu"))


def write_frames(frames_dir: Path, n=8):
    """What render_from_poses.py writes: per camera frames.csv + png (+ depth)."""
    rng = np.random.default_rng(0)
    cams = {"pod_cam0": (8, 6, True), "pod_cam1": (8, 6, False), "pod_cam2": (8, 6, False),
            "oakd_cam0": (10, 6, True), "oakd_cam1": (10, 6, False)}
    for name, (w, h, rgb) in cams.items():
        d = frames_dir / name
        d.mkdir(parents=True)
        with open(d / "frames.csv", "w") as f:
            f.write("index,t_ns\n")
            for k, (t, _, _) in enumerate(circle_trajectory(n)):
                t_ns = int(round(t * NS))
                f.write(f"{k},{t_ns}\n")
                shape = (h, w, 3) if rgb else (h, w)
                Image.fromarray(rng.integers(0, 255, shape, dtype=np.uint8)).save(d / f"{k:06d}.png")
                if name == "oakd_cam0":
                    np.save(d / f"{k:06d}.depth.npy", rng.uniform(0.5, 5, (h, w)).astype(np.float32))


def test_frames2bag_builds_the_contract(tmp_path):
    flight = str(tmp_path / "flight.bag")
    write_flight_bag(flight)
    write_frames(tmp_path / "frames")
    out = str(tmp_path / "combo_day.bag")
    counts = frames2bag(str(tmp_path / "frames"), flight, COMBO, out)

    assert counts["/uav1/pod_cam0/color/image_raw"] == 8
    assert counts["/uav1/oakd_cam0/depth/image_raw"] == 8
    assert counts["/uav1/pod/imu"] == 32 and counts["/uav1/oakd/imu"] == 32
    assert counts["/uav1/ground_truth"] == 8
    assert "/uav1/pod_cam1/depth/image_raw" not in counts

    with Reader(out) as r:
        times = [t for _, t, _ in r.messages()]
        assert times == sorted(times)                              # time-ordered
        enc = {}
        for conn, _, raw in r.messages():
            if conn.msgtype == "sensor_msgs/msg/Image":
                m = TS.deserialize_ros1(raw, conn.msgtype)
                enc[conn.topic] = (m.encoding, m.width, m.height, m.step, len(m.data))
    assert enc["/uav1/pod_cam0/color/image_raw"] == ("rgb8", 8, 6, 24, 144)
    assert enc["/uav1/pod_cam1/color/image_raw"] == ("mono8", 8, 6, 8, 48)
    assert enc["/uav1/oakd_cam0/depth/image_raw"] == ("32FC1", 10, 6, 40, 240)

    # and the result is a normal composite recording: split_bag restores the contracts
    split = split_bag(out, COMBO)
    assert split["oakd"]["counts"]["/uav1/cam0/depth/image_raw"] == 8
    assert split["pod"]["counts"]["/uav1/sensor_pod/imu"] == 32


def test_missing_flight_topics_is_an_error(tmp_path):
    with Writer(str(tmp_path / "bad.bag")) as w:
        c = w.add_connection("/uav1/ground_truth", "nav_msgs/msg/Odometry", typestore=TS)
        t, p, q = circle_trajectory(1)[0]
        w.write(c, int(t * NS), TS.serialize_ros1(odometry(int(t * NS), p, q), "nav_msgs/msg/Odometry"))
    write_frames(tmp_path / "frames", n=1)
    with pytest.raises(ValueError, match="missing flight topics"):
        frames2bag(str(tmp_path / "frames"), str(tmp_path / "bad.bag"), COMBO, str(tmp_path / "o.bag"))
