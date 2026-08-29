"""Tests for bench.split_bag — composite recording -> per-pod contract bags."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rosbags")

from rosbags.rosbag1 import Reader

from bench.evaluate import load_tum
from bench.split_bag import split_bag, topic_map
from bench.rigdef import load_rig
from bench.tests.test_bag_helpers import write_combo_bag

ROOT = Path(__file__).resolve().parents[2]
COMBO = str(ROOT / "rigs" / "pod3_oakdpro.yaml")


def _topics(bag):
    with Reader(bag) as r:
        return {c.topic: c for c in r.connections}


def test_topic_map_restores_single_rig_contract():
    rig = load_rig(COMBO)
    assert topic_map(rig, "oakd") == {
        "/uav1/oakd_cam0/color/image_raw": "/uav1/cam0/color/image_raw",
        "/uav1/oakd_cam0/depth/image_raw": "/uav1/cam0/depth/image_raw",
        "/uav1/oakd_cam1/color/image_raw": "/uav1/cam1/color/image_raw",
        "/uav1/oakd/imu": "/uav1/sensor_pod/imu",
        "/uav1/ground_truth": "/uav1/ground_truth",
    }
    pod = topic_map(rig, "pod")
    assert pod["/uav1/pod_cam2/color/image_raw"] == "/uav1/cam2/color/image_raw"
    assert not any("depth" in t for t in pod)
    with pytest.raises(ValueError, match="composite"):
        topic_map(load_rig(str(ROOT / "rigs" / "oakdpro.yaml")), "x")


def test_split_writes_one_bag_and_gt_per_pod(tmp_path):
    src = str(tmp_path / "combo_day.bag")
    write_combo_bag(src, n_frames=10)
    result = split_bag(src, COMBO)

    assert set(result) == {"pod", "oakd"}
    assert result["pod"]["bag"] == str(tmp_path / "combo_day_pod.bag")
    assert result["oakd"]["rig"].endswith("rigs/oakdpro.yaml") and result["oakd"]["name"] == "oakdpro"

    pod_topics = _topics(result["pod"]["bag"])
    assert set(pod_topics) == {
        "/uav1/cam0/color/image_raw", "/uav1/cam1/color/image_raw",
        "/uav1/cam2/color/image_raw", "/uav1/sensor_pod/imu", "/uav1/ground_truth"}
    oakd_topics = _topics(result["oakd"]["bag"])
    assert set(oakd_topics) == {
        "/uav1/cam0/color/image_raw", "/uav1/cam1/color/image_raw",
        "/uav1/cam0/depth/image_raw", "/uav1/sensor_pod/imu", "/uav1/ground_truth"}
    assert oakd_topics["/uav1/cam0/depth/image_raw"].msgtype == "sensor_msgs/msg/Image"
    assert result["oakd"]["counts"] == {
        "/uav1/cam0/color/image_raw": 10, "/uav1/cam1/color/image_raw": 10,
        "/uav1/cam0/depth/image_raw": 10, "/uav1/sensor_pod/imu": 40, "/uav1/ground_truth": 10}

    # ground truth per pod, in that pod's IMU frame: same z (both at 0.12) but
    # the OAK-D path is offset 13 cm to starboard of the pod path
    gt_pod = load_tum(result["pod"]["gt"])
    gt_oakd = load_tum(result["oakd"]["gt"])
    assert gt_pod.shape == gt_oakd.shape == (10, 8)
    np.testing.assert_allclose(gt_pod[:, 3], gt_oakd[:, 3])
    np.testing.assert_allclose(np.linalg.norm(gt_pod[:, 1:4] - gt_oakd[:, 1:4], axis=1), 0.13, atol=1e-9)


def test_messages_are_copied_verbatim_in_order(tmp_path):
    src = str(tmp_path / "combo.bag")
    write_combo_bag(src, n_frames=5)
    out = split_bag(src, COMBO, out_dir=str(tmp_path / "split"), with_gt=False)
    assert "gt" not in out["pod"]
    with Reader(src) as r:
        src_msgs = [(t, raw) for c, t, raw in r.messages()
                    if c.topic == "/uav1/pod_cam1/color/image_raw"]
    with Reader(out["pod"]["bag"]) as r:
        dst_msgs = [(t, raw) for c, t, raw in r.messages()
                    if c.topic == "/uav1/cam1/color/image_raw"]
        times = [t for _, t, _ in r.messages()]
    assert src_msgs == dst_msgs
    assert times == sorted(times)


def test_resplit_overwrites(tmp_path):
    src = str(tmp_path / "combo.bag")
    write_combo_bag(src, n_frames=3)
    first = split_bag(src, COMBO)
    second = split_bag(src, COMBO)          # must not fail on existing outputs
    assert first["pod"]["bag"] == second["pod"]["bag"]
    assert second["pod"]["counts"]["/uav1/cam0/color/image_raw"] == 3
