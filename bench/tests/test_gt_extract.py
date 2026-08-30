"""Tests for bench.gt_extract — sim ground truth to TUM, optionally at a pod frame."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rosbags")

from bench.evaluate import load_tum
from bench.gt_extract import body_to_sensor_T, extract, read_odometry, transform_poses
from bench.rigdef import load_rig, quat_xyzw_to_R
from bench.tests.test_bag_helpers import circle_trajectory, write_combo_bag

ROOT = Path(__file__).resolve().parents[2]


def test_read_odometry_uses_header_time_and_sorts(tmp_path):
    bag = str(tmp_path / "combo.bag")
    write_combo_bag(bag, n_frames=8)
    poses = read_odometry(bag)
    expected = circle_trajectory(8)
    assert poses.shape == (8, 8)
    np.testing.assert_allclose(poses[:, 0], [t for t, _, _ in expected], atol=1e-6)
    np.testing.assert_allclose(poses[:, 1:4], [p for _, p, _ in expected], atol=1e-12)


def test_transform_to_pod_frame_applies_lever_arm():
    poses = np.array([[r[0], *r[1], *r[2]] for r in circle_trajectory(12)])
    T = np.eye(4)
    T[:3, 3] = [0.0, -0.13, 0.12]                       # OAK-D mount of pod3_oakdpro
    out = transform_poses(poses, T)
    for row_in, row_out in zip(poses, out):
        R = quat_xyzw_to_R(row_in[4:8])
        np.testing.assert_allclose(row_out[1:4], row_in[1:4] + R @ T[:3, 3], atol=1e-12)
        # pure translation: same rotation (quaternion sign is not significant)
        np.testing.assert_allclose(quat_xyzw_to_R(row_out[4:8]), R, atol=1e-12)
    # starboard offset: the sensor path is a slightly smaller circle (r - 0.13)
    r_body = np.linalg.norm(poses[:, 1:3], axis=1)
    r_sens = np.linalg.norm(out[:, 1:3], axis=1)
    np.testing.assert_allclose(r_sens, r_body - 0.13, atol=1e-9)


def test_body_to_sensor_T_from_rig():
    rig = load_rig(str(ROOT / "rigs" / "pod3_oakdpro.yaml"))
    np.testing.assert_allclose(body_to_sensor_T(rig, "oakd")[:3, 3], [0.0, -0.13, 0.12])
    np.testing.assert_allclose(body_to_sensor_T(rig, "pod")[:3, 3], [0.0, 0.0, 0.12])
    single = load_rig(str(ROOT / "rigs" / "pod_2cam.yaml"))
    np.testing.assert_allclose(body_to_sensor_T(single)[:3, 3], [0.0, 0.0, 0.12])


def test_extract_writes_tum_readable_by_evaluate(tmp_path):
    bag = str(tmp_path / "combo.bag")
    write_combo_bag(bag, n_frames=6)
    out = tmp_path / "gt.txt"
    rig = load_rig(str(ROOT / "rigs" / "pod3_oakdpro.yaml"))
    assert extract(bag, str(out), T_body_sensor=body_to_sensor_T(rig, "pod")) == 6
    tum = load_tum(str(out))
    assert tum.shape == (6, 8)
    np.testing.assert_allclose(tum[:, 3], 1.0 + 0.12)   # z lifted by the pod mount
