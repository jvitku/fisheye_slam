"""Tests for bench.gen_openvins_config — rig yaml -> OpenVINS config dir."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from bench.gen_openvins_config import cam_T_cam_imu, generate, overlaps
from bench.rigdef import load_rig

ROOT = Path(__file__).resolve().parents[2]


def load_cv_yaml(path: Path) -> dict:
    """OpenCV yaml = '%YAML:1.0' directive line + plain yaml."""
    text = path.read_text().split("\n", 1)[1]
    return yaml.safe_load(text)


def test_pod3_triangle_config(tmp_path):
    (out,) = generate(str(ROOT / "rigs" / "pod_3cam_triangle.yaml"), str(tmp_path))
    assert out == tmp_path
    chain = load_cv_yaml(out / "kalibr_imucam_chain.yaml")
    assert sorted(chain) == ["cam0", "cam1", "cam2"]
    cam0 = chain["cam0"]
    assert cam0["distortion_model"] == "equidistant" and cam0["camera_model"] == "pinhole"
    assert cam0["distortion_coeffs"] == [0.0, 0.0, 0.0, 0.0]
    assert cam0["intrinsics"] == [140.0, 140.0, 256.0, 256.0]
    assert cam0["resolution"] == [512, 512]
    assert cam0["rostopic"] == "/uav1/cam0/color/image_raw"
    assert cam0["cam_overlaps"] == [1, 2] and chain["cam2"]["cam_overlaps"] == [0, 1]
    # p_IinC: IMU at the pod origin seen from cam0 (2 cm fwd, 6 cm left, 3.5 cm
    # down of it) -> in optical x-right/y-down/z-fwd: (+0.06, -0.035, -0.02)
    T = np.array(cam0["T_cam_imu"])
    np.testing.assert_allclose(T[:3, 3], [0.06, -0.035, -0.02], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], [[0, -1, 0], [0, 0, -1], [1, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(T @ np.linalg.inv(T), np.eye(4), atol=1e-12)

    imu = load_cv_yaml(out / "kalibr_imu_chain.yaml")["imu0"]
    assert imu["rostopic"] == "/uav1/sensor_pod/imu" and imu["update_rate"] == 400.0
    assert imu["gyroscope_noise_density"] == 1.0e-4
    assert imu["accelerometer_random_walk"] == 1.0e-4
    assert imu["model"] == "kalibr" and imu["T_i_b"][0] == [1.0, 0.0, 0.0, 0.0]

    est = load_cv_yaml(out / "estimator_config.yaml")
    assert est["max_cameras"] == 3 and est["use_stereo"] is True
    assert est["use_mask"] is True
    assert est["mask0"] == "mask_cam0.png" and est["mask2"] == "mask_cam2.png"
    assert est["calib_cam_timeoffset"] is True
    assert est["track_frequency"] == 21.0
    assert est["relative_config_imucam"] == "kalibr_imucam_chain.yaml"
    assert est["filepath_est"] == "/out/state_estimate.txt"

    mask = np.asarray(Image.open(out / "mask_cam0.png"))
    assert mask.shape == (512, 512) and mask.dtype == np.uint8
    assert mask[256, 256] == 0 and mask[2, 2] == 255          # 255 = masked out
    r = 140.0 * np.deg2rad(95.0) * 0.98                        # image circle
    assert mask[256, int(256 + r) - 2] == 0 and mask[256, int(256 + r) + 2] == 255


def test_oakdpro_config(tmp_path):
    (out,) = generate(str(ROOT / "rigs" / "oakdpro.yaml"), str(tmp_path))
    chain = load_cv_yaml(out / "kalibr_imucam_chain.yaml")
    cam1 = chain["cam1"]
    assert cam1["distortion_model"] == "radtan" and cam1["distortion_coeffs"] == [0.0] * 4
    assert cam1["intrinsics"] == [762.72, 762.72, 640.0, 400.0]
    assert cam1["resolution"] == [1280, 800]
    # right mono at y = -3.75 cm: the IMU is 3.75 cm to its LEFT -> optical x = -0.0375
    np.testing.assert_allclose(np.array(cam1["T_cam_imu"])[:3, 3], [-0.0375, 0.0, 0.0], atol=1e-12)
    est = load_cv_yaml(out / "estimator_config.yaml")
    assert est["max_cameras"] == 2 and est["use_stereo"] is True and est["use_mask"] is False
    assert "mask0" not in est
    assert not list(out.glob("*.png"))
    imu = load_cv_yaml(out / "kalibr_imu_chain.yaml")["imu0"]
    assert imu["update_rate"] == 200.0 and imu["gyroscope_noise_density"] == 2.4e-4


def test_overlaps_geometry():
    rig3 = load_rig(str(ROOT / "rigs" / "rig_3cam.yaml"))      # front pair + rear
    assert overlaps(rig3, 0) == [1] and overlaps(rig3, 2) == []
    ring = load_rig(str(ROOT / "rigs" / "rig_6cam.yaml"))      # 60 deg ring, 190 deg lenses
    assert overlaps(ring, 0) == [1, 5]


def test_extrinsics_with_yawed_camera():
    rig3 = load_rig(str(ROOT / "rigs" / "rig_3cam.yaml"))
    rear = rig3["cameras"][2]                                   # yaw 180 at x = -0.10
    T = cam_T_cam_imu(rig3["imus"][0], rear)
    # body IMU at the center; the rear camera looks backwards, so the IMU
    # (towards the nose) sits 10 cm BEHIND it on its optical axis
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, -0.10], atol=1e-12)


def test_composite_generates_per_pod(tmp_path):
    dirs = generate(str(ROOT / "rigs" / "pod3_oakdpro.yaml"), str(tmp_path))
    assert [d.name for d in dirs] == ["pod", "oakd"]
    pod = load_cv_yaml(tmp_path / "pod" / "kalibr_imucam_chain.yaml")
    assert pod["cam0"]["rostopic"] == "/uav1/cam0/color/image_raw"   # member contract, not pod_cam0
    oakd = load_cv_yaml(tmp_path / "oakd" / "kalibr_imu_chain.yaml")["imu0"]
    assert oakd["rostopic"] == "/uav1/sensor_pod/imu"
    # the OAK-D's own extrinsics are unaffected by where it sits on the drone
    oak_cam0 = np.array(load_cv_yaml(tmp_path / "oakd" / "kalibr_imucam_chain.yaml")["cam0"]["T_cam_imu"])
    np.testing.assert_allclose(oak_cam0[:3, 3], [0.0375, 0.0, 0.0], atol=1e-12)
