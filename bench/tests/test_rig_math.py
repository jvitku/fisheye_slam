"""Tests for sim/isaac/workspace/rig_math.py — pod-mount composition.

rig_math is deliberately Isaac-free so it can be tested on the host; import it
straight from the sim workspace directory.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim" / "isaac" / "workspace"))

from rig_math import R_to_euler, compose_mount, euler_to_R, load_rig  # noqa: E402

RNG = np.random.default_rng(11)


def test_euler_roundtrip():
    for _ in range(200):
        rpy = [RNG.uniform(-179, 179), RNG.uniform(-89, 89), RNG.uniform(-179, 179)]
        back = R_to_euler(euler_to_R(rpy))
        np.testing.assert_allclose(back, rpy, atol=1e-9)


def test_euler_axis_conventions():
    # yaw +90: body +x -> +y (FLU)
    np.testing.assert_allclose(euler_to_R([0, 0, 90]) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    # pitch +90: body +x -> -z (nose down)
    np.testing.assert_allclose(euler_to_R([0, 90, 0]) @ [1, 0, 0], [0, 0, -1], atol=1e-12)
    # roll +90: body +y -> +z
    np.testing.assert_allclose(euler_to_R([90, 0, 0]) @ [0, 1, 0], [0, 0, 1], atol=1e-12)


def test_compose_identity_passthrough():
    inner = {"position": [0.02, 0.06, -0.035], "rpy_deg": [0.0, 0.0, 0.0]}
    out = compose_mount({"position": [0, 0, 0], "rpy_deg": [0, 0, 0]}, inner)
    np.testing.assert_allclose(out["position"], inner["position"], atol=1e-12)
    np.testing.assert_allclose(out["rpy_deg"], inner["rpy_deg"], atol=1e-12)


def test_compose_translated_yawed_pod():
    pod = {"position": [0.0, 0.0, 0.12], "rpy_deg": [0.0, 0.0, 90.0]}
    cam = {"position": [0.02, 0.06, -0.03], "rpy_deg": [0.0, 0.0, 0.0]}
    out = compose_mount(pod, cam)
    # yaw 90 maps (x, y) -> (-y, x)
    np.testing.assert_allclose(out["position"], [-0.06, 0.02, 0.09], atol=1e-12)
    np.testing.assert_allclose(out["rpy_deg"], [0.0, 0.0, 90.0], atol=1e-9)


def test_compose_rotation_order():
    # pod pitched down 30, camera yawed 90 inside pod: composed R must equal
    # R_pod @ R_cam, not the other way round
    pod = {"position": [0, 0, 0], "rpy_deg": [0, 30, 0]}
    cam = {"position": [0, 0, 0], "rpy_deg": [0, 0, 90]}
    out = compose_mount(pod, cam)
    np.testing.assert_allclose(
        euler_to_R(out["rpy_deg"]), euler_to_R(pod["rpy_deg"]) @ euler_to_R(cam["rpy_deg"]),
        atol=1e-12,
    )


@pytest.mark.parametrize("rig_file,n_cams,has_pod", [
    ("rig_2cam.yaml", 2, False),
    ("rig_3cam.yaml", 3, False),
    ("rig_6cam.yaml", 6, False),
    ("pod_2cam.yaml", 2, True),
    ("pod_3cam_triangle.yaml", 3, True),
])
def test_load_rig_resolves_body_mounts(rig_file, n_cams, has_pod):
    rig = load_rig(str(ROOT / "rigs" / rig_file))
    assert len(rig["cameras"]) == n_cams
    assert ("pod" in rig) == has_pod
    for cam in rig["cameras"]:
        assert "body_mount" in cam
        if has_pod:
            # pod sits 0.12 m above body center; identity pod rotation
            pod_z = rig["pod"]["mount"]["position"][2]
            assert cam["body_mount"]["position"][2] == pytest.approx(
                pod_z + cam["mount"]["position"][2]
            )
        else:
            np.testing.assert_allclose(cam["body_mount"]["position"], cam["mount"]["position"])
    assert "body_mount" in rig["imu"]


def test_pod_cameras_coplanar_same_direction():
    """The pod contract: all cameras in one plane, optical axes parallel."""
    for rig_file in ("pod_2cam.yaml", "pod_3cam_triangle.yaml"):
        rig = load_rig(str(ROOT / "rigs" / rig_file))
        xs = {cam["mount"]["position"][0] for cam in rig["cameras"]}
        assert len(xs) == 1, f"{rig_file}: cameras not coplanar"
        dirs = [tuple(cam["mount"]["rpy_deg"]) for cam in rig["cameras"]]
        assert len(set(dirs)) == 1, f"{rig_file}: cameras not co-oriented"


def test_triangle_geometry_equilateral():
    rig = load_rig(str(ROOT / "rigs" / "pod_3cam_triangle.yaml"))
    pos = np.array([c["mount"]["position"] for c in rig["cameras"]])
    d = [np.linalg.norm(pos[i] - pos[j]) for i, j in ((0, 1), (1, 2), (0, 2))]
    assert max(d) - min(d) < 5e-3
    assert np.mean(d) == pytest.approx(0.12, abs=5e-3)


def test_illuminator_at_triangle_centroid():
    """The IR flood LED sits at the camera-triangle centroid (y-z plane)."""
    rig = load_rig(str(ROOT / "rigs" / "pod_3cam_triangle.yaml"))
    ill = rig["illuminator"]
    centroid = np.mean([c["mount"]["position"] for c in rig["cameras"]], axis=0)
    np.testing.assert_allclose(ill["mount"]["position"][1:], centroid[1:], atol=2e-3)
    # resolved to body frame: pod z-offset applied
    assert "body_mount" in ill
    pod_z = rig["pod"]["mount"]["position"][2]
    assert ill["body_mount"]["position"][2] == pytest.approx(
        pod_z + ill["mount"]["position"][2]
    )


def test_illuminator_resolved_for_both_pods():
    for rig_file in ("pod_2cam.yaml", "pod_3cam_triangle.yaml"):
        rig = load_rig(str(ROOT / "rigs" / rig_file))
        assert rig["illuminator"]["type"] == "ir_850nm"
        assert "body_mount" in rig["illuminator"]
    # body rigs have no illuminator, and load_rig must not choke on that
    assert "illuminator" not in load_rig(str(ROOT / "rigs" / "rig_2cam.yaml"))


def test_load_rig_rejects_non_kb4(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nimu: {topic: /x}\ncameras:\n"
        "  - {name: c0, model: ds, resolution: [512, 512], rate_hz: 20,\n"
        "     intrinsics: {fx: 1, fy: 1, cx: 0, cy: 0},\n"
        "     mount: {position: [0, 0, 0], rpy_deg: [0, 0, 0]}}\n"
    )
    with pytest.raises(ValueError, match="kb4"):
        load_rig(str(bad))
