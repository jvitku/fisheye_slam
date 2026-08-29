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


def test_load_rig_rejects_unknown_model(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nimu: {topic: /x}\ncameras:\n"
        "  - {name: c0, model: ds, resolution: [512, 512], rate_hz: 20,\n"
        "     intrinsics: {fx: 1, fy: 1, cx: 0, cy: 0},\n"
        "     mount: {position: [0, 0, 0], rpy_deg: [0, 0, 0]}}\n"
    )
    with pytest.raises(ValueError, match="kb4 or pinhole"):
        load_rig(str(bad))


def test_oakdpro_rig():
    """OAK-D Pro rig: pinhole stereo pod, depth on cam0 only, 7.5 cm baseline."""
    rig = load_rig(str(ROOT / "rigs" / "oakdpro.yaml"))
    assert "pod" in rig
    assert [c["model"] for c in rig["cameras"]] == ["pinhole", "pinhole"]
    assert [c["depth"] for c in rig["cameras"]] == [True, False]
    assert rig["illuminator"]["type"] == "ir_850nm"
    pos = np.array([c["mount"]["position"] for c in rig["cameras"]])
    assert np.linalg.norm(pos[0] - pos[1]) == pytest.approx(0.075, abs=1e-9)
    # fx from spec-sheet HFOV: fx = (W/2) / tan(HFOV/2)
    cam = rig["cameras"][0]
    w = cam["resolution"][0]
    fx_expected = (w / 2.0) / np.tan(np.deg2rad(cam["fov_deg"] / 2.0))
    assert cam["intrinsics"]["fx"] == pytest.approx(fx_expected, rel=1e-3)


def test_depth_flag_defaults_false():
    """Rigs that never mention depth get an explicit depth=False per camera."""
    rig = load_rig(str(ROOT / "rigs" / "rig_2cam.yaml"))
    assert all(c["depth"] is False for c in rig["cameras"])


# --- list views + composite rigs (rigs/pod3_oakdpro.yaml) -------------------

from rig_math import (  # noqa: E402
    R_to_quat_xyzw, imu_sensor_config, pod_imu, quat_xyzw_to_R, recorded_topics,
)


def test_single_rig_list_views():
    pod = load_rig(str(ROOT / "rigs" / "pod_3cam_triangle.yaml"))
    assert pod["imus"] == [pod["imu"]] and pod["imu"]["kind"] == "pod"
    assert pod["illuminators"] == [pod["illuminator"]]
    assert "composite" not in pod
    body = load_rig(str(ROOT / "rigs" / "rig_2cam.yaml"))
    assert body["imus"][0]["kind"] == "body" and body["illuminators"] == []


def test_composite_rig_flattens_members():
    rig = load_rig(str(ROOT / "rigs" / "pod3_oakdpro.yaml"))
    assert rig["composite"] is True
    assert [c["name"] for c in rig["cameras"]] == [
        "pod_cam0", "pod_cam1", "pod_cam2", "oakd_cam0", "oakd_cam1"]
    assert [c["source_name"] for c in rig["cameras"]] == ["cam0", "cam1", "cam2", "cam0", "cam1"]
    assert [c["model"] for c in rig["cameras"]] == ["kb4"] * 3 + ["pinhole"] * 2
    assert [c["depth"] for c in rig["cameras"]] == [False, False, False, True, False]
    assert [i["topic"] for i in rig["imus"]] == ["/uav1/pod/imu", "/uav1/oakd/imu"]
    assert all(i["kind"] == "pod" and i["source_topic"] == "/uav1/sensor_pod/imu"
               for i in rig["imus"])
    assert [i["rate_hz"] for i in rig["imus"]] == [400, 200]
    assert [ill["pod_ns"] for ill in rig["illuminators"]] == ["pod", "oakd"]
    assert [p["ns"] for p in rig["pods"]] == ["pod", "oakd"]
    assert rig["pods"][1]["name"] == "oakdpro"
    # `source` keeps the member's own single-rig contract for downstream tools
    src = rig["pods"][1]["source"]
    assert [c["name"] for c in src["cameras"]] == ["cam0", "cam1"]
    assert src["imu"]["topic"] == "/uav1/sensor_pod/imu"


def test_composite_mount_override_composes():
    rig = load_rig(str(ROOT / "rigs" / "pod3_oakdpro.yaml"))
    cams = {c["name"]: c for c in rig["cameras"]}
    # member default mount kept for the pod (12 cm above body center)
    np.testing.assert_allclose(cams["pod_cam2"]["body_mount"]["position"], [0.02, 0.0, 0.189], atol=1e-12)
    # OAK-D moved to starboard: cam0 (left mono, +3.75 cm) lands at y = -0.13 + 0.0375
    np.testing.assert_allclose(cams["oakd_cam0"]["body_mount"]["position"], [0.0, -0.0925, 0.12], atol=1e-12)
    np.testing.assert_allclose(cams["oakd_cam1"]["body_mount"]["position"], [0.0, -0.1675, 0.12], atol=1e-12)
    # the override is also reflected in the resolved member source + pod entry
    np.testing.assert_allclose(rig["imus"][1]["body_mount"]["position"], [0.0, -0.13, 0.12], atol=1e-12)
    np.testing.assert_allclose(rig["pods"][1]["source"]["imu"]["body_mount"]["position"],
                               [0.0, -0.13, 0.12], atol=1e-12)


def test_composite_validation(tmp_path):
    body = ROOT / "rigs" / "rig_2cam.yaml"
    pod = ROOT / "rigs" / "pod_2cam.yaml"

    def write(text):
        f = tmp_path / "combo.yaml"
        f.write_text(text)
        return str(f)

    with pytest.raises(ValueError, match="pod rigs"):
        load_rig(write(f"name: x\npods:\n  - {{ns: a, rig: {body}}}\n"))
    with pytest.raises(ValueError, match="alphanumeric"):
        load_rig(write(f"name: x\npods:\n  - {{ns: pod-1, rig: {pod}}}\n"))
    with pytest.raises(ValueError, match="duplicate"):
        load_rig(write(f"name: x\npods:\n  - {{ns: a, rig: {pod}}}\n  - {{ns: a, rig: {pod}}}\n"))
    nested = write(f"name: x\npods:\n  - {{ns: a, rig: {pod}}}\n")
    outer = tmp_path / "outer.yaml"
    outer.write_text(f"name: y\npods:\n  - {{ns: b, rig: {nested}}}\n")
    with pytest.raises(ValueError, match="nested"):
        load_rig(str(outer))
    with pytest.raises(ValueError, match="empty"):
        load_rig(write("name: x\npods: []\n"))


def test_recorded_topics():
    single = load_rig(str(ROOT / "rigs" / "oakdpro.yaml"))
    assert recorded_topics(single) == [
        "/uav1/cam0/color/image_raw", "/uav1/cam1/color/image_raw",
        "/uav1/cam0/depth/image_raw", "/uav1/sensor_pod/imu", "/uav1/ground_truth",
    ]
    combo = recorded_topics(load_rig(str(ROOT / "rigs" / "pod3_oakdpro.yaml")))
    assert "/uav1/oakd_cam0/depth/image_raw" in combo
    assert "/uav1/pod/imu" in combo and "/uav1/oakd/imu" in combo
    assert combo.count("/uav1/ground_truth") == 1 and len(combo) == 9


def test_imu_sensor_config_from_yaml():
    rig = load_rig(str(ROOT / "rigs" / "pod3_oakdpro.yaml"))
    cfg = imu_sensor_config(pod_imu(rig, "oakd"))
    assert cfg["topic"] == "/uav1/oakd/imu" and cfg["update_rate"] == 200.0
    np.testing.assert_allclose(cfg["position"], [0.0, -0.13, 0.12])
    assert cfg["orientation"] == [0.0, 0.0, 0.0]          # [yaw, pitch, roll]
    assert cfg["gyroscope"] == {"noise_density": 2.4e-4, "random_walk": 2.0e-5}
    assert cfg["accelerometer"] == {"noise_density": 1.6e-3, "random_walk": 3.0e-4}
    # the two devices' IMUs differ the way the hardware does
    assert imu_sensor_config(pod_imu(rig, "pod"))["gyroscope"]["noise_density"] == 1.0e-4


def test_pod_imu_selection():
    combo = load_rig(str(ROOT / "rigs" / "pod3_oakdpro.yaml"))
    assert pod_imu(combo, "pod")["topic"] == "/uav1/pod/imu"
    with pytest.raises(ValueError, match="pick a pod"):
        pod_imu(combo)
    with pytest.raises(ValueError, match="no pod ns"):
        pod_imu(combo, "nope")
    single = load_rig(str(ROOT / "rigs" / "pod_2cam.yaml"))
    assert pod_imu(single) is single["imu"]
    with pytest.raises(ValueError):
        pod_imu(single, "pod")


def test_quaternion_roundtrip():
    for _ in range(200):
        R = euler_to_R([RNG.uniform(-179, 179), RNG.uniform(-89, 89), RNG.uniform(-179, 179)])
        np.testing.assert_allclose(quat_xyzw_to_R(R_to_quat_xyzw(R)), R, atol=1e-12)
    np.testing.assert_allclose(R_to_quat_xyzw(np.eye(3)), [0, 0, 0, 1])
