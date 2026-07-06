"""Round-trip and sanity tests for the fisheye camera models."""

import numpy as np
import pytest

from tools.fisheye import KannalaBrandt4, DoubleSphere, EUCM, load_camera

RNG = np.random.default_rng(42)


def make_cameras():
    # Parameters in the ballpark of a 190-deg fisheye on a ~640x480 sensor
    # (kb4 coefficients roughly TUM-VI-like).
    return {
        "kb4": KannalaBrandt4(
            fx=190.9, fy=190.9, cx=254.9, cy=256.8,
            k1=0.0034, k2=0.00077, k3=-0.0034, k4=0.00055,
        ),
        "ds": DoubleSphere(
            fx=157.0, fy=157.0, cx=254.9, cy=256.8, xi=-0.18, alpha=0.59,
        ),
        "eucm": EUCM(
            fx=190.0, fy=190.0, cx=254.9, cy=256.8, alpha=0.6, beta=1.05,
        ),
    }


def sample_points(max_angle_deg: float, n: int = 2000) -> np.ndarray:
    """Random points within max_angle_deg of the optical axis, depth 0.3–20 m."""
    theta = np.deg2rad(max_angle_deg) * np.sqrt(RNG.uniform(0, 1, n))
    phi = RNG.uniform(0, 2 * np.pi, n)
    depth = RNG.uniform(0.3, 20.0, n)
    d = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=1,
    )
    return d * depth[:, None]


@pytest.mark.parametrize("name", ["kb4", "ds", "eucm"])
@pytest.mark.parametrize("max_angle_deg", [30, 60, 85])
def test_roundtrip_project_unproject(name, max_angle_deg):
    cam = make_cameras()[name]
    pts = sample_points(max_angle_deg)
    bearings_gt = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    px, valid_p = cam.project(pts)
    assert valid_p.all(), f"{name}: projection invalid within {max_angle_deg} deg"

    bearings, valid_u = cam.unproject(px)
    assert valid_u.all()
    err = np.rad2deg(np.arccos(np.clip((bearings * bearings_gt).sum(axis=1), -1, 1)))
    assert err.max() < 1e-5, f"{name}: max angular roundtrip error {err.max()} deg"


@pytest.mark.parametrize("name", ["ds", "eucm"])
def test_wide_fov_beyond_90deg(name):
    """DS and EUCM must handle rays behind the z=const pinhole limit (>90 deg)."""
    cam = make_cameras()[name]
    pts = sample_points(115)  # includes points with negative z
    px, valid = cam.project(pts)
    assert valid.mean() > 0.95
    bearings, _ = cam.unproject(px[valid])
    gt = pts[valid] / np.linalg.norm(pts[valid], axis=1, keepdims=True)
    err = np.rad2deg(np.arccos(np.clip((bearings * gt).sum(axis=1), -1, 1)))
    assert err.max() < 1e-5


def test_principal_point_maps_to_axis():
    for name, cam in make_cameras().items():
        b, valid = cam.unproject([[cam.cx, cam.cy]])
        assert valid.all()
        np.testing.assert_allclose(b, [[0.0, 0.0, 1.0]], atol=1e-9, err_msg=name)
        px, valid = cam.project([[0.0, 0.0, 2.0]])
        assert valid.all()
        np.testing.assert_allclose(px, [[cam.cx, cam.cy]], atol=1e-9, err_msg=name)


def test_kb4_rejects_beyond_max_theta():
    cam = make_cameras()["kb4"]
    # a point 150 deg off-axis (behind the camera)
    _, valid = cam.project([[0.5, 0.0, -0.866]])
    assert not valid.any()


def test_load_camera_from_rig_dict():
    cfg = {
        "model": "ds",
        "intrinsics": {"fx": 157.0, "fy": 157.0, "cx": 255.0, "cy": 257.0,
                       "xi": -0.18, "alpha": 0.59},
    }
    cam = load_camera(cfg)
    assert isinstance(cam, DoubleSphere)
    with pytest.raises(ValueError):
        load_camera({"model": "nope", "intrinsics": {}})
