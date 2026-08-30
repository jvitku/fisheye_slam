"""Tests for bench.pose_sampler — fixed-rate SE3 resampling of the ground truth."""

import numpy as np
import pytest

from bench.pose_sampler import sample, slerp
from bench.rigdef import quat_xyzw_to_R


def _yaw_quat(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def _analytic(t):
    """Straight line at 1 m/s along x, yawing at 0.5 rad/s, 1.5 m up."""
    return np.array([t, t, 0.0, 1.5, *_yaw_quat(0.5 * t)])


def test_sampling_matches_analytic_motion():
    gt = np.array([_analytic(t) for t in np.arange(100.0, 110.0, 0.013)])   # irregular ~77 Hz
    out = sample(gt, rate=20.0)
    assert out.shape[1] == 8
    np.testing.assert_allclose(np.diff(out[:, 0]), 0.05, atol=1e-9)         # exact period
    assert out[0, 0] == gt[0, 0] and out[-1, 0] <= gt[-1, 0]
    for row in out[::7]:
        ref = _analytic(row[0])
        np.testing.assert_allclose(row[1:4], ref[1:4], atol=1e-6)
        # compare rotations, sign-invariant
        np.testing.assert_allclose(quat_xyzw_to_R(row[4:8]), quat_xyzw_to_R(ref[4:8]), atol=1e-5)


def test_start_and_duration_window():
    gt = np.array([_analytic(t) for t in np.arange(0.0, 60.0, 0.05)])
    out = sample(gt, rate=10.0, start=5.0, duration=20.0)
    assert out[0, 0] == pytest.approx(5.0) and out[-1, 0] == pytest.approx(25.0)
    assert len(out) == 201
    with pytest.raises(ValueError):
        sample(gt, rate=10.0, start=100.0)


def test_slerp_shortest_arc_and_endpoints():
    q0, q1 = _yaw_quat(0.0), -_yaw_quat(np.pi / 2)      # q1 negated: same rotation
    np.testing.assert_allclose(slerp(q0, q1, 0.0), q0)
    mid = slerp(q0, q1, 0.5)
    np.testing.assert_allclose(quat_xyzw_to_R(mid), quat_xyzw_to_R(_yaw_quat(np.pi / 4)), atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(mid), 1.0)
