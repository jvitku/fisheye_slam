"""Tests for bench.evaluate — synthetic trajectories with known answers."""

import numpy as np
import pytest

from bench.evaluate import associate, ate_stats, coverage_stats, evaluate, load_tum, umeyama

RNG = np.random.default_rng(7)


def circle_traj(n=500, dt=0.05, radius=3.0):
    """Circular trajectory with yaw tangent to the circle. TUM (N,8)."""
    t = np.arange(n) * dt
    ang = 0.4 * t
    xyz = np.stack([radius * np.cos(ang), radius * np.sin(ang), 1.0 + 0.1 * np.sin(t)], axis=1)
    yaw = ang + np.pi / 2
    q = np.stack([np.zeros(n), np.zeros(n), np.sin(yaw / 2), np.cos(yaw / 2)], axis=1)
    return np.concatenate([t[:, None], xyz, q], axis=1)


def write_tum(path, arr):
    np.savetxt(path, arr, fmt="%.9f")


def rigidly_transform(traj, R, t):
    out = traj.copy()
    out[:, 1:4] = (R @ traj[:, 1:4].T).T + t
    # note: quaternions left untouched — fine for ATE/coverage tests
    return out


def random_rotation():
    q = RNG.normal(size=4)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_umeyama_recovers_exact_transform():
    src = RNG.normal(size=(100, 3))
    R, t = random_rotation(), np.array([1.0, -2.0, 0.5])
    s, R_est, t_est = umeyama(src, (R @ src.T).T + t)
    assert s == 1.0
    np.testing.assert_allclose(R_est, R, atol=1e-10)
    np.testing.assert_allclose(t_est, t, atol=1e-10)


def test_ate_zero_for_rigidly_moved_trajectory():
    gt = circle_traj()
    est = rigidly_transform(gt, random_rotation(), np.array([10.0, 5.0, -1.0]))
    stats = ate_stats(gt[:, 1:4], est[:, 1:4])
    assert stats["rmse"] < 1e-9


def test_ate_matches_injected_noise_level():
    gt = circle_traj(n=2000)
    sigma = 0.05
    est = gt.copy()
    est[:, 1:4] += RNG.normal(0, sigma, size=(len(gt), 3))
    stats = ate_stats(gt[:, 1:4], est[:, 1:4])
    expected = sigma * np.sqrt(3)  # per-axis sigma -> 3D norm rmse
    assert stats["rmse"] == pytest.approx(expected, rel=0.1)


def test_associate_tolerates_small_offsets_only():
    t_ref = np.arange(100) * 0.1
    t_query = t_ref + 0.004
    ri, qi = associate(t_ref, t_query, max_diff=0.02)
    assert len(ri) == 100 and (ri == qi).all()
    ri, _ = associate(t_ref, t_ref + 0.05, max_diff=0.02)
    assert len(ri) == 0


def test_coverage_detects_tracking_dropout():
    gt = circle_traj(n=1000, dt=0.02)  # 20 s
    t_est = gt[:, 0]
    # estimator dies between 5 s and 10 s
    t_est = t_est[(t_est < 5.0) | (t_est >= 10.0)]
    stats = coverage_stats(gt[:, 0], t_est, gap_threshold=0.5)
    assert stats["gaps"] == 1
    assert stats["longest_gap"] == pytest.approx(5.0, abs=0.1)
    assert stats["coverage"] == pytest.approx(0.75, abs=0.02)


def test_end_to_end_cli_roundtrip(tmp_path):
    gt = circle_traj(n=800)
    est = rigidly_transform(gt, random_rotation(), np.array([2.0, 0.0, 3.0]))
    est[:, 1:4] += RNG.normal(0, 0.01, size=(len(gt), 3))
    write_tum(tmp_path / "gt.txt", gt)
    write_tum(tmp_path / "est.txt", est)
    m = evaluate(str(tmp_path / "gt.txt"), str(tmp_path / "est.txt"))
    assert m["ate"]["rmse"] < 0.05
    assert m["coverage"] > 0.99
    assert m["gaps"] == 0
    assert m["rpe"]["pairs"] > 0
    loaded = load_tum(str(tmp_path / "gt.txt"))
    assert loaded.shape == gt.shape
