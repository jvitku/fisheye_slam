"""Tests for bench.fly_trajectory path generation (no MAVLink)."""

import numpy as np
import pytest

from bench.fly_trajectory import PATTERNS, setpoints


@pytest.mark.parametrize("pattern,speed", [("slow_scan", 0.6), ("fast_yaw", 1.5)])
def test_pattern_speed_and_continuity(pattern, speed):
    rate, duration = 20.0, 120.0
    pts = np.array([[t, x, y, z, yaw] for t, x, y, z, yaw in setpoints(pattern, duration, rate)])
    assert len(pts) == duration * rate
    steps = np.linalg.norm(np.diff(pts[:, 1:4], axis=0), axis=1)
    # no jumps: never more than 1.5x the nominal step (turns included)
    assert steps.max() <= 1.5 * speed / rate + 1e-9
    # mean speed after the ramp-in is the nominal one (within 10%)
    after_ramp = pts[:, 0] > 5.0
    mean = steps[after_ramp[1:]].sum() / (pts[after_ramp, 0][-1] - pts[after_ramp, 0][0])
    assert mean == pytest.approx(speed, rel=0.10)
    assert np.allclose(pts[:, 3], 1.5)                       # constant altitude


def test_slow_scan_footprint_is_bounded():
    f = PATTERNS["slow_scan"]
    pts = np.array([f(t)[:2] for t in np.arange(0.0, 600.0, 0.05)])
    assert pts[:, 0].min() >= -1e-9 and pts[:, 0].max() <= 6.0 + 1e-9
    assert pts[:, 1].min() >= -1e-9 and pts[:, 1].max() <= 4 * 1.5 + 1e-9   # 5 rows


def test_fast_yaw_heading_sweeps():
    f = PATTERNS["fast_yaw"]
    yaws = np.array([f(t)[3] for t in np.arange(0.0, 30.0, 0.05)])
    # a figure-8 covers the full heading circle
    assert yaws.min() < -2.5 and yaws.max() > 2.5


def test_setpoints_ramp_starts_at_path_origin():
    first = next(iter(setpoints("fast_yaw", 10.0)))
    x0, y0, z0, yaw0 = PATTERNS["fast_yaw"](0.0)
    assert first[1:] == pytest.approx((x0, y0, z0, yaw0))
