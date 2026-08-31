"""DynamicInitializer solver tests on synthetic trajectories.

The solver consumes (a) raw IMU samples and (b) per-frame metric landmark
points in the body frame (in production: init-time cross-camera LK matches).
It must recover gravity direction, the velocity at the last frame and the
frame-0-anchored pose chain — on a platform that is already flying.
World convention: +Z up, gravity g_w = (0, 0, -9.81).
"""
from __future__ import annotations

import numpy as np
import pytest

from podslam.geometry import exp_so3
from podslam.imu import ImuBuffer
from podslam.init_dynamic import DynamicInitializer

G = 9.81
RNG = np.random.default_rng(7)


def make_traj(vel0=(0.4, -0.3, 0.15)):
    """Smooth flying trajectory with analytic derivatives + slow attitude motion."""
    v0 = np.asarray(vel0, float)

    def p(t):
        return np.array([0.9 * np.sin(1.1 * t) + v0[0] * t,
                         v0[1] * t + 0.25 * t * t * 0.3,
                         1.5 + 0.35 * np.sin(0.8 * t) + v0[2] * t])

    def v(t):
        return np.array([0.9 * 1.1 * np.cos(1.1 * t) + v0[0],
                         v0[1] + 0.15 * t,
                         0.35 * 0.8 * np.cos(0.8 * t) + v0[2]])

    def a(t):
        return np.array([-0.9 * 1.1 ** 2 * np.sin(1.1 * t),
                         0.15,
                         -0.35 * 0.8 ** 2 * np.sin(0.8 * t)])

    def R_wb(t):   # body-to-world attitude: modest rocking + slow yaw (a real hover)
        return exp_so3([0.12 * np.sin(0.9 * t), 0.10 * np.sin(0.7 * t + 0.5), 0.3 * t * 0.2])

    return p, v, a, R_wb


def synth_imu(buf, p, v, a, R_wb, t0, t1, rate=400.0, gyro_bias=np.zeros(3)):
    """Fill an ImuBuffer with ideal samples of the analytic trajectory."""
    g_w = np.array([0.0, 0.0, -G])
    dt = 1.0 / rate
    ts = np.arange(t0, t1 + dt / 2, dt)
    for t in ts:
        # gyro from the attitude finite difference (matches how minisim makes IMU)
        dR = R_wb(t).T @ R_wb(t + 1e-4)
        w_hat = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2.0
        gyro = w_hat / 1e-4
        accel = R_wb(t).T @ (a(t) - g_w)
        buf.append(float(t), gyro + gyro_bias, accel)
    return buf


def frame_points(p, R_wb, t, landmarks, noise=0.0):
    pts = {}
    Rbw = R_wb(t).T
    for tid, X in enumerate(landmarks):
        pb = Rbw @ (X - p(t))
        if noise > 0:
            pb = pb + RNG.normal(0.0, noise, 3)
        pts[tid] = pb
    return pts


def run_init(noise=0.0, gyro_bias=np.zeros(3), n_pts=150, t0=5.0, window_s=1.5,
             fps=10.0, vel0=(0.4, -0.3, 0.15)):
    p, v, a, R_wb = make_traj(vel0)
    landmarks = RNG.uniform(-6, 6, (n_pts, 3)) + [0, 0, 1.5]
    keep = np.linalg.norm(landmarks - p(t0), axis=1) > 1.0
    landmarks = landmarks[keep]
    buf = ImuBuffer()
    synth_imu(buf, p, v, a, R_wb, t0 - 0.2, t0 + window_s + 0.2, gyro_bias=gyro_bias)
    ini = DynamicInitializer(buf, window_s=window_s, min_frames=6)
    ts = np.arange(t0, t0 + window_s + 1e-9, 1.0 / fps)
    for t in ts:
        ini.add_frame_points(float(t), frame_points(p, R_wb, t, landmarks, noise))
    assert ini.ready
    r = ini.solve()
    assert r is not None, "solver returned None on a healthy window"
    t_last = ts[-1]
    return r, p, v, R_wb, t_last


def gravity_err_deg(r, R_wb, t_last):
    # world up must map to true world up through the estimated R_W_I and the true attitude:
    # R_W_I maps body -> estimated world; true body up in estimated world:
    up_est_world = r["R_W_I"] @ (R_wb(t_last).T @ np.array([0.0, 0.0, 1.0]))
    return np.degrees(np.arccos(np.clip(up_est_world @ np.array([0.0, 0.0, 1.0]), -1, 1)))


class TestSolverClean:
    def test_gravity_direction(self):
        r, p, v, R_wb, t_last = run_init(noise=0.0)
        assert gravity_err_deg(r, R_wb, t_last) < 0.5

    def test_velocity(self):
        r, p, v, R_wb, t_last = run_init(noise=0.0)
        # estimated world differs from the true world by a yaw; compare magnitudes
        # and the vertical component (both yaw-invariant)
        v_true_w = v(t_last)
        assert abs(np.linalg.norm(r["v_W"]) - np.linalg.norm(v_true_w)) < 0.05
        assert abs(r["v_W"][2] - v_true_w[2]) < 0.05

    def test_gravity_magnitude_estimate(self):
        r, *_ = run_init(noise=0.0)
        assert abs(r["g_norm"] - G) < 0.3


class TestSolverNoisy:
    def test_noisy_points(self):
        # 25 cm point noise ~ the triangulation error of a 10 cm baseline at 3 m
        r, p, v, R_wb, t_last = run_init(noise=0.25)
        assert gravity_err_deg(r, R_wb, t_last) < 2.0
        assert abs(np.linalg.norm(r["v_W"]) - np.linalg.norm(v(t_last))) < 0.25
        assert abs(r["v_W"][2] - v(t_last)[2]) < 0.25

    def test_gyro_bias_tolerated(self):
        r, p, v, R_wb, t_last = run_init(noise=0.1, gyro_bias=np.array([0.004, -0.006, 0.005]))
        assert gravity_err_deg(r, R_wb, t_last) < 2.0

    def test_hover(self):
        # near-static flight: must still produce a sane gravity (quasi-static case)
        r, p, v, R_wb, t_last = run_init(noise=0.1, vel0=(0.02, 0.0, 0.0))
        assert gravity_err_deg(r, R_wb, t_last) < 2.0


class TestSolverDegenerate:
    def test_too_few_frames(self):
        p, v, a, R_wb = make_traj()
        buf = ImuBuffer()
        synth_imu(buf, p, v, a, R_wb, 4.8, 5.4)
        ini = DynamicInitializer(buf, window_s=1.5, min_frames=6)
        landmarks = RNG.uniform(-5, 5, (60, 3))
        for t in (5.0, 5.1, 5.2):
            ini.add_frame_points(t, frame_points(p, R_wb, t, landmarks))
        assert not ini.ready
        assert ini.solve() is None

    def test_chain_break_returns_none(self):
        p, v, a, R_wb = make_traj()
        buf = ImuBuffer()
        synth_imu(buf, p, v, a, R_wb, 4.8, 7.0)
        ini = DynamicInitializer(buf, window_s=1.5, min_frames=6)
        landmarks = RNG.uniform(-5, 5, (80, 3)) + [0, 0, 1.5]
        ts = np.arange(5.0, 6.6, 0.1)
        for n, t in enumerate(ts):
            pts = frame_points(p, R_wb, float(t), landmarks)
            if n >= 8:      # disjoint ids after frame 8: no common points across the break
                pts = {tid + 10_000: pb for tid, pb in pts.items()}
            ini.add_frame_points(float(t), pts)
        assert ini.solve() is None
