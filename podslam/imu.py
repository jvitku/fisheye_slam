"""IMU handling: sample buffer, static initialisation, GTSAM preintegration,
gyro-only rotation prediction for the front-end."""
from __future__ import annotations

import numpy as np

from .geometry import exp_so3, rotation_aligning

G = 9.81


class ImuBuffer:
    """Time-ordered IMU samples (t [s], gyro [rad/s], accel [m/s^2])."""

    def __init__(self, keep_s: float = 10.0):
        self.t: list = []; self.w: list = []; self.a: list = []
        self.keep_s = keep_s

    def append(self, t: float, gyro, accel) -> None:
        if self.t and t <= self.t[-1]:
            return                                   # drop out-of-order duplicates
        self.t.append(float(t)); self.w.append(np.asarray(gyro, float)); self.a.append(np.asarray(accel, float))
        if len(self.t) > 4 and self.t[-1] - self.t[0] > 2 * self.keep_s:
            cut = np.searchsorted(np.asarray(self.t), self.t[-1] - self.keep_s)
            del self.t[:cut]; del self.w[:cut]; del self.a[:cut]

    def between(self, t0: float, t1: float):
        """Samples with t0 < t <= t1 plus the sample before t0 (for dt)."""
        t = np.asarray(self.t)
        i0 = int(np.searchsorted(t, t0, side="right"))
        i1 = int(np.searchsorted(t, t1, side="right"))
        return t[i0:i1], np.asarray(self.w[i0:i1]).reshape(-1, 3), np.asarray(self.a[i0:i1]).reshape(-1, 3)

    def latest(self) -> float | None:
        return self.t[-1] if self.t else None


def delta_rotation(buf: ImuBuffer, t0: float, t1: float, gyro_bias) -> np.ndarray:
    """R_{I0 <- I1}: rotation of the body between t0 and t1 from the gyro alone."""
    ts, ws, _ = buf.between(t0, t1)
    R = np.eye(3)
    t_prev = t0
    for t, w in zip(ts, ws):
        dt = float(t - t_prev)
        if dt > 0:
            R = R @ exp_so3((w - gyro_bias) * dt)
        t_prev = t
    if t1 > t_prev and len(ws):
        R = R @ exp_so3((ws[-1] - gyro_bias) * (t1 - t_prev))
    return R


class StaticInitializer:
    """Wait for a still period, then estimate gravity direction + gyro bias.
    Falls back to 'assume the first window is still' after max_wait_s."""

    def __init__(self, window_s=1.0, gyro_thr=0.03, accel_std_thr=0.35, max_wait_s=3.0):
        self.window_s, self.gyro_thr, self.accel_std_thr, self.max_wait_s = window_s, gyro_thr, accel_std_thr, max_wait_s
        self.t: list = []; self.w: list = []; self.a: list = []
        self.result = None

    def feed(self, t, gyro, accel) -> bool:
        if self.result is not None:
            return True
        self.t.append(float(t)); self.w.append(np.asarray(gyro, float)); self.a.append(np.asarray(accel, float))
        t_arr = np.asarray(self.t)
        if t_arr[-1] - t_arr[0] < self.window_s:
            return False
        # most recent window
        i0 = int(np.searchsorted(t_arr, t_arr[-1] - self.window_s))
        w = np.asarray(self.w[i0:]); a = np.asarray(self.a[i0:])
        still = (np.linalg.norm(w, axis=1).max() < self.gyro_thr) and (a.std(axis=0).max() < self.accel_std_thr)
        forced = (t_arr[-1] - t_arr[0]) > self.max_wait_s
        if still or forced:
            self._finalize(t_arr[-1], w, a, forced=bool(forced and not still))
            return True
        return False

    def _finalize(self, t, w, a, forced):
        a_mean = a.mean(axis=0)
        g_body = a_mean / max(np.linalg.norm(a_mean), 1e-9)          # "up" in body coordinates
        R_W_I = rotation_aligning(g_body, [0.0, 0.0, 1.0])           # maps body up -> world +Z
        self.result = dict(t=float(t), R_W_I=R_W_I, gyro_bias=w.mean(axis=0),
                           accel_bias=np.zeros(3), forced=forced,
                           accel_norm=float(np.linalg.norm(a_mean)))

    def force(self, t=None):
        """Assume the most recent window was still, regardless of motion — the legacy
        fallback, invoked explicitly when dynamic initialisation starves."""
        if self.result is not None or not self.t:
            return self.result
        t_arr = np.asarray(self.t)
        i0 = int(np.searchsorted(t_arr, t_arr[-1] - self.window_s))
        self._finalize(t if t is not None else t_arr[-1],
                       np.asarray(self.w[i0:]), np.asarray(self.a[i0:]), forced=True)
        return self.result


class Preintegrator:
    """gtsam.PreintegratedCombinedMeasurements between two keyframes."""

    # Estimator-side inflation of the datasheet noise densities (accel, gyro, accel walk,
    # gyro walk).  The datasheet values make a 200 Hz IMU pin every 0.15 s keyframe
    # translation to ~0.1 mm, so unmodelled effects (scale factor, misalignment,
    # vibration) dictate the trajectory scale (-1.3 % on TUM-VI) and the gyro bias
    # cannot follow its drift.  Basalt's TUM-VI values, verified on room1:
    # 16.5 -> 8.8 cm ATE, scale 0.99-1.00, yaw drift 2.9 -> 0.9 deg/min.
    DEFAULT_NOISE_SCALE = (5.7, 1.8, 1.2, 4.5)

    def __init__(self, imu, bias=None, noise_scale=None):
        import gtsam
        self.gtsam = gtsam
        ka, kg, kaw, kgw = noise_scale if noise_scale is not None else self.DEFAULT_NOISE_SCALE
        p = gtsam.PreintegrationCombinedParams.MakeSharedU(G)
        p.setGyroscopeCovariance(np.eye(3) * (kg * imu.gyro_noise_density) ** 2)
        p.setAccelerometerCovariance(np.eye(3) * (ka * imu.accel_noise_density) ** 2)
        p.setIntegrationCovariance(np.eye(3) * 1e-8)
        p.setBiasAccCovariance(np.eye(3) * (kaw * imu.accel_random_walk) ** 2)
        p.setBiasOmegaCovariance(np.eye(3) * (kgw * imu.gyro_random_walk) ** 2)
        if hasattr(p, "setBiasAccOmegaInit"):          # dropped in gtsam 4.3
            p.setBiasAccOmegaInit(np.eye(6) * 1e-5)
        self.params = p
        self.bias = bias if bias is not None else gtsam.imuBias.ConstantBias()
        self.pim = gtsam.PreintegratedCombinedMeasurements(p, self.bias)
        self.t_last = None

    def delta_rotvec(self):
        """Rotation vector integrated since the last reset (radians)."""
        return np.asarray(self.gtsam.Rot3.Logmap(self.pim.deltaRij()))

    def reset(self, bias, t: float) -> None:
        self.bias = bias
        self.pim = self.gtsam.PreintegratedCombinedMeasurements(self.params, bias)
        self.t_last = t

    def integrate_until(self, buf: ImuBuffer, t1: float) -> None:
        """Integrate all buffered samples in (t_last, t1]."""
        if self.t_last is None:
            self.t_last = t1
            return
        ts, ws, as_ = buf.between(self.t_last, t1)
        for t, w, a in zip(ts, ws, as_):
            dt = float(t - self.t_last)
            if dt > 0:
                self.pim.integrateMeasurement(a, w, dt)
                self.t_last = float(t)
        if t1 > self.t_last and len(ws):                    # hold the last sample to t1
            self.pim.integrateMeasurement(as_[-1], ws[-1], float(t1 - self.t_last))
            self.t_last = float(t1)

    def predict(self, navstate):
        return self.pim.predict(navstate, self.bias)

    @property
    def dt(self) -> float:
        return float(self.pim.deltaTij())
