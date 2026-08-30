"""Backend v2: sliding-window Levenberg-Marquardt over keyframe states with
multi-camera SMART rig factors (gtsam.SmartProjectionRigFactorPinholePoseCal3_S2
on normalized image coordinates; the spherical/bearing-only variant is the
intended target but its CameraSet is not constructible from Python in
gtsam 4.3a2 — a holder-type bug — so the fisheye rim beyond 80° off-axis
still waits; everything else below applies).

Why this replaces the explicit-landmark fixed-lag smoother (backend.py):
  * robustness — no landmark variables, so no singular linear systems; a
    degenerate or far landmark contributes zero, an outlying measurement is
    rejected inside the factor (dynamic outlier threshold); nothing throws,
    nothing needs a reset; a failed solve just keeps the IMU-predicted state;
  * lens-agnostic — the front-end hands over unit bearings, converted here to
    normalized coordinates (x/z, y/z) with a unit calibration; any lens model
    works, points inside 80° off-axis are used (rim: pending the spherical
    factor binding);
  * multi-camera by construction — one factor per landmark holds measurements
    from any camera of the rig (camera id + extrinsics), so stereo pairs, the
    3-cam triangle and the 6-cam ring are the same code path, and mono
    landmarks from motion parallax need no special init;
  * bounded compute — the window is rebuilt every keyframe from our own
    measurement memory (W keyframes), solved by LM with a fixed iteration cap:
    the cost does not grow with the flight, which is what a Jetson needs.
"""
from __future__ import annotations

import time

import numpy as np


class SmartBackend:
    def __init__(self, rig, lag_s=3.0, px_sigma=1.5, huber_k=1.345, max_iters=6,
                 outlier_thr_sigma=0.0, max_landmark_dist=40.0, verbose=False):
        import gtsam
        from gtsam.symbol_shorthand import B, V, X
        self.gtsam = gtsam
        self.X, self.V, self.B = X, V, B
        self.rig = rig
        self.lag_s = lag_s
        self.max_iters = max_iters
        self.verbose = verbose
        # camera rig with unit calibration: one camera per rig camera at its extrinsics
        self.K = gtsam.Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0)
        self.cam_set = gtsam.CameraSetPinholePoseCal3_S2()
        for c in rig.cameras:
            self.cam_set.push_back(gtsam.PinholePoseCal3_S2(gtsam.Pose3(c.T_imu_cam), self.K))
        self.cz = np.cos(np.deg2rad(80.0))
        # measurement noise in normalized coordinates: pixel sigma / focal length
        sig = float(np.mean([px_sigma / c.fx for c in rig.cameras]))
        # smart factors need an isotropic model; robustness to wrong associations comes
        # from the factor's dynamic outlier rejection (reprojection error threshold)
        self.noise = gtsam.noiseModel.Isotropic.Sigma(2, sig)
        p = gtsam.SmartProjectionParams()
        p.setLinearizationMode(gtsam.LinearizationMode.HESSIAN)
        p.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)
        p.setRankTolerance(1e-9)
        # gtsam's Python wrapper declares these two setters with *bool* arguments
        # (slam.i), so any float is coerced: 40.0 -> True -> threshold 1.0 m, which
        # flagged every landmark further than 1 m as FAR_POINT (error 0 -> IMU-only
        # drift). Pass False (= 0 = disabled) and do both checks in this class instead:
        # far points are gated by max_landmark_dist on our own triangulation, and
        # outliers are rejected in the front-end (fwd/bwd check, RANSAC, stereo verify).
        p.setLandmarkDistanceThreshold(False)
        p.setDynamicOutlierRejectionThreshold(False)
        self.max_landmark_dist = float(max_landmark_dist)
        self.outlier_thr = float(outlier_thr_sigma) * sig if outlier_thr_sigma > 0 else 0.0
        self.params = p
        # memory
        self.kf_t = {}                 # k -> time
        self.kf_state = {}             # k -> (Pose3, vel, bias)
        self.imu_factor = {}           # k -> CombinedImuFactor (k-1 -> k)
        self.lm_meas = {}              # landmark id -> [(k, cam, bearing)]
        self.landmark_t = {}           # landmark id -> last observation time
        self.landmark_obs = {}
        self.lm_point = {}             # landmark id -> last triangulated point (from the factor)
        self.prior = None              # (k0, pose, vel, bias, cov_pose, cov_vel, cov_bias) of the window's first state
        self.init_prior = None
        self.n_resets = 0; self.n_rebuilds = 0; self.n_retired = 0; self.n_dropped = 0
        self.n_failed_solves = 0
        self.n_active_lm = 0
        self.n_valid_lm = 0
        self.t_solve_ms = 0.0
        self.estimate = gtsam.Values()

    # ------------------------------------------------------------ interface
    def can_observe(self, bearing) -> bool:
        return bool(np.all(np.isfinite(bearing)) and bearing[2] > self.cz)

    def initialize(self, k, t, T_W_I, vel, bias, sigmas=(0.05, 0.01, 0.1, 0.1, 0.01)):
        g = self.gtsam
        pose = g.Pose3(T_W_I)
        self.kf_t[k] = t
        self.kf_state[k] = (pose, np.asarray(vel, float), bias)
        self.init_prior = (k, pose, np.asarray(vel, float), bias,
                           np.diag([sigmas[0], sigmas[0], 0.5, sigmas[1], sigmas[1], sigmas[1]]) ** 2,
                           np.eye(3) * sigmas[2] ** 2, np.diag([sigmas[3]] * 3 + [sigmas[4]] * 3) ** 2)
        self.prior = self.init_prior
        self.estimate.insert(self.X(k), pose); self.estimate.insert(self.V(k), np.asarray(vel, float)); self.estimate.insert(self.B(k), bias)

    def add_keyframe(self, k, t, pim, predicted, bias_prev_key):
        g = self.gtsam
        bias = self.kf_state[bias_prev_key][2] if bias_prev_key in self.kf_state else pim.biasHat()
        self.kf_t[k] = t
        self.kf_state[k] = (predicted.pose(), np.asarray(predicted.velocity()), bias)
        self.imu_factor[k] = g.CombinedImuFactor(self.X(k - 1), self.V(k - 1), self.X(k), self.V(k), self.B(k - 1), self.B(k), pim)

    def has_landmark(self, j): return j in self.lm_meas
    def landmark_alive(self, j, t_now): return j in self.landmark_t and (t_now - self.landmark_t[j]) < self.lag_s
    def add_landmark(self, j, point_w, t):
        self.lm_meas.setdefault(j, []); self.landmark_t[j] = t; self.landmark_obs[j] = 0
        self.lm_point[j] = np.asarray(point_w, float) if point_w is not None else None
    def drop_pending_landmark(self, j):
        self.lm_meas.pop(j, None); self.landmark_t.pop(j, None); self.landmark_obs.pop(j, None); self.lm_point.pop(j, None)
    def retire_landmark(self, j):
        self.drop_pending_landmark(j); self.n_retired += 1

    def add_observation(self, k, cam, j, bearing, t) -> bool:
        if not self.can_observe(bearing):
            return False
        b = np.asarray(bearing, float)
        self.lm_meas.setdefault(j, []).append((k, cam, np.array([b[0] / b[2], b[1] / b[2]])))
        self.landmark_t[j] = t
        self.landmark_obs[j] = self.landmark_obs.get(j, 0) + 1
        return True

    def landmark(self, j):
        return self.lm_point.get(j)

    # ---------------------------------------------------------------- solve
    def _window(self, t):
        cutoff = t - self.lag_s
        return sorted(k for k, tk in self.kf_t.items() if tk >= cutoff)

    def optimize(self, k, t):
        g = self.gtsam
        t0 = time.perf_counter()
        window = self._window(t)
        k0 = window[0]
        kset = set(window)
        graph = g.NonlinearFactorGraph()
        values = g.Values()
        for kk in window:
            pose, vel, bias = self.kf_state[kk]
            values.insert(self.X(kk), pose); values.insert(self.V(kk), vel); values.insert(self.B(kk), bias)
            if kk != k0 and kk in self.imu_factor and (kk - 1) in kset:
                graph.add(self.imu_factor[kk])
        # prior on the first window state (marginal of the previous window, or the init prior)
        pk, ppose, pvel, pbias, cp, cv, cb = self.prior if (self.prior and self.prior[0] == k0) else self._prior_for(k0)
        graph.add(g.PriorFactorPose3(self.X(k0), ppose, g.noiseModel.Gaussian.Covariance(cp)))
        graph.add(g.PriorFactorVector(self.V(k0), pvel, g.noiseModel.Gaussian.Covariance(cv)))
        graph.add(g.PriorFactorConstantBias(self.B(k0), pbias, g.noiseModel.Gaussian.Covariance(cb)))
        # smart factors: every landmark with >= 2 measurements inside the window
        factors = {}
        for j, meas in self.lm_meas.items():
            inwin = [(kk, c, b) for kk, c, b in meas if kk in kset]
            if len(inwin) < 2:
                continue
            f = g.SmartProjectionRigFactorPinholePoseCal3_S2(self.noise, self.cam_set, self.params)
            for kk, c, m in inwin:
                f.add(m, self.X(kk), int(c))
            graph.add(f); factors[j] = f
        self.n_active_lm = len(factors)
        params = g.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iters)
        params.setRelativeErrorTol(1e-4)
        params.setVerbosityLM("SILENT")
        try:
            result = g.LevenbergMarquardtOptimizer(graph, values, params).optimize()
            ok = all(np.all(np.isfinite(result.atPose3(self.X(kk)).matrix())) for kk in window)
        except Exception as e:
            ok = False
            if self.verbose:
                print(f"[smart] kf {k}: solve failed ({type(e).__name__}: {str(e)[:80]}) — keeping the predicted state")
        if ok:
            for kk in window:
                self.kf_state[kk] = (result.atPose3(self.X(kk)), np.asarray(result.atVector(self.V(kk))), result.atConstantBias(self.B(kk)))
            self.estimate = result
            # landmark points for diagnostics / cheirality checks
            self.n_valid_lm = 0
            for j, f in factors.items():
                try:
                    if f.isValid():
                        self.n_valid_lm += 1
                        pt = f.point()
                        pt = pt.get() if hasattr(pt, "get") else pt
                        pt = np.asarray(pt, dtype=float).reshape(-1)
                        if pt.shape == (3,) and np.all(np.isfinite(pt)):
                            self.lm_point[j] = pt
                except Exception:
                    pass
            # marginal prior for the next window's first state (only when the window will slide)
            self._update_prior(graph, result, window, t)
        else:
            self.n_failed_solves += 1
        # forget what fell out of the window
        cutoff = t - self.lag_s
        for kk in [kk for kk, tk in self.kf_t.items() if tk < cutoff - self.lag_s]:
            self.kf_t.pop(kk, None); self.kf_state.pop(kk, None); self.imu_factor.pop(kk, None)
        for j in [j for j, tj in self.landmark_t.items() if tj < cutoff]:
            self.drop_pending_landmark(j)
        for j, meas in self.lm_meas.items():
            self.lm_meas[j] = [m for m in meas if m[0] in kset or self.kf_t.get(m[0], 0) >= cutoff]
        self.t_solve_ms = (time.perf_counter() - t0) * 1e3
        return self.state(k)

    def _prior_for(self, k0):
        """Fallback prior when the window's first state has no stored marginal:
        loose position/velocity, tight rotation drift, tight bias."""
        pose, vel, bias = self.kf_state[k0]
        return (k0, pose, vel, bias, np.diag([0.02, 0.02, 0.02, 0.05, 0.05, 0.05]) ** 2,
                np.eye(3) * 0.1 ** 2, np.diag([0.05] * 3 + [0.005] * 3) ** 2)

    def _update_prior(self, graph, result, window, t):
        """Marginal covariance of the keyframe that will be the window's first state next
        time (the second-oldest now) — an approximate marginalisation prior."""
        g = self.gtsam
        if len(window) < 2:
            return
        knext = window[1]
        try:
            marg = g.Marginals(graph, result)
            cp = marg.marginalCovariance(self.X(knext)); cv = marg.marginalCovariance(self.V(knext)); cb = marg.marginalCovariance(self.B(knext))
            pose, vel, bias = self.kf_state[knext]
            self.prior = (knext, pose, vel, bias, cp, cv, cb)
        except Exception as e:
            self.prior = None
            if self.verbose:
                print(f"[smart] marginals failed ({type(e).__name__}); fallback prior next window")

    def state(self, k):
        pose, vel, bias = self.kf_state[k]
        return pose.matrix(), np.asarray(vel), bias

    def navstate(self, k):
        pose, vel, _ = self.kf_state[k]
        return self.gtsam.NavState(pose, vel)
