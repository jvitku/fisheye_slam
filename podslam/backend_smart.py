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
                 outlier_thr_sigma=0.0, max_landmark_dist=40.0, abs_err_tol=1e-2, marg_mode="all",
                 chi2_gate=5.0, min_obs_prior=4, epi=False, max_window_kf=32, max_obs_angle_deg=80.0,
                 px_sigma_adapt=True, verbose=False):
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
        self.cz = np.cos(np.deg2rad(float(max_obs_angle_deg)))   # observation cone about the optical axis
        # measurement noise in normalized coordinates: pixel sigma / focal length
        sig = float(np.mean([px_sigma / c.fx for c in rig.cameras]))
        # smart factors need an isotropic model; robustness to wrong associations comes
        # from the factor's dynamic outlier rejection (reprojection error threshold)
        self.noise = gtsam.noiseModel.Isotropic.Sigma(2, sig)
        # Adaptive pixel noise: after each solve the median whitened error per measurement
        # of the live factors is compared with its expectation (chi2 with 2 dof, halved:
        # median 0.69) and sigma is nudged so the residual statistics match.  The right
        # sigma is rig- and condition-specific (TUM-VI 512x512 ~1.5 px; Hilti 720x540 KLT
        # under fisheye distortion ~3 px: 14.9 -> 8.9 cm when set by hand).
        self.px_sigma_adapt = bool(px_sigma_adapt)
        self.px_sigma = float(px_sigma)
        self.inv_f = sig / float(px_sigma)
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
        # Refine the implicit landmark nonlinearly (Gauss-Newton on the reprojection error)
        # instead of using the raw DLT point: the DLT solution is biased towards the
        # cameras with noisy pixels, which showed up as a ~1.3 % scale under-estimate.
        p.setEnableEPI(bool(epi))
        p.setLandmarkDistanceThreshold(False)
        p.setDynamicOutlierRejectionThreshold(False)
        self.max_landmark_dist = float(max_landmark_dist)
        self.abs_err_tol = float(abs_err_tol)
        self.max_window_kf = int(max_window_kf)
        # 'ended': tracks that ended go into the marginal prior with all their observations,
        #          live tracks keep their (re-linearised) factor and drop the marginalised
        #          keyframes' observations;  'all': every landmark seen from a marginalised
        #          keyframe is absorbed (MSCKF-style);  'pin': legacy over-confident pin.
        self.marg_mode = marg_mode
        self.n_lm_iters = 0
        # Outlier gate on the whitened reprojection error per measurement, evaluated at
        # the CONVERGED window solution (0.5*chi2 with 2 dof: inliers ~1, 99% < 4.6):
        # a landmark above it is retired, and a track that ended is only absorbed into
        # the marginal prior if it passes the gate and has >= min_obs_prior measurements.
        self.chi2_gate = float(chi2_gate)
        self.min_obs_prior = int(min_obs_prior)
        self.n_outliers = 0; self.n_prior_absorbed = 0; self.n_prior_rejected = 0
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
        # Marginalisation prior: nonlinear factors (the init priors, then
        # LinearContainerFactors = Schur complements of every marginalised keyframe)
        # over keyframes that are still in the window.  graph_keys = keyframes that
        # are currently part of the estimator (not yet marginalised).
        self.prior_factors = []
        self.graph_keys = set()
        self.n_marginalized = 0
        self.n_marg_fallback = 0
        self.n_resets = 0; self.n_rebuilds = 0; self.n_retired = 0; self.n_dropped = 0
        self.n_failed_solves = 0
        self.n_active_lm = 0
        self.n_valid_lm = 0
        self.t_solve_ms = 0.0
        self.estimate = gtsam.Values()

    # ------------------------------------------------------------ interface
    def can_observe(self, bearing) -> bool:
        return bool(np.all(np.isfinite(bearing)) and bearing[2] > self.cz)

    def initialize(self, k, t, T_W_I, vel, bias, sigmas=(1e-3, 0.01, 0.1, 0.1, 0.01), yaw_sigma=1e-3):
        g = self.gtsam
        pose = g.Pose3(T_W_I)
        self.kf_t[k] = t
        self.kf_state[k] = (pose, np.asarray(vel, float), bias)
        # gauge anchor: the first pose's position and yaw are fixed hard (unobservable
        # otherwise they random-walk through the re-linearised marginal prior)
        cp = np.diag([sigmas[1], sigmas[1], yaw_sigma, sigmas[0], sigmas[0], sigmas[0]]) ** 2
        cv = np.eye(3) * sigmas[2] ** 2
        cb = np.diag([sigmas[3]] * 3 + [sigmas[4]] * 3) ** 2
        self.prior_factors = [g.PriorFactorPose3(self.X(k), pose, g.noiseModel.Gaussian.Covariance(cp)),
                              g.PriorFactorVector(self.V(k), np.asarray(vel, float), g.noiseModel.Gaussian.Covariance(cv)),
                              g.PriorFactorConstantBias(self.B(k), bias, g.noiseModel.Gaussian.Covariance(cb))]
        self.graph_keys = {k}
        self.estimate.insert(self.X(k), pose); self.estimate.insert(self.V(k), np.asarray(vel, float)); self.estimate.insert(self.B(k), bias)

    def add_keyframe(self, k, t, pim, predicted, bias_prev_key):
        g = self.gtsam
        bias = self.kf_state[bias_prev_key][2] if bias_prev_key in self.kf_state else pim.biasHat()
        self.kf_t[k] = t
        self.kf_state[k] = (predicted.pose(), np.asarray(predicted.velocity()), bias)
        self.graph_keys.add(k)
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

    def _smart_factor(self, meas):
        g = self.gtsam
        f = g.SmartProjectionRigFactorPinholePoseCal3_S2(self.noise, self.cam_set, self.params)
        for kk, c, m in meas:
            f.add(m, self.X(kk), int(c))
        return f

    def _values(self, keys):
        v = self.gtsam.Values()
        for kk in keys:
            pose, vel, bias = self.kf_state[kk]
            v.insert(self.X(kk), pose); v.insert(self.V(kk), vel); v.insert(self.B(kk), bias)
        return v

    def _marginalize(self, gone, window, t_now):
        """Schur-complement the keyframes in `gone` out of the estimator (VINS/OKVIS style):
        every factor that touches them — the current prior, the IMU factors and the smart
        factors of landmarks seen from them — is linearised at the last solution and
        eliminated; the remaining Gaussian factors over the window states become the new
        prior (LinearContainerFactors, re-linearised by shifting their rhs when the window
        states move).  Landmark observations that went into the prior are consumed: later
        observations of the same landmark start a fresh factor, so nothing is counted twice."""
        g = self.gtsam
        gone = sorted(gone)
        gset = set(gone)
        keys_m = set()
        for kk in gone:
            keys_m.update((self.X(kk), self.V(kk), self.B(kk)))
        values = self._values(gone + list(window))
        gfg = g.GaussianFactorGraph()
        kept = []
        for f in self.prior_factors:
            if any(key in keys_m for key in f.keys()):
                gfg.add(f.linearize(values))
            else:
                kept.append(f)
        for kk in list(window) + gone:
            f = self.imu_factor.get(kk)
            if f is not None and (kk in gset or (kk - 1) in gset) and (kk - 1) in self.kf_state:
                gfg.add(f.linearize(values))
        # Landmarks seen from a marginalised keyframe: a track that has ENDED goes into
        # the prior with all its observations (MSCKF-style: full information, then
        # dropped); a track that is still alive keeps its nonlinear factor and merely
        # loses the observations at the marginalised keyframes (one measurement of
        # information lost, nothing counted twice, no frozen linearisation).
        consumed = []
        wset = set(window)
        for j, meas in self.lm_meas.items():
            if not any(kk in gset for kk, _, _ in meas):
                continue
            if self.marg_mode != "all" and self.landmark_t.get(j, -1.0) >= t_now:      # still tracked
                self.lm_meas[j] = [(kk, c, m) for kk, c, m in meas if kk not in gset]
                continue
            inside = [(kk, c, m) for kk, c, m in meas if kk in gset or kk in wset]
            if len(inside) >= max(2, self.min_obs_prior):
                f = self._smart_factor(inside)
                try:
                    if f.error(values) / len(inside) <= self.chi2_gate:
                        lin = f.linearize(values)
                        if lin is not None:
                            gfg.add(lin); self.n_prior_absorbed += 1
                    else:
                        self.n_prior_rejected += 1
                except Exception:
                    self.n_prior_rejected += 1
            consumed.append(j)
        present = set(gfg.keys()) if hasattr(gfg, "keys") else keys_m
        order = g.Ordering([key for key in [self.X(kk) for kk in gone] + [self.V(kk) for kk in gone] + [self.B(kk) for kk in gone] if key in present])
        new_prior = []
        if order.size() > 0:
            _, rem = gfg.eliminatePartialMultifrontal(order)
            for i in range(rem.size()):
                lf = rem.at(i)
                if lf is None or lf.size() == 0:
                    continue
                new_prior.append(g.LinearContainerFactor(lf, values))
        self.prior_factors = kept + new_prior
        for j in consumed:
            self.drop_pending_landmark(j)     # its information now lives in the prior
        for kk in gone:
            self.graph_keys.discard(kk)
            self.kf_state.pop(kk, None); self.imu_factor.pop(kk, None); self.kf_t.pop(kk, None)
        self.n_marginalized += len(gone)

    def optimize(self, k, t):
        g = self.gtsam
        t0 = time.perf_counter()
        # every keyframe that is not marginalised yet is in the graph; the ones older
        # than the lag are marginalised AFTER this solve, at their solved values
        window = sorted(self.graph_keys)
        kset = set(window)
        k0 = window[0]
        graph = g.NonlinearFactorGraph()
        values = self._values(window)
        for kk in window:
            if kk != k0 and kk in self.imu_factor and (kk - 1) in kset:
                graph.add(self.imu_factor[kk])
        if not self.prior_factors:
            self._fallback_prior([], window)
        for f in self.prior_factors:
            graph.add(f)
        # smart factors: every landmark with >= 2 measurements inside the window
        factors = {}
        for j, meas in self.lm_meas.items():
            inwin = [(kk, c, b) for kk, c, b in meas if kk in kset]
            if len(inwin) < 2:
                continue
            f = self._smart_factor(inwin)
            graph.add(f); factors[j] = f
        self.n_active_lm = len(factors)
        params = g.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iters)
        # The marginal prior carries a large constant error term, so a *relative*
        # decrease test stops LM after one iteration long before the state has
        # converged (that was a 1 cm/keyframe lag).  Stop on the absolute decrease.
        params.setRelativeErrorTol(0.0)
        params.setAbsoluteErrorTol(float(self.abs_err_tol))
        params.setVerbosityLM("SILENT")
        try:
            opt = g.LevenbergMarquardtOptimizer(graph, values, params)
            result = opt.optimize()
            self.n_lm_iters = opt.iterations()
            ok = all(np.all(np.isfinite(result.atPose3(self.X(kk)).matrix())) for kk in window)
        except Exception as e:
            ok = False
            if self.verbose:
                print(f"[smart] kf {k}: solve failed ({type(e).__name__}: {str(e)[:80]}) — keeping the predicted state")
        if ok:
            for kk in window:
                self.kf_state[kk] = (result.atPose3(self.X(kk)), np.asarray(result.atVector(self.V(kk))), result.atConstantBias(self.B(kk)))
            self.estimate = result
            # landmark points for diagnostics / cheirality checks + outlier gate
            self.n_valid_lm = 0
            outliers = []
            for j, f in factors.items():
                try:
                    if f.error(result) / max(1, len(f.measured())) > self.chi2_gate:
                        outliers.append(j); continue
                    if f.isValid():
                        self.n_valid_lm += 1
                        pt = f.point()
                        pt = pt.get() if hasattr(pt, "get") else pt
                        pt = np.asarray(pt, dtype=float).reshape(-1)
                        if pt.shape == (3,) and np.all(np.isfinite(pt)):
                            self.lm_point[j] = pt
                except Exception:
                    pass
            for j in outliers:
                self.drop_pending_landmark(j)
            self.n_outliers += len(outliers)
            if self.px_sigma_adapt and len(factors) >= 20:
                errs = []
                for j, f in factors.items():
                    try:
                        n_m = len(f.measured())
                        if n_m < 6:
                            continue                                # the implicit landmark absorbs 3 dof
                        dof = 2 * n_m - 3                           # residual dof of one landmark's block
                        e = f.error(result) / (0.5 * dof)           # error = 0.5*chi2 -> per-dof ratio
                        if np.isfinite(e) and e > 0:
                            errs.append(e)
                    except Exception:
                        pass
                if len(errs) >= 20:
                    ratio = float(np.median(errs)) / 0.93           # chi2_k/k median for k ~ 9-40 dof
                    target = float(np.clip(self.px_sigma * np.sqrt(ratio), 0.5, 6.0))
                    self.px_sigma = 0.9 * self.px_sigma + 0.1 * target        # slow, stable
                    self.noise = g.noiseModel.Isotropic.Sigma(2, self.px_sigma * self.inv_f)
        else:
            self.n_failed_solves += 1
        # slide: marginalise the keyframes that fell out of the lag window
        cutoff = t - self.lag_s
        n_over = max(0, len(window) - self.max_window_kf)
        gone = [kk for i, kk in enumerate(window) if self.kf_t[kk] < cutoff or i < n_over]
        remaining = [kk for kk in window if kk not in set(gone)]
        if gone and remaining and self.marg_mode == "pin" and ok:
            # reference: the old over-confident pin (full-window marginal covariances)
            kn = remaining[0]
            try:
                marg = g.Marginals(graph, result)
                pose, vel, bias = self.kf_state[kn]
                self.prior_factors = [g.PriorFactorPose3(self.X(kn), pose, g.noiseModel.Gaussian.Covariance(marg.marginalCovariance(self.X(kn)))),
                                      g.PriorFactorVector(self.V(kn), vel, g.noiseModel.Gaussian.Covariance(marg.marginalCovariance(self.V(kn)))),
                                      g.PriorFactorConstantBias(self.B(kn), bias, g.noiseModel.Gaussian.Covariance(marg.marginalCovariance(self.B(kn))))]
            except Exception:
                self._fallback_prior([], remaining)
            for kk in gone:
                self.graph_keys.discard(kk); self.kf_state.pop(kk, None); self.imu_factor.pop(kk, None); self.kf_t.pop(kk, None)
            self.n_marginalized += len(gone)
        elif gone and remaining:
            try:
                self._marginalize(gone, remaining, t)
            except Exception as e:
                self.n_marg_fallback += 1
                if self.verbose:
                    print(f"[smart] kf {k}: marginalisation failed ({type(e).__name__}: {str(e)[:80]}); fallback prior")
                self._fallback_prior(gone, remaining)
        # forget landmarks that have not been seen for a whole window
        for j in [j for j, tj in self.landmark_t.items() if tj < cutoff]:
            self.drop_pending_landmark(j)
        rset = set(remaining)
        for j, meas in self.lm_meas.items():
            self.lm_meas[j] = [m for m in meas if m[0] in rset]
        self.t_solve_ms = (time.perf_counter() - t0) * 1e3
        return self.state(k)

    def _fallback_prior(self, gone, window):
        """Safety net when the marginalisation fails: pin the oldest window state with
        loose position/velocity, tight rotation drift and tight bias (loses the past's
        information, never counts anything twice)."""
        g = self.gtsam
        k0 = window[0]
        pose, vel, bias = self.kf_state[k0]
        self.prior_factors = [
            g.PriorFactorPose3(self.X(k0), pose, g.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05]))),
            g.PriorFactorVector(self.V(k0), vel, g.noiseModel.Isotropic.Sigma(3, 0.1)),
            g.PriorFactorConstantBias(self.B(k0), bias, g.noiseModel.Diagonal.Sigmas(np.array([0.05] * 3 + [0.005] * 3)))]
        for kk in gone:
            self.graph_keys.discard(kk)
            self.kf_state.pop(kk, None); self.imu_factor.pop(kk, None); self.kf_t.pop(kk, None)

    def state(self, k):
        pose, vel, bias = self.kf_state[k]
        return pose.matrix(), np.asarray(vel), bias

    def navstate(self, k):
        pose, vel, _ = self.kf_state[k]
        return self.gtsam.NavState(pose, vel)
