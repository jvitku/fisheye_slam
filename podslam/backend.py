"""Estimator: a fixed-lag smoother (GTSAM, BSD) over keyframe states
(pose, velocity, IMU bias), combined IMU factors between keyframes and
projection factors on normalized image coordinates for triangulated landmarks.
Lens models are handled by the front-end (bearings in), so any camera works
here. Old states are marginalised by the smoother; the world frame is NEVER
re-initialised — on solver trouble we rebuild the graph around the last good
estimate (podslam requirement #1 from the cuVSLAM diagnosis)."""
from __future__ import annotations

import re

import numpy as np

MAX_THETA_DEG = 80.0    # projection factors only inside this half-angle (perspective division)


class Backend:
    def __init__(self, rig, lag_s=4.0, px_sigma=1.5, huber_k=1.345, verbose=False):
        import gtsam
        from gtsam.symbol_shorthand import B, L, V, X
        # fixed-lag smoothers: gtsam core (>= 4.3) or gtsam_unstable (4.2)
        gu = gtsam if hasattr(gtsam, "IncrementalFixedLagSmoother") else __import__("gtsam_unstable")
        self.gtsam, self.gu = gtsam, gu
        self.X, self.V, self.B, self.L = X, V, B, L
        self.rig = rig
        self.lag_s = lag_s
        self.verbose = verbose
        self.K = gtsam.Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0)
        self.body_P_sensor = [gtsam.Pose3(c.T_imu_cam) for c in rig.cameras]
        self.px_noise = []
        for c in rig.cameras:
            base = gtsam.noiseModel.Isotropic.Sigma(2, px_sigma / c.fx)
            self.px_noise.append(gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(huber_k), base))
        self.cz = np.cos(np.deg2rad(MAX_THETA_DEG))
        self._new_smoother()
        self.estimate = gtsam.Values()
        self.landmark_t = {}          # landmark id -> last observation time
        self.landmark_obs = {}        # landmark id -> number of factors
        self.kf_t = {}                # keyframe index -> time
        self.kf_state = {}            # keyframe index -> (pose, vel, bias) latest estimate
        self.imu_factor = {}          # keyframe index k -> CombinedImuFactor (k-1 -> k)
        self.lm_factors = {}          # landmark id -> [projection factors]
        self.lm_point = {}            # landmark id -> latest point estimate
        self.n_rebuilds = 0
        self.n_resets = 0
        self.n_dropped = 0
        self.n_retired = 0

    # ------------------------------------------------------------------ setup
    def _new_smoother(self):
        params = self.gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        self.smoother = self.gu.IncrementalFixedLagSmoother(self.lag_s, params)
        self.graph = self.gtsam.NonlinearFactorGraph()
        self.values = self.gtsam.Values()
        self.stamps = {}                     # key -> timestamp, pending for the next update

    def _stamp(self, key, t):
        self.stamps[key] = float(t)

    def _stamps_map(self):
        m = self.gu.FixedLagSmootherKeyTimestampMap()
        for key, t in self.stamps.items():
            m.insert((key, t))
        return m

    def initialize(self, k: int, t: float, T_W_I: np.ndarray, vel, bias, sigmas=(0.05, 0.01, 0.1, 0.1, 0.01)):
        """Prior on the first keyframe: rotation tight (roll/pitch from gravity), yaw/position free-ish."""
        g = self.gtsam
        pose = g.Pose3(T_W_I)
        self.values.insert(self.X(k), pose)
        self.values.insert(self.V(k), np.asarray(vel, float))
        self.values.insert(self.B(k), bias)
        pose_noise = g.noiseModel.Diagonal.Sigmas(np.array([sigmas[0], sigmas[0], 0.5, sigmas[1], sigmas[1], sigmas[1]]))
        self.graph.add(g.PriorFactorPose3(self.X(k), pose, pose_noise))
        self.graph.add(g.PriorFactorVector(self.V(k), np.asarray(vel, float), g.noiseModel.Isotropic.Sigma(3, sigmas[2])))
        self.graph.add(g.PriorFactorConstantBias(self.B(k), bias, g.noiseModel.Diagonal.Sigmas(np.array([sigmas[3]] * 3 + [sigmas[4]] * 3))))
        for key in (self.X(k), self.V(k), self.B(k)):
            self._stamp(key, t)
        self.kf_t[k] = t

    # -------------------------------------------------------------- keyframes
    def add_keyframe(self, k: int, t: float, pim, predicted, bias_prev_key: int):
        """New keyframe k with IMU factor from keyframe k-1 (pim integrated between them)."""
        g = self.gtsam
        self.values.insert(self.X(k), predicted.pose())
        self.values.insert(self.V(k), predicted.velocity())
        self.values.insert(self.B(k), self.estimate.atConstantBias(self.B(bias_prev_key)) if self.estimate.exists(self.B(bias_prev_key)) else pim.biasHat())
        f = g.CombinedImuFactor(self.X(k - 1), self.V(k - 1), self.X(k), self.V(k), self.B(k - 1), self.B(k), pim)
        self.graph.add(f)
        self.imu_factor[k] = f
        for key in (self.X(k), self.V(k), self.B(k)):
            self._stamp(key, t)
        self.kf_t[k] = t

    def has_landmark(self, j: int) -> bool:
        return j in self.landmark_t

    def landmark_alive(self, j: int, t_now: float) -> bool:
        return j in self.landmark_t and (t_now - self.landmark_t[j]) < self.lag_s * 0.8

    def add_landmark(self, j: int, point_w, t: float):
        self.values.insert(self.L(j), np.asarray(point_w, float))
        self.landmark_t[j] = t
        self.landmark_obs[j] = 0
        self.lm_point[j] = np.asarray(point_w, float)
        self.lm_factors[j] = []
        self._stamp(self.L(j), t)

    def can_observe(self, bearing) -> bool:
        """Projection factors need a perspective division: finite bearing inside the half-angle."""
        return bool(np.all(np.isfinite(bearing)) and bearing[2] > self.cz)

    def add_observation(self, k: int, cam: int, j: int, bearing, t: float) -> bool:
        if not self.can_observe(bearing):
            return False
        m = np.array([bearing[0] / bearing[2], bearing[1] / bearing[2]])
        f = self.gtsam.GenericProjectionFactorCal3_S2(m, self.px_noise[cam], self.X(k), self.L(j), self.K, self.body_P_sensor[cam])
        self.graph.add(f)
        self.lm_factors.setdefault(j, []).append((k, f))
        self.landmark_t[j] = t
        self.landmark_obs[j] = self.landmark_obs.get(j, 0) + 1
        self._stamp(self.L(j), t)
        return True

    def drop_pending_landmark(self, j: int):
        """A landmark added this round but with too few observations: never send it —
        remove its value AND every pending factor that references it."""
        key = self.L(j)
        if self.values.exists(key):
            self.values.erase(key)
        if any(key in self.graph.at(i).keys() for i in range(self.graph.size())):
            g = self.gtsam.NonlinearFactorGraph()
            for i in range(self.graph.size()):
                f = self.graph.at(i)
                if key not in f.keys():
                    g.add(f)
            self.graph = g
        self.stamps.pop(key, None)
        self.landmark_t.pop(j, None); self.landmark_obs.pop(j, None)
        self.lm_factors.pop(j, None); self.lm_point.pop(j, None)

    def retire_landmark(self, j: int):
        """Stop observing landmark j (it stays in the smoother until the lag marginalises it)."""
        self.landmark_t.pop(j, None); self.landmark_obs.pop(j, None)
        self.n_retired += 1

    def _remember_estimate(self):
        est = self.estimate
        for k in list(self.kf_t):
            if est.exists(self.X(k)):
                self.kf_state[k] = (est.atPose3(self.X(k)), np.asarray(est.atVector(self.V(k))), est.atConstantBias(self.B(k)))
        for j in list(self.lm_point):
            if est.exists(self.L(j)):
                self.lm_point[j] = np.asarray(est.atPoint3(self.L(j)))

    def _forget_old(self, t: float):
        cutoff = t - self.lag_s
        for k in [k for k, tk in self.kf_t.items() if tk < cutoff - self.lag_s]:   # keep one lag of history for rebuilds
            self.kf_t.pop(k, None); self.kf_state.pop(k, None); self.imu_factor.pop(k, None)
        for j in [j for j, tj in self.landmark_t.items() if tj < cutoff]:
            self.landmark_t.pop(j, None); self.landmark_obs.pop(j, None)
            self.lm_factors.pop(j, None); self.lm_point.pop(j, None)

    def _rebuild(self, k: int, t: float, exclude: set):
        """Recreate the smoother from memory: every keyframe inside the lag with its IMU
        factor, a prior on the oldest kept keyframe (the marginal we no longer have),
        every alive landmark except `exclude` with all its factors whose poses are
        kept. The world frame and the map survive; only the offender is lost."""
        g = self.gtsam
        cutoff = t - self.lag_s
        kept = sorted(kk for kk, tk in self.kf_t.items() if tk >= cutoff and kk in self.kf_state)
        if not kept:
            raise RuntimeError("no keyframes to rebuild from")
        self._new_smoother()
        pending_graph, pending_values = self.graph, self.values          # (empty after _new_smoother)
        k0 = kept[0]
        pose, vel, bias = self.kf_state[k0]
        self.graph.add(g.PriorFactorPose3(self.X(k0), pose, g.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05]))))
        self.graph.add(g.PriorFactorVector(self.V(k0), vel, g.noiseModel.Isotropic.Sigma(3, 0.1)))
        self.graph.add(g.PriorFactorConstantBias(self.B(k0), bias, g.noiseModel.Diagonal.Sigmas(np.array([0.05] * 3 + [0.005] * 3))))
        for kk in kept:
            pose, vel, bias = self.kf_state[kk]
            self.values.insert(self.X(kk), pose); self.values.insert(self.V(kk), vel); self.values.insert(self.B(kk), bias)
            for key in (self.X(kk), self.V(kk), self.B(kk)):
                self._stamp(key, self.kf_t[kk])
            if kk != k0 and kk in self.imu_factor and (kk - 1) in self.kf_state and (kk - 1) >= k0:
                self.graph.add(self.imu_factor[kk])
        kept_set = set(kept)
        n_lm = 0
        for j, facs in list(self.lm_factors.items()):
            if j in exclude or j not in self.lm_point or j not in self.landmark_t:
                continue
            usable = [f for kk, f in facs if kk in kept_set]
            if len(usable) < 2:
                continue
            self.values.insert(self.L(j), self.lm_point[j])
            for f in usable:
                self.graph.add(f)
            self._stamp(self.L(j), self.landmark_t[j])
            n_lm += 1
        for j in exclude:
            self.landmark_t.pop(j, None); self.landmark_obs.pop(j, None); self.lm_factors.pop(j, None); self.lm_point.pop(j, None)
        self.smoother.update(self.graph, self.values, self._stamps_map())
        self.estimate = self.smoother.calculateEstimate()
        self.graph = g.NonlinearFactorGraph(); self.values = g.Values(); self.stamps = {}
        self.n_rebuilds += 1
        if self.verbose:
            print(f"[backend] kf {k}: rebuilt the smoother from memory: {len(kept)} keyframes, {n_lm} landmarks (dropped {sorted(exclude)})")

    # ---------------------------------------------------------------- solve
    def optimize(self, k: int, t: float):
        """Push the pending graph/values, return (T_W_I, vel, bias) of keyframe k.

        iSAM2 is not exception-safe: after a failed update the object is unusable,
        so any failure is answered by rebuilding the smoother from our own memory of
        the window (keyframes, IMU factors, landmark factors), minus the landmark
        that caused it. The map and the world frame survive."""
        g = self.gtsam
        for j in [j for j, n in self.landmark_obs.items() if n < 2 and self.values.exists(self.L(j))]:
            self.drop_pending_landmark(j)
        excluded: set = set()
        for attempt in range(4):
            try:
                self.smoother.update(self.graph, self.values, self._stamps_map())
                self.estimate = self.smoother.calculateEstimate()
                break
            except Exception as e:
                m = re.search(r"Symbol: l(\d+)", str(e))
                if m:
                    excluded.add(int(m.group(1)))
                if self.verbose:
                    print(f"[backend] kf {k}: smoother failure ({type(e).__name__}: {str(e)[:60].strip()}); rebuilding without {sorted(excluded)}")
                try:
                    # keep this keyframe's pending contribution: fold pending factors/values into memory first
                    self._absorb_pending_into_memory(k, t, excluded)
                    self._rebuild(k, t, excluded)
                    break
                except Exception as e2:
                    if self.verbose:
                        print(f"[backend] kf {k}: rebuild failed ({type(e2).__name__}: {str(e2)[:60].strip()})")
                    if attempt == 3:
                        self.n_resets += 1
                        self._soft_reset(k, t)
        self.graph = g.NonlinearFactorGraph(); self.values = g.Values(); self.stamps = {}
        self._remember_estimate()
        self._forget_old(t)
        return self.state(k)

    def _absorb_pending_into_memory(self, k: int, t: float, excluded: set):
        """The pending values of keyframe k (predicted state, new landmark points) become
        part of memory so the rebuild includes this keyframe; pending factors are already
        registered in imu_factor / lm_factors when they were created."""
        v = self.values
        if v.exists(self.X(k)):
            self.kf_state[k] = (v.atPose3(self.X(k)), np.asarray(v.atVector(self.V(k))), v.atConstantBias(self.B(k)))
        for j in list(self.lm_point):
            if v.exists(self.L(j)):
                self.lm_point[j] = np.asarray(v.atPoint3(self.L(j)))

    def _soft_reset(self, k: int, t: float):
        """Keep the world frame: restart the smoother with a prior on the newest
        state we have (the predicted one if the update never went through)."""
        g = self.gtsam
        if self.estimate.exists(self.X(k)):
            pose, vel, bias = self.estimate.atPose3(self.X(k)), self.estimate.atVector(self.V(k)), self.estimate.atConstantBias(self.B(k))
        elif self.values.exists(self.X(k)):
            pose, vel, bias = self.values.atPose3(self.X(k)), self.values.atVector(self.V(k)), self.values.atConstantBias(self.B(k))
        else:
            kk = max(kk for kk in self.kf_t if self.estimate.exists(self.X(kk)))
            pose, vel, bias = self.estimate.atPose3(self.X(kk)), self.estimate.atVector(self.V(kk)), self.estimate.atConstantBias(self.B(kk))
        self._new_smoother()
        self.landmark_t.clear(); self.landmark_obs.clear()
        self.values.insert(self.X(k), pose); self.values.insert(self.V(k), vel); self.values.insert(self.B(k), bias)
        self.graph.add(g.PriorFactorPose3(self.X(k), pose, g.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))))
        self.graph.add(g.PriorFactorVector(self.V(k), vel, g.noiseModel.Isotropic.Sigma(3, 0.2)))
        self.graph.add(g.PriorFactorConstantBias(self.B(k), bias, g.noiseModel.Diagonal.Sigmas(np.array([0.1] * 3 + [0.01] * 3))))
        for key in (self.X(k), self.V(k), self.B(k)):
            self._stamp(key, t)
        self.smoother.update(self.graph, self.values, self._stamps_map())
        self.estimate = self.smoother.calculateEstimate()

    def state(self, k: int):
        pose = self.estimate.atPose3(self.X(k))
        return pose.matrix(), np.asarray(self.estimate.atVector(self.V(k))), self.estimate.atConstantBias(self.B(k))

    def navstate(self, k: int):
        return self.gtsam.NavState(self.estimate.atPose3(self.X(k)), self.estimate.atVector(self.V(k)))

    def landmark(self, j: int):
        return np.asarray(self.estimate.atPoint3(self.L(j))) if self.estimate.exists(self.L(j)) else None
