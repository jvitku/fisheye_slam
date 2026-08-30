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
        self.stamps = self.gu.FixedLagSmootherKeyTimestampMap()

    def _stamp(self, key, t):
        self.stamps.insert((key, float(t)))

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
        self.graph.add(g.CombinedImuFactor(self.X(k - 1), self.V(k - 1), self.X(k), self.V(k), self.B(k - 1), self.B(k), pim))
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
        self._stamp(self.L(j), t)

    def can_observe(self, bearing) -> bool:
        """Projection factors need a perspective division: finite bearing inside the half-angle."""
        return bool(np.all(np.isfinite(bearing)) and bearing[2] > self.cz)

    def add_observation(self, k: int, cam: int, j: int, bearing, t: float) -> bool:
        if not self.can_observe(bearing):
            return False
        m = np.array([bearing[0] / bearing[2], bearing[1] / bearing[2]])
        self.graph.add(self.gtsam.GenericProjectionFactorCal3_S2(m, self.px_noise[cam], self.X(k), self.L(j), self.K, self.body_P_sensor[cam]))
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
        self.landmark_t.pop(j, None); self.landmark_obs.pop(j, None)

    def retire_landmark(self, j: int):
        """Stop observing landmark j (it stays in the smoother until the lag marginalises it)."""
        self.landmark_t.pop(j, None); self.landmark_obs.pop(j, None)
        self.n_retired += 1

    # ---------------------------------------------------------------- solve
    def optimize(self, k: int, t: float):
        """Push the pending graph/values, return (T_W_I, vel, bias) of keyframe k."""
        g = self.gtsam
        # guard: a brand-new landmark must arrive with >= 2 projection factors
        for j in [j for j, n in self.landmark_obs.items() if n < 2 and self.values.exists(self.L(j))]:
            self.drop_pending_landmark(j)
        for attempt in range(4):
            try:
                self.smoother.update(self.graph, self.values, self.stamps)
                self.estimate = self.smoother.calculateEstimate()
                break
            except Exception as e:      # IndeterminantLinearSystem etc.
                m = re.search(r"Symbol: l(\d+)", str(e))
                if m is not None and attempt < 3:
                    # the offender is a landmark: if it is new, drop it; if it already lives
                    # in the smoother, stop feeding it (remove its pending factors) — retry
                    # either way before giving up on the graph
                    j = int(m.group(1))
                    if self.verbose:
                        print(f"[backend] kf {k}: landmark l{j} made the system singular ({type(e).__name__}); "
                              f"{'dropping' if self.values.exists(self.L(j)) else 'retiring'} it and retrying")
                    self.drop_pending_landmark(j); self.n_dropped += 1
                    continue
                self.n_resets += 1
                if self.verbose:
                    print(f"[backend] smoother failure at kf {k}: {type(e).__name__}: {str(e)[:120]} -> soft reset")
                self._soft_reset(k, t)
                break
        self.graph = g.NonlinearFactorGraph(); self.values = g.Values(); self.stamps = self.gu.FixedLagSmootherKeyTimestampMap()
        # forget marginalised landmarks
        cutoff = t - self.lag_s
        for j in [j for j, tj in self.landmark_t.items() if tj < cutoff]:
            self.landmark_t.pop(j, None); self.landmark_obs.pop(j, None)
        return self.state(k)

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
        self.smoother.update(self.graph, self.values, self.stamps)
        self.estimate = self.smoother.calculateEstimate()

    def state(self, k: int):
        pose = self.estimate.atPose3(self.X(k))
        return pose.matrix(), np.asarray(self.estimate.atVector(self.V(k))), self.estimate.atConstantBias(self.B(k))

    def navstate(self, k: int):
        return self.gtsam.NavState(self.estimate.atPose3(self.X(k)), self.estimate.atVector(self.V(k)))

    def landmark(self, j: int):
        return np.asarray(self.estimate.atPoint3(self.L(j))) if self.estimate.exists(self.L(j)) else None
