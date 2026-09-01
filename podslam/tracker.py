"""The pipeline: conditioning -> front-end -> keyframes/landmarks -> backend -> pose.

Robustness policy (from docs/CUVSLAM_DIAGNOSIS_2026-08-30.md):
  - the world frame is fixed at initialisation and never re-created; a visual
    loss is bridged by IMU propagation, not by a new frame;
  - the front-end gets the gyro rotation since the last frame as its motion
    prediction (no internal constant-velocity model);
  - images are photometrically normalised before tracking (exposure steps);
  - per-frame masks (image circle, saturation, learned) apply to detection,
    tracking and stereo alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backend import Backend
from .backend_smart import SmartBackend
from .conditioning import build_conditioner, build_mask_provider
from .frontend import build_frontend
from .frontend.common import stereo_verify
from .imu import ImuBuffer, Preintegrator, StaticInitializer, delta_rotation


@dataclass
class TrackerConfig:
    frontend: str = "klt"
    frontend_cfg: dict = field(default_factory=dict)
    preprocess: str = "norm"
    masks: str = "none"                   # per-frame mask providers ('sat', 'learned:<pt>')
    circle_mask: bool = True              # static image-circle mask for f-theta lenses
    kf_every: int = 3                     # keyframe at least every N frames
    # motion-adaptive keyframes: also key when (>= kf_min_every frames since the last
    # keyframe and) the IMU rotation since the keyframe >= kf_rot_deg or the median
    # feature parallax >= kf_parallax_px.  Fast rotation is where tracks die and
    # where 3-frame keyframes (26 deg apart at 3 rad/s) starve the landmarks.
    # Off by default: on room1 the 5 deg / 15 px setting doubled the keyframes and
    # made day worse (8.8 -> 12.5 cm, 32-keyframe cap shortens the window); night 16.2 -> 15.5.
    kf_min_every: int = 1
    kf_rot_deg: float = 0.0
    kf_parallax_px: float = 0.0
    max_window_kf: int = 32               # keyframe-count cap of the sliding window (cost bound)
    max_obs_angle_deg: float = 80.0       # smart backend: observations further off-axis are not used
    depth_feedback: bool = False          # estimator landmark depths as stereo guesses: no gain on Hilti, hurts TUM-VI
    px_sigma_adapt: bool = False          # residual-driven sigma converges to 0.5-0.9 px (residuals at the
                                          # converged solution do not see calibration/distortion errors): off
    dyn_weight: bool = False              # smart backend: per-landmark temporal-consistency down-weighting
                                          # (wind sway); replaces the global px_sigma 2.5 stopgap when on
    px_adapt_up: bool = False             # smart backend: one-sided global sigma adaptation (>= rig nominal)
    anchor_delay_kf: int = 0              # smart backend: soft gauge at init, hard re-anchor at this keyframe (0 = legacy)
    kf_dense_init_s: float = 0.0          # keyframe every frame for this long after init (landmark maturity)
    kf_min_track_ratio: float = 0.6       # ...or earlier when tracks fall below this share
    lag_s: float = 4.0
    px_sigma: float = 1.5
    backend: str = "smart"                # smart (bearing-only smart rig factors, window LM) | explicit (fixed-lag iSAM2)
    mono_landmarks: bool = True           # smart backend: every track becomes a landmark (parallax over time)
    min_landmark_obs: int = 2
    max_landmarks_per_kf: int = 120
    marg_mode: str = "all"          # smart backend: 'ended' | 'all' | 'pin' (see backend_smart)
    smart_epi: bool = False           # smart backend: nonlinear landmark refinement (gtsam throws inside LM: keep off)
    imu_noise_scale: tuple = (5.7, 1.8, 1.2, 4.5)   # estimator-side IMU noise inflation (see imu.Preintegrator)
    init_acc_bias_sigma: float = 0.1  # prior sigma of the accelerometer bias at init [m/s^2] (BMI085-class IMUs: ~0.3)
    init_tilt_sigma: float = 0.01     # prior sigma of the first pose's roll/pitch [rad]; 0.05 helped Hilti's Sim3 but
                                      # cost 8.8 -> 12.2 cm on TUM-VI day (bisected): keep tight.  Gravity from the static
                                      # accelerometer mean carries the accel bias (0.17 m/s^2 = 1 deg on the Hilti
                                      # BMI085); roll/pitch are observable, only yaw/position need the hard anchor
    # initialisation: 'static' = wait-for-still (legacy, forced fallback after 3 s);
    # 'auto' = still-detection when genuinely still, else moving-platform init from
    # cross-camera metric structure + IMU (init_dynamic); 'dynamic' = always the latter.
    # Both non-static modes fall back to the forced-static assumption after
    # init_max_s of starvation (featureless / matchless scenes) — never worse than legacy.
    init_mode: str = "static"             # static | auto | dynamic
    init_window_s: float = 1.5            # dynamic init: collection window
    init_max_s: float = 6.0               # dynamic init: give up and force static after this
    init_stride: int = 2                  # dynamic init: cross-cam match every Nth frame
    dyn_vel_sigma: float = 0.3            # dynamic init: velocity prior sigma [m/s]
    dyn_tilt_sigma: float = 0.03          # dynamic init: roll/pitch prior sigma [rad]
    dyn_gyro_bias_sigma: float = 0.02     # dynamic init: gyro bias prior sigma [rad/s] (bias not estimated)
    verbose: bool = False


@dataclass
class Estimate:
    t_ns: int
    T_W_I: np.ndarray | None
    ok: bool
    keyframe: bool
    n_obs: list
    status: str = ""
    n_landmarks: int = 0


class Tracker:
    def __init__(self, rig, config: TrackerConfig | None = None, frontend=None):
        self.rig = rig
        self.cfg = config or TrackerConfig()
        self.frontend = frontend or build_frontend(self.cfg.frontend, rig, self.cfg.frontend_cfg)
        self.condition = build_conditioner(self.cfg.preprocess)
        self.mask_provider = build_mask_provider(self.cfg.masks)
        self.static_masks = [c.circle_mask() if self.cfg.circle_mask else None for c in rig.cameras]
        self.imu = ImuBuffer()
        # auto/dynamic: still-detection may win (auto) or serve as the starvation
        # fallback (force()), but must never fire the 3 s forced assumption itself
        self.init = StaticInitializer() if self.cfg.init_mode == "static" else StaticInitializer(max_wait_s=float("inf"))
        self.dyn_init = None
        self.t_first_frame = None
        self.init_frame_i = -1
        self.backend = None
        self.pim = Preintegrator(rig.imu, noise_scale=self.cfg.imu_noise_scale)
        self.k = -1                        # current keyframe index
        self.kf_navstate = None
        self.bias = None
        self.gyro_bias = np.zeros(3)
        self.t_prev_frame = None
        self.frames_since_kf = 0
        self.n_tracks_at_kf = 0
        self.px_at_kf = {}
        self.track_to_lm = {}              # front-end track id -> landmark id
        self.next_lm = 0
        self.t_init = None
        self.initialized = False

    # ------------------------------------------------------------------ IMU
    def register_imu(self, t_ns: int, gyro, accel) -> None:
        t = t_ns * 1e-9
        self.imu.append(t, gyro, accel)
        if not self.initialized:
            self.init.feed(t, gyro, accel)

    # ---------------------------------------------------------------- frames
    def track(self, t_ns: int, images: list, masks: list | None = None) -> Estimate:
        t = t_ns * 1e-9
        imgs = [self.condition(im) for im in images]
        ms = []
        for i, im in enumerate(imgs):
            base = self.static_masks[i]
            m = self.mask_provider(im, base)
            if masks is not None and masks[i] is not None:
                m = masks[i] if m is None else np.minimum(m, masks[i])
            ms.append(m)
        if not self.initialized:
            if self.cfg.init_mode == "static":
                if self.init.result is None:
                    return Estimate(t_ns, None, False, False, [0] * len(imgs), "waiting for static init")
                self._initialize(t, imgs, ms)
                return Estimate(t_ns, self.kf_navstate.pose().matrix(), True, True, self._last_nobs, "initialized", self._n_lm)
            return self._pre_init_dynamic(t_ns, t, imgs, ms)

        dR = delta_rotation(self.imu, self.t_prev_frame, t, self.gyro_bias)
        feats = self.frontend.process(t_ns, imgs, ms, dR)
        self.t_prev_frame = t
        self.frames_since_kf += 1
        self.pim.integrate_until(self.imu, t)
        predicted = self.pim.predict(self.kf_navstate)
        n0 = sum(len(c) for c in feats.cams) if getattr(self.frontend, "per_cam", False) else len(feats.cams[0])
        is_kf = (self.frames_since_kf >= self.cfg.kf_every) or (n0 < self.cfg.kf_min_track_ratio * max(self.n_tracks_at_kf, 1))
        if self.cfg.kf_dense_init_s > 0 and self.t_init is not None and (t - self.t_init) < self.cfg.kf_dense_init_s:
            is_kf = True
        if not is_kf and self.frames_since_kf >= self.cfg.kf_min_every:
            rot = np.degrees(np.linalg.norm(self.pim.delta_rotvec()))
            par = self._parallax_since_kf(feats)
            is_kf = (self.cfg.kf_rot_deg > 0 and rot >= self.cfg.kf_rot_deg) or (self.cfg.kf_parallax_px > 0 and par >= self.cfg.kf_parallax_px)
        self._last_nobs = [len(c) for c in feats.cams]
        if not is_kf:
            return Estimate(t_ns, predicted.pose().matrix(), True, False, self._last_nobs, "propagated", self._n_lm)
        self._keyframe(t, feats, predicted)
        return Estimate(t_ns, self.kf_navstate.pose().matrix(), True, True, self._last_nobs, "keyframe", self._n_lm)

    # ------------------------------------------------------------- internals
    @property
    def _n_lm(self) -> int:
        if self.backend is None:
            return 0
        return getattr(self.backend, "n_valid_lm", len(self.backend.landmark_t))

    def _initialize(self, t, imgs, ms, feats=None):
        r = self.init.result
        self.gyro_bias = r["gyro_bias"]
        T = np.eye(4); T[:3, :3] = r["R_W_I"]
        self._finish_init(t, imgs, ms, feats, T, np.zeros(3),
                          sigmas=(1e-3, self.cfg.init_tilt_sigma, 0.1, self.cfg.init_acc_bias_sigma, 0.01))

    def _initialize_dynamic(self, t, feats, r):
        self.gyro_bias = r["gyro_bias"]
        T = np.eye(4); T[:3, :3] = r["R_W_I"]; T[:3, 3] = r["p_W"]
        self._finish_init(t, None, None, feats, T, r["v_W"],
                          sigmas=(1e-3, self.cfg.dyn_tilt_sigma, self.cfg.dyn_vel_sigma,
                                  self.cfg.init_acc_bias_sigma, self.cfg.dyn_gyro_bias_sigma))

    def _finish_init(self, t, imgs, ms, feats, T, vel, sigmas):
        import gtsam
        self.t_init = t
        self.bias = gtsam.imuBias.ConstantBias(np.zeros(3), self.gyro_bias)
        if self.cfg.backend == "smart":
            self.backend = SmartBackend(self.rig, lag_s=self.cfg.lag_s, px_sigma=self.cfg.px_sigma, marg_mode=self.cfg.marg_mode, epi=self.cfg.smart_epi,
                                        max_window_kf=self.cfg.max_window_kf, max_obs_angle_deg=self.cfg.max_obs_angle_deg,
                                        px_sigma_adapt=self.cfg.px_sigma_adapt, px_adapt_up=self.cfg.px_adapt_up, dyn_weight=self.cfg.dyn_weight, anchor_delay_kf=self.cfg.anchor_delay_kf, verbose=self.cfg.verbose)
        else:
            self.backend = Backend(self.rig, lag_s=self.cfg.lag_s, px_sigma=self.cfg.px_sigma, verbose=self.cfg.verbose)
        self.k = 0
        self.backend.initialize(0, t, T, np.asarray(vel, float), self.bias, sigmas=sigmas)
        self.kf_navstate = gtsam.NavState(gtsam.Pose3(T), np.asarray(vel, float))
        self.pim.reset(self.bias, t)
        if feats is None:
            feats = self.frontend.process(int(t * 1e9), imgs, ms, None)
        self._last_nobs = [len(c) for c in feats.cams]
        self._add_landmarks_and_factors(0, t, feats, T)
        self.backend.optimize(0, t)
        T_est, v, b = self.backend.state(0)
        self.kf_navstate = gtsam.NavState(gtsam.Pose3(T_est), v)
        self.t_prev_frame = t
        self.frames_since_kf = 0
        if getattr(self.frontend, "per_cam", False):
            self.n_tracks_at_kf = sum(len(c) for c in feats.cams)
            self.px_at_kf = {int(i): p for c in feats.cams for i, p in zip(c.ids, c.px)}
        else:
            self.n_tracks_at_kf = len(feats.cams[0])
            self.px_at_kf = {int(i): p for i, p in zip(feats.cams[0].ids, feats.cams[0].px)}
        self.initialized = True

    def _pre_init_dynamic(self, t_ns, t, imgs, ms):
        """auto/dynamic modes: track features while collecting cross-camera metric
        structure; initialise from whichever is ready first — still-detection
        (auto) or the moving-platform solve. Falls back to the forced static
        assumption after init_max_s (matchless scenes: never worse than legacy)."""
        from .init_dynamic import CrossCamMatcher, DynamicInitializer
        dR = None if self.t_prev_frame is None else delta_rotation(self.imu, self.t_prev_frame, t, np.zeros(3))
        feats = self.frontend.process(t_ns, imgs, ms, dR)
        self.t_prev_frame = t
        self._last_nobs = [len(c) for c in feats.cams]
        if self.t_first_frame is None:
            self.t_first_frame = t
        if self.dyn_init is None:
            self.dyn_init = DynamicInitializer(self.imu, window_s=self.cfg.init_window_s)
            self._matcher = CrossCamMatcher(self.rig)
        if self.cfg.init_mode == "auto" and self.init.result is not None:
            self._initialize(t, imgs, ms, feats=feats)
            return Estimate(t_ns, self.kf_navstate.pose().matrix(), True, True, self._last_nobs, "initialized", self._n_lm)
        self.init_frame_i += 1
        if self.init_frame_i % max(self.cfg.init_stride, 1) == 0:
            pts, stats = self._matcher.match(imgs, ms, feats)
            self.dyn_init.add_frame_points(t, pts)
            self._init_match_stats = stats
        if self.dyn_init.ready:
            r = self.dyn_init.solve()
            if r is not None:
                self._initialize_dynamic(t, feats, r)
                if self.cfg.verbose:
                    print(f"dynamic init: {r['n_frames']} frames / {r['span_s']:.2f} s, ~{r['n_pts']} pts, "
                          f"|v| {np.linalg.norm(r['v_W']):.2f} m/s, |g| {r['g_norm']:.2f}")
                return Estimate(t_ns, self.kf_navstate.pose().matrix(), True, True, self._last_nobs, "initialized dynamic", self._n_lm)
        if t - self.t_first_frame > self.cfg.init_max_s:
            self.init.force(t)
            self._initialize(t, imgs, ms, feats=feats)
            return Estimate(t_ns, self.kf_navstate.pose().matrix(), True, True, self._last_nobs, "initialized forced-static", self._n_lm)
        n = len(self.dyn_init.frames)
        return Estimate(t_ns, None, False, False, self._last_nobs, f"dynamic init: {n} frames")

    def _feed_depths(self, feats, T_W_I):
        """Give the front-end the estimator's landmark depths (tracking camera frame) as
        the next stereo initial guesses."""
        be = self.backend
        cam0 = self.rig.cameras[0]
        T_c_w = np.linalg.inv(T_W_I @ cam0.T_imu_cam)
        depths = {}
        for i in feats.cams[0].ids:
            j = self.track_to_lm.get(int(i))
            if j is None:
                continue
            p_w = be.landmark(j)
            if p_w is None:
                continue
            p_w = np.asarray(p_w, dtype=float).reshape(-1)
            if p_w.shape != (3,) or not np.all(np.isfinite(p_w)):
                continue
            z = float(T_c_w[2, :3] @ p_w + T_c_w[2, 3])
            if 0.1 < z < 100.0:
                depths[int(i)] = z
        if depths:
            self.frontend.set_depths(depths)

    def _parallax_since_kf(self, feats) -> float:
        cam0 = feats.cams[0]
        if not self.px_at_kf or len(cam0) == 0:
            return 0.0
        d = [np.linalg.norm(cam0.px[n] - self.px_at_kf[int(i)]) for n, i in enumerate(cam0.ids) if int(i) in self.px_at_kf]
        return float(np.median(d)) if d else 0.0

    def _keyframe(self, t, feats, predicted):
        import gtsam
        self.k += 1
        k = self.k
        self.backend.add_keyframe(k, t, self.pim.pim, predicted, k - 1)
        self._add_landmarks_and_factors(k, t, feats, predicted.pose().matrix())
        T_est, v, b = self.backend.optimize(k, t)
        if self.cfg.depth_feedback:
            self._feed_depths(feats, T_est)
        self.bias = b
        self.gyro_bias = np.asarray(b.gyroscope())
        self.kf_navstate = gtsam.NavState(gtsam.Pose3(T_est), v)
        self.pim.reset(b, t)
        self.frames_since_kf = 0
        if getattr(self.frontend, "per_cam", False):
            self.n_tracks_at_kf = sum(len(c) for c in feats.cams)
            self.px_at_kf = {int(i): p for c in feats.cams for i, p in zip(c.ids, c.px)}
        else:
            self.n_tracks_at_kf = len(feats.cams[0])
            self.px_at_kf = {int(i): p for i, p in zip(feats.cams[0].ids, feats.cams[0].px)}

    def _add_landmarks_and_factors(self, k, t, feats, T_W_I):
        """Existing landmarks: add projection factors. New tracks: create the
        landmark and add factors from every camera that sees the track id.
        Iterates every camera's native tracks; with the stereo-centric front-end
        (shared ids) the seen-set makes this identical to the old cam0-only loop."""
        be = self.backend
        by_cam = [{int(i): n for n, i in enumerate(c.ids)} for c in feats.cams]
        n_new = 0
        seen: set[int] = set()
        for c0 in range(len(feats.cams)):
            cam0 = feats.cams[c0]
            others = [c for c in range(len(feats.cams)) if c != c0]
            for n, tid in enumerate(cam0.ids):
                tid = int(tid)
                if tid in seen:
                    continue
                seen.add(tid)
                j = self.track_to_lm.get(tid)
                if j is not None and not be.landmark_alive(j, t):
                    self.track_to_lm.pop(tid, None); j = None
                if j is None:
                    pair = next(((c, by_cam[c][tid]) for c in others if tid in by_cam[c]), None)
                    if self.cfg.backend == "smart" and self.cfg.mono_landmarks:
                        # smart factors triangulate themselves: any track is a landmark
                        if n_new >= self.cfg.max_landmarks_per_kf * 3 or not be.can_observe(cam0.bearings[n]):
                            continue
                        j = self.next_lm; self.next_lm += 1
                        be.add_landmark(j, None, t); self.track_to_lm[tid] = j; n_new += 1
                        be.add_observation(k, c0, j, cam0.bearings[n], t)
                        for c in others:
                            if tid in by_cam[c]:
                                be.add_observation(k, c, j, feats.cams[c].bearings[by_cam[c][tid]], t)
                        continue
                    # explicit backend: new landmark only if stereo-observed now (metric depth)
                    if pair is None or n_new >= self.cfg.max_landmarks_per_kf:
                        continue
                    c, m = pair
                    b0, bc = cam0.bearings[n], feats.cams[c].bearings[m]
                    if not (be.can_observe(b0) and be.can_observe(bc)):   # both rays must become factors
                        continue
                    p_imu, _ = stereo_verify(self.rig, c0, c, b0, bc, cam0.px[n], feats.cams[c].px[m], max_px=3.0)
                    if p_imu is None:
                        continue
                    p_w = T_W_I[:3, :3] @ p_imu + T_W_I[:3, 3]
                    j = self.next_lm; self.next_lm += 1
                    be.add_landmark(j, p_w, t)
                    self.track_to_lm[tid] = j
                    n_new += 1
                    be.add_observation(k, c0, j, b0, t)
                    be.add_observation(k, c, j, bc, t)
                    continue
                p_w = be.landmark(j)
                if p_w is not None and not self._in_front(p_w, T_W_I):
                    # the optimiser pushed it behind us (usually a wrong stereo match): retire it
                    self.track_to_lm.pop(tid, None); be.retire_landmark(j)
                    continue
                be.add_observation(k, c0, j, cam0.bearings[n], t)
                for c in others:
                    if tid in by_cam[c]:
                        be.add_observation(k, c, j, feats.cams[c].bearings[by_cam[c][tid]], t)

    def _in_front(self, p_w, T_W_I, min_z=0.15) -> bool:
        p_w = np.asarray(p_w, dtype=float).reshape(-1)
        if p_w.shape != (3,) or not np.all(np.isfinite(p_w)):
            return True                       # unknown point: nothing to retire on
        T_I_W = np.linalg.inv(T_W_I)
        p_i = T_I_W[:3, :3] @ p_w + T_I_W[:3, 3]
        for cam in self.rig.cameras:
            T_c_i = cam.T_cam_imu
            if (T_c_i[:3, :3] @ p_i + T_c_i[:3, 3])[2] > min_z:
                return True
        return False
