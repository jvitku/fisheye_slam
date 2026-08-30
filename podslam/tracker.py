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
    init_tilt_sigma: float = 0.05     # prior sigma of the first pose's roll/pitch [rad]: gravity from the static
                                      # accelerometer mean carries the accel bias (0.17 m/s^2 = 1 deg on the Hilti
                                      # BMI085); roll/pitch are observable, only yaw/position need the hard anchor
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
        self.init = StaticInitializer()
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
            if self.init.result is None:
                return Estimate(t_ns, None, False, False, [0] * len(imgs), "waiting for static init")
            self._initialize(t, imgs, ms)
            return Estimate(t_ns, self.kf_navstate.pose().matrix(), True, True, self._last_nobs, "initialized", self._n_lm)

        dR = delta_rotation(self.imu, self.t_prev_frame, t, self.gyro_bias)
        feats = self.frontend.process(t_ns, imgs, ms, dR)
        self.t_prev_frame = t
        self.frames_since_kf += 1
        self.pim.integrate_until(self.imu, t)
        predicted = self.pim.predict(self.kf_navstate)
        n0 = len(feats.cams[0])
        is_kf = (self.frames_since_kf >= self.cfg.kf_every) or (n0 < self.cfg.kf_min_track_ratio * max(self.n_tracks_at_kf, 1))
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

    def _initialize(self, t, imgs, ms):
        import gtsam
        r = self.init.result
        self.gyro_bias = r["gyro_bias"]
        self.bias = gtsam.imuBias.ConstantBias(np.zeros(3), self.gyro_bias)
        T = np.eye(4); T[:3, :3] = r["R_W_I"]
        if self.cfg.backend == "smart":
            self.backend = SmartBackend(self.rig, lag_s=self.cfg.lag_s, px_sigma=self.cfg.px_sigma, marg_mode=self.cfg.marg_mode, epi=self.cfg.smart_epi,
                                        max_window_kf=self.cfg.max_window_kf, verbose=self.cfg.verbose)
        else:
            self.backend = Backend(self.rig, lag_s=self.cfg.lag_s, px_sigma=self.cfg.px_sigma, verbose=self.cfg.verbose)
        self.k = 0
        self.backend.initialize(0, t, T, np.zeros(3), self.bias, sigmas=(1e-3, self.cfg.init_tilt_sigma, 0.1, self.cfg.init_acc_bias_sigma, 0.01))
        self.kf_navstate = gtsam.NavState(gtsam.Pose3(T), np.zeros(3))
        self.pim.reset(self.bias, t)
        feats = self.frontend.process(int(t * 1e9), imgs, ms, None)
        self._last_nobs = [len(c) for c in feats.cams]
        self._add_landmarks_and_factors(0, t, feats, T)
        self.backend.optimize(0, t)
        T_est, v, b = self.backend.state(0)
        self.kf_navstate = gtsam.NavState(gtsam.Pose3(T_est), v)
        self.t_prev_frame = t
        self.frames_since_kf = 0
        self.n_tracks_at_kf = len(feats.cams[0])
        self.px_at_kf = {int(i): p for i, p in zip(feats.cams[0].ids, feats.cams[0].px)}
        self.initialized = True

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
        self.bias = b
        self.gyro_bias = np.asarray(b.gyroscope())
        self.kf_navstate = gtsam.NavState(gtsam.Pose3(T_est), v)
        self.pim.reset(b, t)
        self.frames_since_kf = 0
        self.n_tracks_at_kf = len(feats.cams[0])
        self.px_at_kf = {int(i): p for i, p in zip(feats.cams[0].ids, feats.cams[0].px)}

    def _add_landmarks_and_factors(self, k, t, feats, T_W_I):
        """Existing landmarks: add projection factors. New tracks with a stereo
        match: triangulate (in the current pose), create the landmark, add
        factors from every camera that sees it."""
        be = self.backend
        cam0 = feats.cams[0]
        by_cam = [{int(i): n for n, i in enumerate(c.ids)} for c in feats.cams]
        n_new = 0
        for n, tid in enumerate(cam0.ids):
            tid = int(tid)
            j = self.track_to_lm.get(tid)
            if j is not None and not be.landmark_alive(j, t):
                self.track_to_lm.pop(tid, None); j = None
            if j is None:
                pair = next(((c, by_cam[c][tid]) for c in range(1, len(feats.cams)) if tid in by_cam[c]), None)
                if self.cfg.backend == "smart" and self.cfg.mono_landmarks:
                    # smart factors triangulate themselves: any track is a landmark
                    if n_new >= self.cfg.max_landmarks_per_kf * 3 or not be.can_observe(cam0.bearings[n]):
                        continue
                    j = self.next_lm; self.next_lm += 1
                    be.add_landmark(j, None, t); self.track_to_lm[tid] = j; n_new += 1
                    be.add_observation(k, 0, j, cam0.bearings[n], t)
                    for c in range(1, len(feats.cams)):
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
                p_imu, _ = stereo_verify(self.rig, 0, c, b0, bc, cam0.px[n], feats.cams[c].px[m], max_px=3.0)
                if p_imu is None:
                    continue
                p_w = T_W_I[:3, :3] @ p_imu + T_W_I[:3, 3]
                j = self.next_lm; self.next_lm += 1
                be.add_landmark(j, p_w, t)
                self.track_to_lm[tid] = j
                n_new += 1
                be.add_observation(k, 0, j, b0, t)
                be.add_observation(k, c, j, bc, t)
                continue
            p_w = be.landmark(j)
            if p_w is not None and not self._in_front(p_w, T_W_I):
                # the optimiser pushed it behind us (usually a wrong stereo match): retire it
                self.track_to_lm.pop(tid, None); be.retire_landmark(j)
                continue
            be.add_observation(k, 0, j, cam0.bearings[n], t)
            for c in range(1, len(feats.cams)):
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
