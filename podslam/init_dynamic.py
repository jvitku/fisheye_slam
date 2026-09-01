"""Moving-platform (dynamic) initialisation: gravity + velocity from vision + IMU.

The static initialiser assumes the first second is motionless — on a window that
starts mid-flight its forced fallback swallows thrust and tilt into the gravity
estimate and anchors the world wrong (3.49 m ATE on the PX4 mid-flight window).

This module initialises while already moving, exploiting what a multi-camera
rig gives for free: *metric* structure from the inter-camera baselines, no
mono-SfM scale ambiguity.

    per frame:  cross-camera LK matches -> stereo_verify -> metric points
                (body frame)                                {track id: p_imu}
    rotation :  gyro chain (bias ~ 0 over a 1.5 s window: < 0.5 deg error)
    position :  chain of robust common-point alignments between consecutive
                frames (rotation fixed by the gyro -> translation-only, median
                + MAD inlier re-mean)
    solve    :  linear LSQ for gravity g0 (frame-0 coords) and per-frame
                velocities from the preintegrated IMU position/velocity
                equations; then re-solve velocities with |g| pinned to 9.81.

World convention matches StaticInitializer: +Z up; R_W_I maps body "up" to +Z.
The solver is image-free (unit-testable): images enter only through
cross_cam_points(), which produces the per-frame metric points.
"""
from __future__ import annotations

import numpy as np

from .frontend.common import stereo_verify_batch
from .geometry import exp_so3, rotation_aligning
from .imu import G, ImuBuffer


def _gyro_delta_R(ts, ws, t0, t1, bias=None):
    """R_{I(t0) <- I(t1)} from raw gyro samples (hold-last to t1)."""
    R = np.eye(3)
    t_prev = t0
    b = np.zeros(3) if bias is None else bias
    for t, w in zip(ts, ws):
        dt = float(t - t_prev)
        if dt > 0:
            R = R @ exp_so3((w - b) * dt)
        t_prev = t
    if t1 > t_prev and len(ws):
        R = R @ exp_so3((ws[-1] - b) * (t1 - t_prev))
    return R


def _preintegrate(ts, ws, accs, t0, t1):
    """(dP, dV) in frame-of-t0 coordinates, gravity-free specific-force integral:
    dV = sum R_t0<-tau a dt,  dP = sum v dt + 0.5 R a dt^2 (bias 0)."""
    R = np.eye(3)
    dV = np.zeros(3)
    dP = np.zeros(3)
    t_prev = t0

    def step(a_w_body, w_body, dt):
        nonlocal R, dV, dP
        a0 = R @ a_w_body
        dP += dV * dt + 0.5 * a0 * dt * dt
        dV += a0 * dt
        R = R @ exp_so3(w_body * dt)

    for t, w, a in zip(ts, ws, accs):
        dt = float(t - t_prev)
        if dt > 0:
            step(a, w, dt)
        t_prev = t
    if t1 > t_prev and len(ws):
        step(accs[-1], ws[-1], float(t1 - t_prev))
    return dP, dV


def _robust_translation(d, min_inliers=6):
    """Median + 3xMAD inlier re-mean of per-point translation votes d (N,3).
    Returns (t, n_inliers) or (None, 0)."""
    if len(d) < min_inliers:
        return None, 0
    med = np.median(d, axis=0)
    r = np.linalg.norm(d - med, axis=1)
    mad = np.median(r)
    keep = r <= max(3.0 * mad, 0.05)
    if keep.sum() < min_inliers:
        return None, 0
    return d[keep].mean(axis=0), int(keep.sum())


class DynamicInitializer:
    """Collects per-frame metric body-frame points + IMU, solves gravity/velocity.

    imu: ImuBuffer shared with the tracker (raw samples, keep_s covers the window).
    """

    def __init__(self, imu: ImuBuffer, window_s: float = 1.5, min_frames: int = 6,
                 min_common: int = 8, g_mag: float = G):
        self.imu = imu
        self.window_s = window_s
        self.min_frames = min_frames
        self.min_common = min_common
        self.g_mag = g_mag
        self.frames: list[tuple[float, dict]] = []      # (t, {tid: p_body (3,)})

    def add_frame_points(self, t: float, pts: dict) -> None:
        if pts:
            self.frames.append((float(t), {int(i): np.asarray(p, float) for i, p in pts.items()}))

    @property
    def ready(self) -> bool:
        return (len(self.frames) >= self.min_frames
                and self.frames[-1][0] - self.frames[0][0] >= self.window_s - 1e-9)

    @property
    def span_s(self) -> float:
        return self.frames[-1][0] - self.frames[0][0] if self.frames else 0.0

    def solve(self) -> dict | None:
        if not self.ready:
            return None
        m = len(self.frames)
        t0 = self.frames[0][0]
        # rotation chain C_k: frame-k coords -> frame-0 coords (gyro, bias 0)
        C = [np.eye(3)]
        dPs, dVs, dts = [], [], []
        for k in range(1, m):
            ta, tb = self.frames[k - 1][0], self.frames[k][0]
            ts, ws, accs = self.imu.between(ta, tb)
            if len(ts) == 0:
                return None
            C.append(C[k - 1] @ _gyro_delta_R(ts, ws, ta, tb))
            dP, dV = _preintegrate(ts, ws, accs, ta, tb)
            dPs.append(dP); dVs.append(dV); dts.append(tb - ta)
        # translation chain o_k: body-k origin in frame-0 coords
        o = [np.zeros(3)]
        for k in range(1, m):
            pa, pb = self.frames[k - 1][1], self.frames[k][1]
            common = pa.keys() & pb.keys()
            if len(common) < self.min_common:
                return None
            d = np.array([C[k - 1] @ pa[i] - C[k] @ pb[i] for i in common])
            t_ab, n_in = _robust_translation(d, self.min_common)
            if t_ab is None:
                return None
            o.append(o[k - 1] + t_ab)
        # LSQ: unknowns x = [v_0..v_{m-1} (frame-0 coords), g0] (3m + 3)
        # o_{k+1} = o_k + v_k dt + 0.5 g0 dt^2 + C_k dP_k
        # v_{k+1} = v_k + g0 dt + C_k dV_k
        n_unk = 3 * m + 3
        A = np.zeros((6 * (m - 1), n_unk))
        b = np.zeros(6 * (m - 1))
        for k in range(m - 1):
            dt = dts[k]
            rp = slice(6 * k, 6 * k + 3)
            rv = slice(6 * k + 3, 6 * k + 6)
            A[rp, 3 * k:3 * k + 3] = np.eye(3) * dt
            A[rp, 3 * m:] = np.eye(3) * 0.5 * dt * dt
            b[rp] = o[k + 1] - o[k] - C[k] @ dPs[k]
            A[rv, 3 * k:3 * k + 3] = -np.eye(3)
            A[rv, 3 * (k + 1):3 * (k + 1) + 3] = np.eye(3)
            A[rv, 3 * m:] = -np.eye(3) * dt
            b[rv] = C[k] @ dVs[k]
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        g0 = x[3 * m:]
        g_norm = float(np.linalg.norm(g0))
        if not (0.5 * self.g_mag < g_norm < 1.5 * self.g_mag):
            return None
        # pin |g| = g_mag, re-solve the velocities (still linear)
        u = g0 / g_norm
        g_fix = self.g_mag * u
        Av = A[:, :3 * m]
        bv = b - A[:, 3 * m:] @ g_fix
        v, *_ = np.linalg.lstsq(Av, bv, rcond=None)
        v = v.reshape(m, 3)
        # world: +Z up. Body "up" in frame-0 coords is -u (gravity points down).
        R_W_B0 = rotation_aligning(-u, [0.0, 0.0, 1.0])
        t_last = self.frames[-1][0]
        return dict(
            t=t_last,
            R_W_I=R_W_B0 @ C[-1],
            v_W=R_W_B0 @ v[-1],
            p_W=R_W_B0 @ o[-1],
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3),
            g_norm=g_norm,
            n_frames=m,
            span_s=t_last - t0,
            n_pts=int(np.mean([len(f[1]) for f in self.frames])),
            forced=False,
            dynamic=True,
        )


# --------------------------------------------------------------- cross-cam LK

class CrossCamMatcher:
    """Init-time metric structure across a divergent multi-camera rig.

    Raw LK cannot bridge an 80-120 deg viewpoint change between distorted
    fisheyes (measured: 1.2 % match rate). So each pair (i -> j) gets a
    precomputed *rotation warp*: cam i's image resampled into cam j's geometry
    at infinite depth. Appearance is then aligned up to the translation
    parallax (<= ~15 px at 1.5 m depth on a 14 cm baseline) — exactly what a
    single LK pass finds. Forward/backward check + stereo_verify gate the rest.

    Maps are lazily built per pair and cached (one 512^2 unproject/rotate/
    project chain each); pairs with < min_overlap warp coverage are skipped
    forever. Also reusable for cross-camera landmark unification (mapping v2).
    """

    def __init__(self, rig, max_px=3.0, fb_max=1.5, min_cand=8, max_theta_deg=80.0,
                 min_overlap=0.03, max_depth=40.0, lk=None):
        import cv2
        self.rig = rig
        self.max_px = max_px
        self.fb_max = fb_max
        self.min_cand = min_cand
        self.cos_max = np.cos(np.deg2rad(max_theta_deg))
        self.min_overlap = min_overlap
        self.max_depth = float(max_depth)
        self.lk = lk or dict(winSize=(21, 21), maxLevel=3,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.maps: dict[tuple, object] = {}       # (i, j) -> (map_x, map_y) | None

    def _pair_map(self, i, j):
        """Resampling map rendering cam i's image in cam j's pixel grid (rotation only)."""
        key = (i, j)
        if key in self.maps:
            return self.maps[key]
        cam_i, cam_j = self.rig.cameras[i], self.rig.cameras[j]
        w, h = cam_j.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.0, np.arange(h, dtype=np.float64))
        px = np.stack([u.ravel(), v.ravel()], axis=1)
        b_j, ok_j = cam_j.model.unproject(px)
        R_ci_cj = self.rig.T_cam_cam(i, j)[:3, :3]
        b_i = b_j @ R_ci_cj.T
        src, ok_i = cam_i.model.project(b_i)
        ok = ok_j & ok_i & (b_i[:, 2] > 0.05)      # in front of cam i
        if ok.mean() < self.min_overlap:
            self.maps[key] = None
            return None
        map_x = np.where(ok, src[:, 0], -1.0).reshape(h, w).astype(np.float32)
        map_y = np.where(ok, src[:, 1], -1.0).reshape(h, w).astype(np.float32)
        self.maps[key] = (map_x, map_y)
        return self.maps[key]

    def match(self, images, masks, feats, all_hits=False):
        """feats: FrameFeatures from a per-camera front-end (disjoint id spaces).
        Returns ({track id: p_imu}, stats); with all_hits=True the dict values are
        lists of every camera-pair hit [(lk_err, p_imu), ...] for consensus tests."""
        import cv2
        hits: dict[int, list] = {}
        pts: dict[int, tuple[float, np.ndarray]] = {}   # tid -> (lk_err, p_imu)
        n_att = n_ok = 0
        for i, cam_i in enumerate(self.rig.cameras):
            ci = feats.cams[i]
            if len(ci) == 0:
                continue
            b_i = np.asarray(ci.bearings, float)
            px_i = np.asarray(ci.px, np.float32)
            ids_i = np.asarray(ci.ids)
            central_i = b_i[:, 2] > self.cos_max
            if not central_i.any():
                continue
            for j, cam_j in enumerate(self.rig.cameras):
                if j == i:
                    continue
                m = self._pair_map(i, j)
                if m is None:
                    continue
                R_cj_ci = self.rig.T_cam_cam(j, i)[:3, :3]
                b_in_j = b_i @ R_cj_ci.T
                guess, gvalid = cam_j.model.project(b_in_j)
                cand = central_i & gvalid & (b_in_j[:, 2] > self.cos_max)
                h, w = images[j].shape[:2]
                gx = np.clip(guess[:, 0], 0, w - 1).astype(int)
                gy = np.clip(guess[:, 1], 0, h - 1).astype(int)
                cand &= m[0][gy, gx] >= 0          # warp coverage at the guess
                if masks is not None and masks[j] is not None:
                    cand &= masks[j][gy, gx] > 0
                if cand.sum() < self.min_cand:
                    continue
                idx = np.nonzero(cand)[0]
                n_att += len(idx)
                warp = cv2.remap(images[i], m[0], m[1], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                g = guess[idx].astype(np.float32)
                nxt, st, err = cv2.calcOpticalFlowPyrLK(warp, images[j], g.copy(), g.copy(),
                                                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(images[j], warp, nxt, g.copy(),
                                                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
                fb = np.linalg.norm(back - g, axis=1)
                ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_max)
                ok &= (nxt[:, 0] >= 0) & (nxt[:, 0] < w) & (nxt[:, 1] >= 0) & (nxt[:, 1] < h)
                if masks is not None and masks[j] is not None:
                    xi = np.clip(nxt[:, 0].astype(int), 0, w - 1)
                    yi = np.clip(nxt[:, 1].astype(int), 0, h - 1)
                    ok &= masks[j][yi, xi] > 0
                if not ok.any():
                    continue
                sel = np.nonzero(ok)[0]
                b_j, bvalid = cam_j.model.unproject(nxt[sel])
                sel = sel[bvalid]
                if len(sel) == 0:
                    continue
                b_j = b_j[bvalid]
                p_imu, _, keep = stereo_verify_batch(self.rig, i, j, b_i[idx[sel]], b_j,
                                                     px_i[idx[sel]], nxt[sel], max_px=self.max_px,
                                                     max_depth=self.max_depth)
                e_all = err.ravel()
                for s, k_ok, p in zip(sel, keep, p_imu):
                    if not k_ok:
                        continue
                    tid = int(ids_i[idx[s]])
                    e = float(e_all[s])
                    hits.setdefault(tid, []).append((e, p))
                    if tid not in pts or e < pts[tid][0]:
                        pts[tid] = (e, p)
                    n_ok += 1
        if all_hits:
            return hits, dict(attempted=n_att, matched=n_ok, unique=len(hits))
        return {tid: p for tid, (e, p) in pts.items()}, dict(attempted=n_att, matched=n_ok, unique=len(pts))
