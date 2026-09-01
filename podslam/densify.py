"""Semi-dense fisheye mapping: grid-seeded cross-camera stereo + temporal refinement.

Range, not pose, is the map's error budget (measured: rebuilding the map with
ground-truth poses does not improve it — 11.5 vs 9.8 cm median). For a pair with
baseline b at depth z the range sigma is z^2 * sigma_px / (b * f): the rig's
7-14 cm baselines give ~3 cm at 1 m but ~54 cm at 4 m, which is the radial
smear in the raw cloud. Two fixes, both here:

  1. uncertainty gate: a measurement is kept only while its predicted sigma_z
     stays under max_sigma — the usable depth follows from the ACTUAL baseline
     of the pair that saw it, instead of one global max_depth;
  2. temporal refinement: the drone's own motion is a far better baseline than
     the rig (0.3-1.5 m over a second). Each coarse point is re-projected into
     an earlier keyframe of the same camera, LK-refined there, and
     re-triangulated across the motion baseline.

    densifier = Densifier(rig, grid_step=12)
    pts_w = densifier.world_points(T_W_I, imgs, masks)   # (N,3) WORLD frame
    mapper.add_points(pts_w)
"""

from __future__ import annotations

import numpy as np

from .frontend.base import CamObs, FrameFeatures
from .frontend.common import triangulate_rays
from .init_dynamic import CrossCamMatcher


class Densifier:
    def __init__(self, rig, grid_step=12, max_px=1.5, max_depth=8.0, max_theta_deg=75.0,
                 max_sigma=0.10, sigma_px=0.5, temporal=True, min_baseline=0.25,
                 history=12, temporal_win=15):
        self.rig = rig
        self.max_sigma = float(max_sigma)
        self.sigma_px = float(sigma_px)
        self.temporal = bool(temporal)
        self.min_baseline = float(min_baseline)
        self.temporal_win = int(temporal_win)
        self.hist: list = []                 # [(T_W_I, [imgs])] recent keyframes
        self.hist_max = int(history)
        self.n_temporal = 0
        self.n_gated = 0
        self.matcher = CrossCamMatcher(rig, max_px=max_px, max_depth=max_depth,
                                       max_theta_deg=max_theta_deg, min_cand=20)
        self.grids = []                    # per-cam (px, bearings) grid seeds
        cos_max = np.cos(np.deg2rad(max_theta_deg))
        for cam in rig.cameras:
            w, h = cam.size
            xs, ys = np.meshgrid(np.arange(grid_step // 2, w, grid_step, dtype=np.float32),
                                 np.arange(grid_step // 2, h, grid_step, dtype=np.float32))
            px = np.stack([xs.ravel(), ys.ravel()], 1)
            b, ok = cam.model.unproject(px.astype(np.float64))
            ok &= b[:, 2] > cos_max
            m = cam.circle_mask()
            if m is not None:
                xi = px[:, 0].astype(int); yi = px[:, 1].astype(int)
                ok &= m[yi, xi] > 0
            self.grids.append((px[ok].astype(np.float32), b[ok]))

    def _sigma(self, z, baseline, fx):
        """Range sigma of a triangulation: z^2 * sigma_px / (b * f)."""
        return z * z * self.sigma_px / max(baseline * fx, 1e-9)

    def refine_temporal(self, T_W_I, imgs, masks, pts_body):
        """Re-triangulate coarse body-frame points across the motion baseline.
        Returns (points_world, kept_mask)."""
        import cv2
        if not len(pts_body):
            return np.zeros((0, 3)), np.zeros(0, bool)
        R_w, t_w = T_W_I[:3, :3], T_W_I[:3, 3]
        pts_w = (R_w @ np.asarray(pts_body).T).T + t_w
        out = np.array(pts_w, dtype=float, copy=True)
        kept = np.zeros(len(pts_w), bool)
        lk = dict(winSize=(15, 15), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        for ci, cam in enumerate(self.rig.cameras):
            T_ic = cam.T_imu_cam
            c_now = R_w @ T_ic[:3, 3] + t_w                     # camera centre, world
            R_cw_now = (R_w @ T_ic[:3, :3])                     # cam->world rotation
            # visible-in-this-camera subset
            p_cam = (R_cw_now.T @ (pts_w - c_now).T).T
            vis = p_cam[:, 2] > 0.2
            if vis.sum() < 8:
                continue
            idx = np.nonzero(vis)[0]
            px_now, ok_now = cam.model.project(p_cam[idx])
            idx = idx[ok_now]
            if len(idx) < 8:
                continue
            px_now = px_now[ok_now].astype(np.float32)
            # oldest history frame that gives a usable baseline
            best = None
            for T_old, imgs_old in self.hist:
                c_old = T_old[:3, :3] @ T_ic[:3, 3] + T_old[:3, 3]
                b = float(np.linalg.norm(c_now - c_old))
                if b >= self.min_baseline:
                    best = (T_old, imgs_old, c_old, b)
                    break
            if best is None:
                continue
            T_old, imgs_old, c_old, base = best
            R_cw_old = T_old[:3, :3] @ T_ic[:3, :3]
            p_old = (R_cw_old.T @ (pts_w[idx] - c_old).T).T
            guess, ok_g = cam.model.project(p_old)
            sel = ok_g & (p_old[:, 2] > 0.2)
            if sel.sum() < 8:
                continue
            j = idx[sel]
            g = guess[sel].astype(np.float32)
            src = px_now[sel]
            nxt, st, _e = cv2.calcOpticalFlowPyrLK(imgs[ci], imgs_old[ci], src, g.copy(),
                                                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            back, st2, _e2 = cv2.calcOpticalFlowPyrLK(imgs_old[ci], imgs[ci], nxt, src.copy(),
                                                      flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            good = (st.ravel() == 1) & (st2.ravel() == 1)
            good &= np.linalg.norm(back - src, axis=1) < 1.5
            if not good.any():
                continue
            b_old, v_old = cam.model.unproject(nxt[good].astype(np.float64))
            b_now, v_now = cam.model.unproject(src[good].astype(np.float64))
            jj = j[good]
            d_now = b_now @ R_cw_now.T
            d_old = b_old @ R_cw_old.T
            for n, k in enumerate(jj):
                if not (v_old[n] and v_now[n]):
                    continue
                p, s0, s1, ok = triangulate_rays(c_now, d_now[n], c_old, d_old[n])
                if not ok or s0 < 0.2 or s1 < 0.2:
                    continue
                if self._sigma(s0, base, cam.model.fx) > self.max_sigma:
                    continue
                # sanity: the refinement must stay near the coarse point
                if np.linalg.norm(p - pts_w[k]) > 1.0:
                    continue
                out[k] = p
                kept[k] = True
                self.n_temporal += 1
        return out, kept

    def world_points(self, T_W_I, imgs, masks):
        """Keyframe entry point: coarse cross-camera points, temporally refined
        where the motion baseline allows, uncertainty-gated, in WORLD frame."""
        coarse = self.points(imgs, masks)
        R_w, t_w = T_W_I[:3, :3], T_W_I[:3, 3]
        if not len(coarse):
            self._push(T_W_I, imgs)
            return np.zeros((0, 3))
        if self.temporal and self.hist:
            pts_w, kept = self.refine_temporal(T_W_I, imgs, masks, coarse)
        else:
            pts_w = (R_w @ coarse.T).T + t_w
            kept = np.zeros(len(coarse), bool)
        # uncertainty gate for points that only ever had the rig baseline:
        # keep them while their own sigma is voxel-scale
        if (~kept).any():
            rng = np.linalg.norm(pts_w[~kept] - t_w, axis=1)
            fx = float(np.mean([c.model.fx for c in self.rig.cameras]))
            b_rig = float(np.mean([np.linalg.norm(c.T_imu_cam[:3, 3]) for c in self.rig.cameras])) * 2.0
            ok_rig = self._sigma(rng, max(b_rig, 1e-3), fx) <= self.max_sigma
            self.n_gated += int((~ok_rig).sum())
            idx = np.nonzero(~kept)[0]
            kept[idx[ok_rig]] = True
        self._push(T_W_I, imgs)
        return pts_w[kept]

    def _push(self, T_W_I, imgs):
        self.hist.insert(0, (np.array(T_W_I, copy=True), [im.copy() for im in imgs]))
        del self.hist[self.hist_max:]

    def points(self, imgs, masks, consensus_tol=0.15, min_pairs=1):
        """Cross-camera verified metric points, body/IMU frame (N,3).
        A seed only survives when >= 2 camera pairs agree on its 3D position
        within consensus_tol — the short-baseline range noise of a single pair
        is the map's dominant error, and independent pairs rarely agree on a
        wrong range."""
        cams = []
        base = 0
        for px, b in self.grids:
            ids = np.arange(base, base + len(px), dtype=np.int64)
            base += len(px)
            cams.append(CamObs(ids=ids, px=px, bearings=b))
        feats = FrameFeatures(t_ns=0, cams=cams)
        hits, _ = self.matcher.match(imgs, masks, feats, all_hits=True)
        out = []
        for tid, hs in hits.items():
            if len(hs) < min_pairs:
                continue
            if min_pairs < 2:
                out.append(min(hs, key=lambda h: h[0])[1])
                continue
            ps = np.asarray([p for _, p in hs])
            med = np.median(ps, axis=0)
            close = np.linalg.norm(ps - med, axis=1) <= consensus_tol
            if close.sum() >= 2:
                out.append(ps[close].mean(axis=0))
        return np.asarray(out) if out else np.zeros((0, 3))
