"""Classical front-end: Shi-Tomasi detection + pyramidal Lucas-Kanade tracking,
IMU-predicted initial flow (survives fast rotation), forward-backward check,
essential-matrix RANSAC, per-frame stereo matching by LK across cameras with
triangulation verification. Everything is OpenCV (Apache-2.0)."""
from __future__ import annotations

import cv2
import numpy as np

from .base import CamObs, FrameFeatures, Frontend
from .common import essential_inliers, stereo_verify

DEFAULTS = dict(max_features=300, min_distance=12, quality=0.01, win=21, levels=4,
                # noise-adaptive detector gate: a new corner must have a min-eigenvalue
                # response >= noise_gate x the frame's median response (the noise floor).
                # Day frames: the 300th best corner is 50-110x the floor; night frames
                # (TUM-VI room1 darkened): only 3-4x, i.e. the budget was filled with
                # sensor noise.  min_features: fall back to the plain ranking below it.
                noise_gate=4.0, min_features=60,
                fb_err_px=1.0, stereo_max_px=2.0, default_depth=3.0, rim_ang_deg=6.0,
                ransac_thr_norm=0.004, grid=0)


class KltFrontend(Frontend):
    def __init__(self, rig, config=None):
        super().__init__(rig, config)
        self.cfg = {**DEFAULTS, **(config or {})}
        self.cam0 = rig.cameras[0]
        self.n_cams = len(rig.cameras)
        self.lk = dict(winSize=(self.cfg["win"],) * 2, maxLevel=self.cfg["levels"],
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.reset()

    def reset(self):
        self.prev_img = None
        self.prev_px = np.zeros((0, 2), np.float32)
        self.prev_b = np.zeros((0, 3), np.float64)
        self.ids = np.zeros(0, np.int64)
        self.depth = {}                     # id -> last stereo depth (m)
        self.next_id = 0

    # --- helpers -----------------------------------------------------------
    def _bearings(self, cam, px):
        b, valid = cam.model.unproject(px.astype(np.float64))
        return b, valid

    def _predict_px(self, dR_imu):
        """IMU-rotation prediction of the previous cam0 features in the current image."""
        if dR_imu is None or len(self.prev_b) == 0:
            return self.prev_px.copy()
        R_ic = self.cam0.T_imu_cam[:3, :3]
        dR_cam = R_ic.T @ dR_imu @ R_ic                  # R_{C_prev <- C_cur}
        b_pred = (dR_cam.T @ self.prev_b.T).T
        px, valid = self.cam0.model.project(b_pred)
        px = px.astype(np.float32)
        px[~valid] = self.prev_px[~valid]
        return px

    def _detect(self, img, mask, n_new):
        if n_new <= 0:
            return np.zeros((0, 2), np.float32)
        m = np.full(img.shape, 255, np.uint8) if mask is None else mask.copy()
        for x, y in self.prev_px:
            cv2.circle(m, (int(x), int(y)), self.cfg["min_distance"], 0, -1)
        quality = self.cfg["quality"]
        gate = float(self.cfg.get("noise_gate", 0.0))
        if gate > 0:
            resp = cv2.cornerMinEigenVal(img, blockSize=5)
            valid = resp[m > 0] if mask is not None else resp
            floor = float(np.median(valid)) if valid.size else 0.0
            rmax = float(resp.max())
            if rmax > 0 and floor > 0:
                quality = max(quality, min(0.5, gate * floor / rmax))
        pts = cv2.goodFeaturesToTrack(img, maxCorners=int(n_new), qualityLevel=quality,
                                      minDistance=self.cfg["min_distance"], mask=m, blockSize=5)
        pts = np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2).astype(np.float32)
        need = int(self.cfg.get("min_features", 0)) - len(self.prev_px)
        if gate > 0 and quality > self.cfg["quality"] and len(pts) < need:
            # too few real corners: keep tracking on the best of what there is
            more = cv2.goodFeaturesToTrack(img, maxCorners=int(need), qualityLevel=self.cfg["quality"],
                                           minDistance=self.cfg["min_distance"], mask=m, blockSize=5)
            if more is not None:
                pts = more.reshape(-1, 2).astype(np.float32)
        return pts

    def _inside(self, px, shape, mask):
        h, w = shape
        ok = (px[:, 0] >= 1) & (px[:, 0] < w - 1) & (px[:, 1] >= 1) & (px[:, 1] < h - 1)
        if mask is not None and ok.any():
            xi = np.clip(px[:, 0].astype(int), 0, w - 1); yi = np.clip(px[:, 1].astype(int), 0, h - 1)
            ok &= mask[yi, xi] > 0
        return ok

    # --- main ----------------------------------------------------------------
    def process(self, t_ns, images, masks, dR_imu):
        img0 = images[0]
        mask0 = masks[0] if masks else None
        n_new = 0
        # 1. temporal tracking in cam0
        if self.prev_img is not None and len(self.prev_px) > 0:
            guess = self._predict_px(dR_imu)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_img, img0, self.prev_px, guess.copy(),
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(img0, self.prev_img, nxt, self.prev_px.copy(),
                                                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            fb = np.linalg.norm(back - self.prev_px, axis=1)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg["fb_err_px"])
            ok &= self._inside(nxt, img0.shape, mask0)
            px, ids, bprev = nxt[ok], self.ids[ok], self.prev_b[ok]
            b, bvalid = self._bearings(self.cam0, px)
            px, ids, bprev, b = px[bvalid], ids[bvalid], bprev[bvalid], b[bvalid]
            if len(px) >= 8:
                R_ic = self.cam0.T_imu_cam[:3, :3]
                dR_cam = None if dR_imu is None else R_ic.T @ dR_imu @ R_ic
                keep = essential_inliers(bprev, b, dR_cam, thr_norm=self.cfg["ransac_thr_norm"],
                                         ang_thr_deg=self.cfg["rim_ang_deg"])
                px, ids, b = px[keep], ids[keep], b[keep]
        else:
            px = np.zeros((0, 2), np.float32); ids = np.zeros(0, np.int64); b = np.zeros((0, 3))
        # 2. replenish
        self.prev_px = px
        new = self._detect(img0, mask0, self.cfg["max_features"] - len(px))
        if len(new):
            bn, vn = self._bearings(self.cam0, new)
            new, bn = new[vn], bn[vn]
            new_ids = np.arange(self.next_id, self.next_id + len(new), dtype=np.int64)
            self.next_id += len(new)
            px = np.concatenate([px, new]); ids = np.concatenate([ids, new_ids]); b = np.concatenate([b, bn])
            n_new = len(new)
        cams = [CamObs(ids=ids, px=px, bearings=b)]
        # 3. stereo: cam0 -> cam_j by LK with a depth-based initial guess
        for j in range(1, self.n_cams):
            cams.append(self._stereo(j, images[j], masks[j] if masks else None, img0, px, ids, b))
        # 4. state for the next frame
        self.prev_img, self.prev_px, self.prev_b, self.ids = img0, px.astype(np.float32), b, ids
        live = set(ids.tolist()); self.depth = {k: v for k, v in self.depth.items() if k in live}
        return FrameFeatures(t_ns=t_ns, cams=cams, n_new=n_new)

    def _stereo(self, j, img_j, mask_j, img0, px0, ids, b0):
        if len(px0) == 0:
            return CamObs()
        cam_j = self.rig.cameras[j]
        T_cj_c0 = self.rig.T_cam_cam(j, 0)
        depths = np.array([self.depth.get(int(i), self.cfg["default_depth"]) for i in ids])
        p_c0 = b0 * depths[:, None]
        p_cj = (T_cj_c0[:3, :3] @ p_c0.T).T + T_cj_c0[:3, 3]
        guess, gvalid = cam_j.model.project(p_cj)
        guess = guess.astype(np.float32)
        guess[~gvalid] = px0[~gvalid]
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(img0, img_j, px0.astype(np.float32), guess.copy(),
                                              flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(img_j, img0, nxt, px0.astype(np.float32).copy(),
                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        fb = np.linalg.norm(back - px0, axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg["fb_err_px"] * 1.5)
        ok &= self._inside(nxt, img_j.shape, mask_j)
        out_ids, out_px, out_b = [], [], []
        if ok.any():
            bj, vj = self._bearings(cam_j, nxt[ok])
            for k, (i, bb, vv) in enumerate(zip(np.nonzero(ok)[0], bj, vj)):
                if not vv:
                    continue
                p, depth = stereo_verify(self.rig, 0, j, b0[i], bb, px0[i], nxt[i], max_px=self.cfg["stereo_max_px"])
                if p is None:
                    continue
                self.depth[int(ids[i])] = depth
                out_ids.append(ids[i]); out_px.append(nxt[i]); out_b.append(bb)
        if not out_ids:
            return CamObs()
        return CamObs(ids=np.asarray(out_ids, np.int64), px=np.asarray(out_px, np.float32),
                      bearings=np.asarray(out_b, np.float64))
