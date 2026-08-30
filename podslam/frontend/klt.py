"""Classical front-end: Shi-Tomasi detection + pyramidal Lucas-Kanade tracking,
IMU-predicted initial flow (survives fast rotation), forward-backward check,
essential-matrix RANSAC, per-frame stereo matching by LK across cameras with
triangulation verification. Everything is OpenCV (Apache-2.0)."""
from __future__ import annotations

import cv2
import numpy as np

from .base import CamObs, FrameFeatures, Frontend
from .common import essential_inliers, stereo_verify, stereo_verify_batch

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
        self.tmpl = {}                      # id -> (template image, px at template)
        self.tmpl_imgs = {}                 # id(img) -> retained reference frames
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
            if self.cfg.get("kf_template", False) and len(px):
                px = self._refine_against_templates(img0, px, ids)
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

    def _refine_against_templates(self, img0, px, ids):
        """Basalt-style drift correction: re-align each track against the patch from the
        frame where it was (re-)anchored, instead of pure frame-to-frame chaining.
        Small corrections only (<= tmpl_max_corr px); a large deviation or a failed
        match re-anchors the template at the current frame."""
        max_corr = float(self.cfg.get("tmpl_max_corr", 2.0))
        reanchor_px = float(self.cfg.get("tmpl_reanchor_px", 45.0))
        groups = {}
        for n, tid in enumerate(ids):
            tid = int(tid)
            t = self.tmpl.get(tid)
            if t is None or np.linalg.norm(px[n] - t[1]) > reanchor_px:
                self.tmpl[tid] = (self.tmpl_imgs.setdefault(id(img0), img0), px[n].copy())
                continue
            groups.setdefault(id(t[0]), []).append(n)
        for img_id, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:5]:
            idxs = np.asarray(idxs)
            ref = self.tmpl_imgs.get(img_id)
            if ref is None:
                continue
            ref_px = np.stack([self.tmpl[int(ids[n])][1] for n in idxs]).astype(np.float32)
            out, st, _ = cv2.calcOpticalFlowPyrLK(ref, img0, ref_px, px[idxs].copy().astype(np.float32),
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            corr = np.linalg.norm(out - px[idxs], axis=1)
            good = (st.ravel() == 1) & (corr <= max_corr)
            px[idxs[good]] = out[good]
            for n in idxs[~good]:                       # template no longer matches: re-anchor
                self.tmpl[int(ids[n])] = (self.tmpl_imgs.setdefault(id(img0), img0), px[n].copy())
        # forget templates of dead tracks and unreferenced images
        live = set(int(i) for i in ids)
        self.tmpl = {t: v for t, v in self.tmpl.items() if t in live}
        used = {id(v[0]) for v in self.tmpl.values()} | {id(img0)}
        self.tmpl_imgs = {k2: v for k2, v in self.tmpl_imgs.items() if k2 in used}
        return px

    def set_depths(self, depths):
        for i, d in depths.items():
            if np.isfinite(d) and d > 0.05:
                self.depth[int(i)] = float(d)

    def _epipolar_seed(self, img0, img_j, px0, b0, T_cj_c0, cam_j, sel, half=4):
        """Depth seed per selected feature: max ZNCC over candidate depths along the
        epipolar curve in cam_j (9x9 patches, integer positions)."""
        n = len(px0); out = np.full(n, np.nan)
        idx = np.nonzero(sel)[0]
        if len(idx) == 0:
            return out
        cands = np.asarray(self.cfg.get("stereo_zncc_depths", (0.4, 0.55, 0.75, 1.0, 1.35, 1.8, 2.4, 3.2, 4.3, 5.8, 8.0, 11.0, 16.0, 25.0)))
        h, w = img0.shape
        yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
        def patches(img, pts):                                  # (m, 81) float32, NaN outside
            xs = np.rint(pts[:, 0]).astype(int)[:, None, None] + xx; ys = np.rint(pts[:, 1]).astype(int)[:, None, None] + yy
            ok = (xs.min((1, 2)) >= 0) & (xs.max((1, 2)) < w) & (ys.min((1, 2)) >= 0) & (ys.max((1, 2)) < h)
            p = np.zeros((len(pts), (2 * half + 1) ** 2), np.float32)
            if ok.any():
                p[ok] = img[ys[ok], xs[ok]].reshape(int(ok.sum()), -1).astype(np.float32)
            p -= p.mean(1, keepdims=True); p /= (np.linalg.norm(p, axis=1, keepdims=True) + 1e-6)
            return p, ok
        p0, ok0 = patches(img0, px0[idx])
        best = np.full(len(idx), -1.0); best_d = np.full(len(idx), np.nan)
        for d in cands:
            p_cj = (T_cj_c0[:3, :3] @ (b0[idx] * d).T).T + T_cj_c0[:3, 3]
            pj, valid = cam_j.model.project(p_cj)
            pj = np.where(valid[:, None], pj, -1e4)
            pjp, okj = patches(img_j, pj)
            sc = np.where(ok0 & okj, (p0 * pjp).sum(1), -1.0)
            better = sc > best
            best[better] = sc[better]; best_d[better] = d
        good = best >= float(self.cfg.get("stereo_zncc_min", 0.6))
        out[idx[good]] = best_d[good]
        return out

    def _stereo(self, j, img_j, mask_j, img0, px0, ids, b0):
        if len(px0) == 0:
            return CamObs()
        cam_j = self.rig.cameras[j]
        T_cj_c0 = self.rig.T_cam_cam(j, 0)
        # Multi-hypothesis initial guess along the epipolar curve: LK only converges
        # from a guess within its pyramid reach (~40 px), and a single default depth
        # biases the matches towards it (Hilti exp14: +5 % scale, 27 cm ATE with a
        # 1.5 m default).  Run LK from the track's last depth (if known) and from a
        # few fixed depths, keep the forward/backward-consistent match with the
        # lowest LK error, then verify it geometrically (stereo_verify).
        known = np.array([self.depth.get(int(i), np.nan) for i in ids])
        # New features (no depth yet): seed the depth by a coarse ZNCC search along the
        # epipolar curve instead of a fixed default.  Measured on Hilti exp14: LK from a
        # 3 m guess converges to matches biased towards the guess (+8 % depth < 1.5 m,
        # -4 % > 3 m -> +2 % trajectory scale).
        if self.cfg.get("stereo_zncc", False) and (~np.isfinite(known)).any():   # off: no gain (Hilti), hurts TUM-VI
            seed = self._epipolar_seed(img0, img_j, px0, b0, T_cj_c0, cam_j, ~np.isfinite(known))
            known = np.where(np.isfinite(known), known, seed)
        hyps = [np.where(np.isfinite(known), known, self.cfg["default_depth"])]
        # extra fixed-depth hypotheses are OFF by default: choosing among them by LK error
        # picks photometrically-best but geometrically-wrong matches on repetitive texture
        # (TUM-VI room1 posters/checkerboards: day 8.8 -> 13.7 cm); no gain on Hilti either
        hyps += [np.full(len(ids), float(d)) for d in self.cfg.get("stereo_depths", ())]
        best_nxt = np.zeros((len(ids), 2), np.float32); best_err = np.full(len(ids), np.inf, np.float32)
        px0f = px0.astype(np.float32)
        for depths in hyps:
            p_c0 = b0 * depths[:, None]
            p_cj = (T_cj_c0[:3, :3] @ p_c0.T).T + T_cj_c0[:3, 3]
            guess, gvalid = cam_j.model.project(p_cj)
            guess = guess.astype(np.float32)
            guess[~gvalid] = px0f[~gvalid]
            nxt_h, st, err = cv2.calcOpticalFlowPyrLK(img0, img_j, px0f, guess.copy(),
                                                      flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(img_j, img0, nxt_h, px0f.copy(),
                                                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            fb = np.linalg.norm(back - px0f, axis=1)
            ok_h = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg["fb_err_px"] * 1.5)
            ok_h &= self._inside(nxt_h, img_j.shape, mask_j)
            e = np.where(ok_h, err.ravel(), np.inf).astype(np.float32)
            better = e < best_err
            best_nxt[better] = nxt_h[better]; best_err[better] = e[better]
        # second pass (stereo_passes >= 2): re-initialise every accepted match from the
        # depth it implies and let LK converge again; the coarse-to-fine solution is
        # biased towards its initial guess (Hilti exp14 basement: walls nearer than the
        # 3 m default -> depths +2.5 % too far -> +2.5 % trajectory scale)
        for _ in range(int(self.cfg.get("stereo_passes", 1)) - 1):
            ok1 = np.isfinite(best_err)
            if not ok1.any():
                break
            d_new = known.copy()
            bj1, vj1 = self._bearings(cam_j, best_nxt[ok1])
            for i, bb, vv in zip(np.nonzero(ok1)[0], bj1, vj1):
                if vv:
                    _, dep = stereo_verify(self.rig, 0, j, b0[i], bb, px0[i], best_nxt[i], max_px=self.cfg["stereo_max_px"] * 2)
                    if dep > 0:
                        d_new[i] = dep
            depths = np.where(np.isfinite(d_new), d_new, self.cfg["default_depth"])
            p_cj = (T_cj_c0[:3, :3] @ (b0 * depths[:, None]).T).T + T_cj_c0[:3, 3]
            guess, gvalid = cam_j.model.project(p_cj); guess = guess.astype(np.float32); guess[~gvalid] = px0f[~gvalid]
            nxt_h, st, err = cv2.calcOpticalFlowPyrLK(img0, img_j, px0f, guess.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(img_j, img0, nxt_h, px0f.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            fb = np.linalg.norm(back - px0f, axis=1)
            ok_h = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg["fb_err_px"] * 1.5) & self._inside(nxt_h, img_j.shape, mask_j)
            best_nxt[ok_h] = nxt_h[ok_h]; best_err[ok_h] = err.ravel()[ok_h]
        nxt = best_nxt
        ok = np.isfinite(best_err)
        if not ok.any():
            return CamObs()
        idx = np.nonzero(ok)[0]
        bj, vj = self._bearings(cam_j, nxt[idx])
        idx = idx[vj]; bj = bj[vj]
        if len(idx) == 0:
            return CamObs()
        _, depths_ok, keep = stereo_verify_batch(self.rig, 0, j, b0[idx], bj, px0[idx], nxt[idx],
                                                 max_px=self.cfg["stereo_max_px"])
        idx2 = idx[keep]
        for i, dep in zip(idx2, depths_ok[keep]):
            self.depth[int(ids[i])] = float(dep)
        if len(idx2) == 0:
            return CamObs()
        return CamObs(ids=ids[idx2].astype(np.int64), px=nxt[idx2].astype(np.float32),
                      bearings=bj[keep].astype(np.float64))
