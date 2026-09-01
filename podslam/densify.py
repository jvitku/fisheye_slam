"""Semi-dense fisheye mapping: grid-seeded cross-camera stereo per keyframe.

A divergent multi-camera rig has no depth sensor, so its map is sparse
landmarks. The rig's inter-camera baselines still carry metric depth in every
overlap region — the same rotation-warp LK matching that powers the
moving-platform initialiser, seeded on a REGULAR GRID instead of tracked
corners, yields thousands of verified metric points per keyframe. Voxel
fusion across keyframes (DenseMapper, min_hits) averages the per-point range
noise of the short baselines.

    densifier = Densifier(rig, grid_step=14)
    pts_imu = densifier.points(imgs, masks)          # (N,3) body frame
    mapper.add_points((T_W_I[:3,:3] @ pts_imu.T).T + T_W_I[:3,3])
"""
from __future__ import annotations

import numpy as np

from .frontend.base import CamObs, FrameFeatures
from .init_dynamic import CrossCamMatcher


class Densifier:
    def __init__(self, rig, grid_step=12, max_px=1.5, max_depth=5.0, max_theta_deg=75.0):
        self.rig = rig
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
