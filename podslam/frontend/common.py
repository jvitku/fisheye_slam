"""Shared geometry for front-ends: stereo verification and outlier rejection."""
from __future__ import annotations

import numpy as np


def triangulate_rays(o0, d0, o1, d1):
    """Midpoint of the closest points of two rays. Returns (point, s0, s1, ok)."""
    w = o0 - o1
    a, b, c = float(d0 @ d0), float(d0 @ d1), float(d1 @ d1)
    d, e = float(d0 @ w), float(d1 @ w)
    den = a * c - b * b
    if den < 1e-9:
        return None, 0.0, 0.0, False
    s0 = (b * e - c * d) / den
    s1 = (a * e - b * d) / den
    p = 0.5 * ((o0 + s0 * d0) + (o1 + s1 * d1))
    return p, s0, s1, True


def stereo_verify(rig, cam_i, cam_j, b_i, b_j, px_i, px_j, max_px=2.0, min_depth=0.2, max_depth=40.0,
                  min_parallax_deg=0.5):
    """Triangulate a cam_i/cam_j bearing pair in the IMU frame; accept if both
    depths are positive, the rays are not (near-)parallel — a point with no
    parallax has no depth information and makes the solver's linear system
    singular — and the reprojection error is small. Returns (point_imu | None, depth_i)."""
    ci, cj = rig.cameras[cam_i], rig.cameras[cam_j]
    Ri, ti = ci.T_imu_cam[:3, :3], ci.T_imu_cam[:3, 3]
    Rj, tj = cj.T_imu_cam[:3, :3], cj.T_imu_cam[:3, 3]
    if not (np.all(np.isfinite(b_i)) and np.all(np.isfinite(b_j))):
        return None, 0.0
    di, dj = Ri @ b_i, Rj @ b_j
    if np.degrees(np.arccos(np.clip(di @ dj, -1.0, 1.0))) < min_parallax_deg:
        return None, 0.0
    p, si, sj, ok = triangulate_rays(ti, di, tj, dj)
    if not ok or not np.all(np.isfinite(p)) or si < min_depth or sj < min_depth or si > max_depth:
        return None, 0.0
    # reprojection check
    pi = Ri.T @ (p - ti); pj = Rj.T @ (p - tj)
    ui, vi = ci.model.project(pi[None]); uj, vj = cj.model.project(pj[None])
    if not (vi[0] and vj[0]):
        return None, 0.0
    if np.linalg.norm(ui[0] - px_i) > max_px or np.linalg.norm(uj[0] - px_j) > max_px:
        return None, 0.0
    return p, float(si)


def essential_inliers(b_prev, b_cur, dR_cam=None, max_theta_deg=80.0, thr_norm=0.004, ang_thr_deg=6.0):
    """Outlier rejection for temporal matches given bearings in the same camera.
    Central region (theta < max_theta): 5-point RANSAC on normalized coordinates.
    Rim: rotation-compensated angular residual against the IMU-predicted bearing
    (dR_cam = R_{C_prev <- C_cur}). Returns a boolean inlier mask."""
    import cv2
    n = len(b_prev)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    cz = np.cos(np.deg2rad(max_theta_deg))
    central = (b_prev[:, 2] > cz) & (b_cur[:, 2] > cz)
    idx = np.nonzero(central)[0]
    if len(idx) >= 8:
        p = b_prev[idx, :2] / b_prev[idx, 2:3]
        c = b_cur[idx, :2] / b_cur[idx, 2:3]
        E, mask = cv2.findEssentialMat(p.astype(np.float64), c.astype(np.float64), focal=1.0, pp=(0.0, 0.0),
                                       method=cv2.RANSAC, prob=0.999, threshold=thr_norm)
        if mask is not None:
            keep[idx] = mask.ravel().astype(bool)
    if dR_cam is not None:
        rim = ~central
        if rim.any():
            pred = (dR_cam.T @ b_prev[rim].T).T          # previous bearing expressed in the current camera
            cosang = np.sum(pred * b_cur[rim], axis=1).clip(-1, 1)
            keep[rim] = np.degrees(np.arccos(cosang)) < ang_thr_deg
    return keep
