"""Realtime occupancy grid (octomap-style log-odds voxels) from the fisheye rig.

Skydio-class navigation builds its world model from the same cameras that do
VIO. Here the sources are the estimator's landmarks plus the semi-dense
cross-camera stereo points (podslam/densify.py); every point also carves the
free space along the sensor ray (Amanatides-Woo voxel traversal), so the grid
distinguishes 'seen empty' from 'unknown' — the property a planner needs and
a raw point cloud lacks.

    occ = OccupancyGrid(voxel=0.15)
    occ.integrate(origin_w, pts_w)         # per keyframe
    centers, odds = occ.occupied()
"""
from __future__ import annotations

import numpy as np

L_HIT, L_MISS = 0.85, -0.35
L_MIN, L_MAX = -2.0, 3.5
L_OCC = 1.6                     # occupied above this (~2 net hits: range noise needs corroboration)


class OccupancyGrid:
    def __init__(self, voxel: float = 0.15, max_range: float = 6.0):
        self.voxel = float(voxel)
        self.max_range = float(max_range)
        self.logodds: dict[tuple, float] = {}

    def _bump(self, key, dl):
        v = self.logodds.get(key, 0.0) + dl
        self.logodds[key] = min(max(v, L_MIN), L_MAX)

    def integrate(self, origin_w: np.ndarray, pts_w: np.ndarray) -> None:
        """Hits at the points, misses along each ray origin->point."""
        if len(pts_w) == 0:
            return
        o = np.asarray(origin_w, float)
        v = self.voxel
        for p in np.asarray(pts_w, float):
            d = p - o
            r = float(np.linalg.norm(d))
            if r < 1e-6 or r > self.max_range:
                continue
            # Amanatides-Woo traversal from origin voxel to endpoint voxel
            cur = np.floor(o / v).astype(np.int64)
            end = np.floor(p / v).astype(np.int64)
            step = np.sign(d).astype(np.int64)
            d_safe = np.where(np.abs(d) < 1e-12, 1e-12, d)
            t_delta = np.abs(v / d_safe)
            nxt = (cur + (step > 0)) * v
            t_max = (nxt - o) / d_safe
            guard = 0
            while not np.array_equal(cur, end) and guard < 200:
                self._bump(tuple(cur), L_MISS)
                a = int(np.argmin(t_max))
                cur[a] += step[a]
                t_max[a] += t_delta[a]
                guard += 1
            self._bump(tuple(end), L_HIT)

    def occupied(self):
        """(centers (N,3), log-odds (N,)) of voxels above the occupancy threshold."""
        keys = [k for k, l in self.logodds.items() if l > L_OCC]
        if not keys:
            return np.zeros((0, 3)), np.zeros(0)
        c = (np.asarray(keys, float) + 0.5) * self.voxel
        return c, np.asarray([self.logodds[k] for k in keys])

    def stats(self):
        occ = sum(1 for l in self.logodds.values() if l > L_OCC)
        free = sum(1 for l in self.logodds.values() if l < -L_OCC)
        return dict(occupied=occ, free=free, touched=len(self.logodds))
