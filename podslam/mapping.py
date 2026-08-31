"""Dense mapping on top of the VIO estimate — the SLAM output lane.

Fuses, in the world frame using ESTIMATED keyframe poses:
  * the backend's triangulated landmarks (every rig; sparse, accurate),
  * depth images where the rig has them (OAK-D style; dense),
into a voxel-averaged pointcloud (npz + PLY).  Map quality is therefore an
end-to-end measure: pose error AND reconstruction error both land in it.

    mapper = DenseMapper(voxel=0.05)
    mapper.update_landmarks(backend.lm_point)           # after each solve
    mapper.add_depth(T_W_C, cam_model, depth_hw)        # when a depth frame exists
    mapper.finalize(out_npz)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class DenseMapper:
    def __init__(self, voxel: float = 0.05, min_hits: int = 2, max_depth: float = 20.0,
                 depth_stride: int = 6):
        self.voxel = float(voxel)
        self.min_hits = int(min_hits)
        self.max_depth = float(max_depth)
        self.stride = int(depth_stride)
        self._sum: dict[tuple, np.ndarray] = {}
        self._n: dict[tuple, int] = {}
        self.landmarks: dict[int, np.ndarray] = {}
        self._ray_cache = {}

    # ------------------------------------------------------------- sources
    def update_landmarks(self, lm_point: dict) -> None:
        for j, p in lm_point.items():
            self.landmarks[j] = np.asarray(p, float)

    def add_depth(self, T_W_C: np.ndarray, model, depth: np.ndarray) -> None:
        """Backproject a depth image (H,W float32 m; 0 = invalid) into the map."""
        h, w = depth.shape
        key = (id(model), h, w)
        if key not in self._ray_cache:
            xs, ys = np.meshgrid(np.arange(0, w, self.stride) + 0.5,
                                 np.arange(0, h, self.stride) + 0.5)
            rays, ok = model.unproject(np.stack([xs.ravel(), ys.ravel()], 1))
            self._ray_cache[key] = (xs.astype(int), ys.astype(int), rays, ok)
        xs, ys, rays, ok = self._ray_cache[key]
        d = depth[ys.ravel() - 0, xs.ravel() - 0] if False else depth[np.minimum(ys.ravel(), h - 1), np.minimum(xs.ravel(), w - 1)]
        good = ok & (d > 0.15) & (d < self.max_depth)
        if not good.any():
            return
        # rays are unit; depth images store z-depth for pinhole, range for f-theta.
        rz = np.clip(rays[good, 2], 1e-6, None)
        pts_c = rays[good] * (d[good] / rz)[:, None]
        pts_w = (T_W_C[:3, :3] @ pts_c.T).T + T_W_C[:3, 3]
        self._accumulate(pts_w)

    def add_points(self, pts_w: np.ndarray) -> None:
        if len(pts_w):
            self._accumulate(np.asarray(pts_w, float))

    # -------------------------------------------------------------- fusion
    def _accumulate(self, pts: np.ndarray) -> None:
        keys = np.floor(pts / self.voxel).astype(np.int64)
        # pack voxel index to a single int for fast grouping
        packed = (keys[:, 0] + (1 << 20)) * (1 << 42) + (keys[:, 1] + (1 << 20)) * (1 << 21) + (keys[:, 2] + (1 << 20))
        order = np.argsort(packed)
        packed, pts = packed[order], pts[order]
        uniq, start, cnt = np.unique(packed, return_index=True, return_counts=True)
        sums = np.add.reduceat(pts, start, axis=0)
        for u, s, c in zip(uniq.tolist(), sums, cnt.tolist()):
            if u in self._sum:
                self._sum[u] += s
                self._n[u] += c
            else:
                self._sum[u] = s.copy()
                self._n[u] = c

    def cloud(self) -> tuple[np.ndarray, np.ndarray]:
        """(points, kind) with kind 0 = dense/depth voxel, 1 = landmark."""
        dense = [self._sum[u] / self._n[u] for u in self._sum if self._n[u] >= self.min_hits]
        lms = []
        for v in self.landmarks.values():
            a = np.asarray(v, dtype=float).reshape(-1)
            if a.size == 3 and np.isfinite(a).all():
                lms.append(a)
        rows = [np.asarray(d, dtype=float).reshape(-1) for d in dense] + lms
        pts = np.stack(rows) if rows else np.zeros((0, 3))
        kind = np.array([0] * len(dense) + [1] * len(lms), dtype=np.int8)
        return pts, kind

    def finalize(self, out: str | Path) -> dict:
        pts, kind = self.cloud()
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out.with_suffix(".npz"), points=pts, kind=kind,
                            voxel=self.voxel, min_hits=self.min_hits)
        with open(out.with_suffix(".ply"), "w") as f:
            f.write("ply\nformat ascii 1.0\n"
                    f"element vertex {len(pts)}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "end_header\n")
            for p, k in zip(pts, kind):
                c = "80 200 120" if k else "200 200 210"
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c}\n")
        return dict(n_points=int(len(pts)), n_dense=int((kind == 0).sum()),
                    n_landmarks=int((kind == 1).sum()))
