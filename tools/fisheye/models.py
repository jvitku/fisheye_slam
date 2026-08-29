"""Fisheye camera models used across the fisheye_slam evaluation.

The three model families that matter for this project (see docs/REPORT.md):

- KannalaBrandt4 ("kb4", OpenCV-fisheye / equidistant polynomial): what
  ORB-SLAM3/OpenMAVIS and Kalibr's "pinhole-equi" use.
- DoubleSphere ("ds"): Basalt's native model, best accuracy/cost for >180 deg
  lenses, closed-form unprojection.
- EUCM: extended unified model, closed-form both ways, used by several
  multi-fisheye depth pipelines.

All models implement:
    project(points)   : (N,3) camera-frame points -> (N,2) pixels, (N,) validity
    unproject(pixels) : (N,2) pixels -> (N,3) unit bearing vectors, (N,) validity

Conventions: x right, y down, z forward (optical axis). Vectorized numpy, float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_points(a, dim: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != dim:
        raise ValueError(f"expected (N,{dim}) array, got {a.shape}")
    return a


@dataclass
class KannalaBrandt4:
    """Equidistant polynomial fisheye model (OpenCV fisheye, Kalibr pinhole-equi).

    r(theta) = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
    """

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    # Rays beyond this angle from the optical axis are reported invalid.
    max_theta: float = np.deg2rad(110.0)

    def _d(self, theta: np.ndarray) -> np.ndarray:
        t2 = theta * theta
        return theta * (1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4))))

    def _d_prime(self, theta: np.ndarray) -> np.ndarray:
        t2 = theta * theta
        return 1.0 + t2 * (3 * self.k1 + t2 * (5 * self.k2 + t2 * (7 * self.k3 + t2 * 9 * self.k4)))

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        p = _as_points(points, 3)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        valid = theta < self.max_theta
        d = self._d(theta)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(r > 1e-12, d / r, 1.0)  # small-angle: d/r -> ~1 with theta≈r/z
        # exact small-angle handling: for r ~ 0, u = f*theta*(x/r) -> f*x/z
        near_axis = r <= 1e-12
        u = np.where(near_axis, self.fx * x / np.maximum(z, 1e-12), self.fx * x * scale) + self.cx
        v = np.where(near_axis, self.fy * y / np.maximum(z, 1e-12), self.fy * y * scale) + self.cy
        return np.stack([u, v], axis=1), valid

    def unproject(self, pixels) -> tuple[np.ndarray, np.ndarray]:
        px = _as_points(pixels, 2)
        mx = (px[:, 0] - self.cx) / self.fx
        my = (px[:, 1] - self.cy) / self.fy
        rd = np.hypot(mx, my)
        # Solve d(theta) = rd by Newton iterations (d is monotonic on [0, max_theta]
        # for sane calibrations).
        theta = rd.copy()
        for _ in range(8):
            step = (self._d(theta) - rd) / self._d_prime(theta)
            theta = theta - step
        theta = np.clip(theta, 0.0, None)
        valid = (theta < self.max_theta) & (np.abs(self._d(theta) - rd) < 1e-9)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(rd > 1e-12, np.sin(theta) / rd, 1.0)
        bearings = np.stack([mx * s, my * s, np.cos(theta)], axis=1)
        # near axis: sin(theta)/rd -> 1 only if d(theta)~theta~rd; fine for kb4
        bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
        return bearings, valid


@dataclass
class DoubleSphere:
    """Double Sphere model (Usenko et al. 2018), Basalt's native fisheye model."""

    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    alpha: float

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        p = _as_points(points, 3)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        d1 = np.sqrt(x * x + y * y + z * z)
        zeta = self.xi * d1 + z
        d2 = np.sqrt(x * x + y * y + zeta * zeta)
        denom = self.alpha * d2 + (1.0 - self.alpha) * zeta
        # Validity (projection region, from the paper): z > -w2 * d1
        w1 = self.alpha / (1.0 - self.alpha) if self.alpha <= 0.5 else (1.0 - self.alpha) / self.alpha
        w2 = (w1 + self.xi) / np.sqrt(2.0 * w1 * self.xi + self.xi * self.xi + 1.0)
        valid = (z > -w2 * d1) & (denom > 1e-12)
        denom = np.where(valid, denom, 1.0)
        u = self.fx * x / denom + self.cx
        v = self.fy * y / denom + self.cy
        return np.stack([u, v], axis=1), valid

    def unproject(self, pixels) -> tuple[np.ndarray, np.ndarray]:
        px = _as_points(pixels, 2)
        mx = (px[:, 0] - self.cx) / self.fx
        my = (px[:, 1] - self.cy) / self.fy
        r2 = mx * mx + my * my
        if self.alpha > 0.5:
            valid = r2 <= 1.0 / (2.0 * self.alpha - 1.0)
        else:
            valid = np.ones_like(r2, dtype=bool)
        disc = 1.0 - (2.0 * self.alpha - 1.0) * r2
        disc = np.where(disc > 0, disc, 0.0)
        mz = (1.0 - self.alpha * self.alpha * r2) / (
            self.alpha * np.sqrt(disc) + 1.0 - self.alpha
        )
        k = (mz * self.xi + np.sqrt(mz * mz + (1.0 - self.xi * self.xi) * r2)) / (mz * mz + r2)
        bearings = np.stack([k * mx, k * my, k * mz - self.xi], axis=1)
        bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
        return bearings, valid


@dataclass
class EUCM:
    """Extended Unified Camera Model (Khomutenko et al. 2016)."""

    fx: float
    fy: float
    cx: float
    cy: float
    alpha: float
    beta: float

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        p = _as_points(points, 3)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        rho2 = self.beta * (x * x + y * y) + z * z
        rho = np.sqrt(rho2)
        denom = self.alpha * rho + (1.0 - self.alpha) * z
        # projection region: z > -w * rho
        if self.alpha <= 0.5:
            w = self.alpha / (1.0 - self.alpha)
        else:
            w = (1.0 - self.alpha) / self.alpha
        valid = (z > -w * rho) & (denom > 1e-12)
        denom = np.where(valid, denom, 1.0)
        u = self.fx * x / denom + self.cx
        v = self.fy * y / denom + self.cy
        return np.stack([u, v], axis=1), valid

    def unproject(self, pixels) -> tuple[np.ndarray, np.ndarray]:
        px = _as_points(pixels, 2)
        mx = (px[:, 0] - self.cx) / self.fx
        my = (px[:, 1] - self.cy) / self.fy
        r2 = mx * mx + my * my
        gamma = 1.0 - self.alpha
        disc = 1.0 - (self.alpha - gamma) * self.beta * r2
        valid = disc >= 0.0
        if self.alpha > 0.5:
            valid &= r2 <= 1.0 / (self.beta * (2.0 * self.alpha - 1.0))
        disc = np.where(disc > 0, disc, 0.0)
        mz = (1.0 - self.beta * self.alpha * self.alpha * r2) / (
            self.alpha * np.sqrt(disc) + gamma
        )
        bearings = np.stack([mx, my, mz], axis=1)
        bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
        return bearings, valid


_MODEL_REGISTRY = {
    "kb4": KannalaBrandt4,
    "ds": DoubleSphere,
    "eucm": EUCM,
}


def load_camera(cfg: dict):
    """Build a camera model from a rig-yaml camera dict: {model: kb4, intrinsics: {...}}."""
    model = cfg["model"]
    if model not in _MODEL_REGISTRY:
        raise ValueError(f"unknown camera model '{model}', expected one of {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[model](**cfg["intrinsics"])
