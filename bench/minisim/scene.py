"""Minisim scene definitions — the single source of truth for BOTH render lanes
(the fast analytic raycaster and the Blender realism lane) and for the map
evaluation (exact point-to-surface distances).

Scenes are lists of primitives with procedural albedo:
    {"kind": "room",     "center", "half"}                    interior of a box
    {"kind": "box",      "center", "half"}                    solid box
    {"kind": "cylinder", "center_xy", "radius", "z0", "z1"}   vertical cylinder
    {"kind": "ground",   "z"}                                 infinite floor
Each carries texture parameters (base albedo, contrast, noise scale) rendered by
value-noise of the 3-D hit position — deterministic, no UV maps, identical in
both lanes.

Trajectories are analytic C2 functions of time (position + yaw), so IMU truth
comes from differentiation, like podslam/tests/test_synthetic.py.

Conditions (shared spec):
    day         ambient + directional sun
    night       zero ambient; only the rig's IR flood(s), inverse-square + cone
    day_fog     day + homogeneous fog (Koschmieder, beta [1/m])
    night_fog   night + fog: flood backscatter airlight grows near the camera
    day_dust    day + propwash dust particles near the cameras
    night_dust  night + dust: particles backscatter the flood (bright blobs)
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- textures


def _hash3(q: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random [0,1) per integer 3-vector (vectorised)."""
    h = (q[..., 0] * 374761393 + q[..., 1] * 668265263 + q[..., 2] * 2147483647) & 0x7FFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0x7FFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF).astype(np.float64) / float(0xFFFFFF)


def value_noise(p: np.ndarray, scale: float) -> np.ndarray:
    """Trilinear value noise of 3-D points (N,3) at the given spatial scale [m]."""
    q = p / scale
    q0 = np.floor(q).astype(np.int64)
    f = q - q0
    f = f * f * (3.0 - 2.0 * f)                      # smoothstep
    out = np.zeros(p.shape[:-1])
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner = q0 + np.array([dx, dy, dz])
                w = (f[..., 0] if dx else 1 - f[..., 0]) * \
                    (f[..., 1] if dy else 1 - f[..., 1]) * \
                    (f[..., 2] if dz else 1 - f[..., 2])
                out += w * _hash3(corner)
    return out


def albedo(p: np.ndarray, tex: dict) -> np.ndarray:
    """Procedural albedo in [0,1] for hit points (N,3)."""
    a = tex.get("base", 0.5)
    out = np.full(p.shape[:-1], float(a))
    for scale, amp in tex.get("octaves", ((0.4, 0.25), (0.08, 0.15))):
        out += amp * (value_noise(p, scale) - 0.5) * 2.0
    return np.clip(out, 0.02, 1.0)


# ---------------------------------------------------------------- scenes

TEX_WALL = {"base": 0.55, "octaves": ((0.9, 0.18), (0.16, 0.14), (0.035, 0.06))}
TEX_FURN = {"base": 0.42, "octaves": ((0.35, 0.22), (0.06, 0.12))}
TEX_FLOOR = {"base": 0.38, "octaves": ((0.5, 0.15), (0.07, 0.12))}
TEX_BARK = {"base": 0.30, "octaves": ((0.5, 0.10), (0.05, 0.22), (0.015, 0.08))}
TEX_GROUND = {"base": 0.33, "octaves": ((1.2, 0.15), (0.18, 0.18), (0.04, 0.10))}
TEX_ROCK = {"base": 0.47, "octaves": ((0.25, 0.2), (0.05, 0.1))}


def indoor_scene() -> list[dict]:
    """A 10 x 7 x 3.2 m room with furniture boxes and columns."""
    rng = np.random.default_rng(42)
    prims = [
        {"kind": "room", "center": [0.0, 0.0, 1.6], "half": [5.0, 3.5, 1.6], "tex": TEX_WALL},
    ]
    # furniture: boxes along walls and in the middle
    for i in range(11):
        w, d, h = rng.uniform(0.4, 1.6), rng.uniform(0.3, 1.0), rng.uniform(0.4, 1.5)
        x = rng.uniform(-4.2, 4.2)
        y = rng.choice([-1, 1]) * rng.uniform(1.2, 3.0) if i < 7 else rng.uniform(-1.0, 1.0)
        prims.append({"kind": "box", "center": [x, float(y), h / 2], "half": [w / 2, d / 2, h / 2],
                      "tex": dict(TEX_FURN, base=float(rng.uniform(0.3, 0.6)))})
    for x, y in ((-2.5, 0.0), (2.5, 0.6)):
        prims.append({"kind": "cylinder", "center_xy": [x, y], "radius": 0.16, "z0": 0.0, "z1": 3.2,
                      "tex": TEX_ROCK})
    return prims


def forest_scene() -> list[dict]:
    """Ground + ~60 tree trunks and obstacles within 20 m of a 26 x 14 m loop path."""
    rng = np.random.default_rng(7)
    prims = [{"kind": "ground", "z": 0.0, "tex": TEX_GROUND}]
    # the flight loop (see forest_trajectory) is an ellipse a=13, b=7 at h~2;
    # plant trees everywhere except a 2.2 m corridor around the path
    def on_path(x, y):
        th = np.arctan2(y / 7.0, x / 13.0)
        px, py = 13.0 * np.cos(th), 7.0 * np.sin(th)
        return np.hypot(x - px, y - py) < 2.4
    n = 0
    while n < 60:
        x, y = rng.uniform(-22, 22), rng.uniform(-16, 16)
        if on_path(x, y):
            continue
        r = rng.uniform(0.10, 0.42)
        h = rng.uniform(5.0, 14.0)
        prims.append({"kind": "cylinder", "center_xy": [float(x), float(y)], "radius": float(r),
                      "z0": 0.0, "z1": float(h), "tex": dict(TEX_BARK, base=float(rng.uniform(0.22, 0.4)))})
        n += 1
    for _ in range(10):                                    # rocks / fallen logs
        x, y = rng.uniform(-20, 20), rng.uniform(-14, 14)
        if on_path(x, y):
            continue
        prims.append({"kind": "box", "center": [float(x), float(y), 0.25],
                      "half": [float(rng.uniform(0.2, 1.4)), float(rng.uniform(0.2, 0.5)), 0.25],
                      "tex": TEX_ROCK})
    return prims


# ------------------------------------------------------------- trajectories


def _yaw_R(yaw, pitch=0.0, roll=0.0):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def indoor_trajectory(t: float):
    """Scan pattern inside the room; two fast-yaw segments; 110 s."""
    ramp = min(t / 6.0, 1.0) if t > 1.5 else 0.0           # static first 1.5 s
    x = 3.0 * np.sin(2 * np.pi * t / 40.0) * ramp
    y = 1.9 * np.sin(2 * np.pi * t / 23.0 + 1.0) * ramp
    z = 1.5 + 0.5 * np.sin(2 * np.pi * t / 17.0) * ramp
    yaw = 0.9 * np.sin(2 * np.pi * t / 31.0) * ramp
    for t0 in (45.0, 80.0):                                 # fast yaw sweeps, ~2.2 rad/s peak
        if t0 <= t < t0 + 4.0:
            yaw += np.pi * (1 - np.cos(np.pi * (t - t0) / 4.0)) / 2.0 * 1.4
        elif t >= t0 + 4.0:
            yaw += 1.4 * np.pi
    pitch = 0.12 * np.sin(2 * np.pi * t / 9.0) * ramp
    roll = 0.10 * np.sin(2 * np.pi * t / 7.0 + 0.5) * ramp
    return _yaw_R(yaw, pitch, roll), np.array([x, y, z])


def forest_trajectory(t: float):
    """Elliptical loop (a=13, b=7) at ~2.2 m/s with weave and bobbing; 110 s."""
    ramp = min(max(t - 1.5, 0.0) / 6.0, 1.0)
    th = 2 * np.pi * (t * ramp * 0.5 + 0.0) / 55.0          # one lap ~ 55 s after ramp
    x = 13.0 * np.sin(th)
    y = -7.0 * np.cos(th) + 7.0                              # start at origin edge
    z = 2.0 + 0.6 * np.sin(2 * np.pi * t / 11.0) * ramp
    # face along the velocity + weave
    yaw = np.arctan2(7.0 * np.sin(th), 13.0 * np.cos(th)) if ramp > 0 else 0.0
    yaw += 0.35 * np.sin(2 * np.pi * t / 8.0) * ramp
    pitch = 0.10 * np.sin(2 * np.pi * t / 6.5) * ramp
    roll = 0.12 * np.sin(2 * np.pi * t / 5.0 + 1.0) * ramp
    return _yaw_R(yaw, pitch, roll), np.array([x, y, z])


SCENES = {
    "indoor": (indoor_scene, indoor_trajectory, 110.0),
    "forest": (forest_scene, forest_trajectory, 110.0),
}


def imu_from_trajectory(traj, t: float, dt: float = 1e-4):
    """True body rates and specific force from the analytic trajectory."""
    R0, p0 = traj(t - dt)
    R1, p1 = traj(t)
    R2, p2 = traj(t + dt)
    a_w = (p0 - 2 * p1 + p2) / dt**2 + np.array([0.0, 0.0, 9.81])
    dR = R1.T @ R2
    w_skew = (dR - dR.T) / (2 * dt)
    gyro = np.array([w_skew[2, 1], w_skew[0, 2], w_skew[1, 0]])
    accel = R1.T @ a_w
    return gyro, accel


# ------------------------------------------------------------------- dust


class DustField:
    """Propwash-lifted particles near the body: constant direction + jitter,
    respawn inside a body-frame box. Deterministic per seed."""

    def __init__(self, n=140, seed=3, direction=(0.75, 0.2, -0.62), speed=1.6, jitter=0.55,
                 box_lo=(-0.4, -0.9, -0.6), box_hi=(1.4, 0.9, 0.5), radius=(0.0006, 0.0028)):
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = np.array(box_lo), np.array(box_hi)
        d = np.array(direction, float)
        self.dir = d / np.linalg.norm(d)
        self.speed, self.jitter = speed, jitter
        self.p = self.rng.uniform(self.lo, self.hi, (n, 3))
        self.r = self.rng.uniform(*radius, n)

    def step(self, dt: float) -> None:
        v = self.dir[None] * self.speed + self.rng.normal(0, self.jitter, self.p.shape)
        self.p += v * dt
        out = np.any((self.p < self.lo) | (self.p > self.hi), axis=1)
        if out.any():
            self.p[out] = self.rng.uniform(self.lo, self.hi, (int(out.sum()), 3))


CONDITIONS = {
    "day":        {"ambient": 0.55, "sun": 0.65, "fog_beta": 0.0, "dust": False, "flood": False},
    "night":      {"ambient": 0.004, "sun": 0.0, "fog_beta": 0.0, "dust": False, "flood": True},
    "day_fog":    {"ambient": 0.55, "sun": 0.45, "fog_beta": 0.16, "dust": False, "flood": False},
    "night_fog":  {"ambient": 0.004, "sun": 0.0, "fog_beta": 0.16, "dust": False, "flood": True},
    "day_dust":   {"ambient": 0.55, "sun": 0.65, "fog_beta": 0.0, "dust": True, "flood": False},
    "night_dust": {"ambient": 0.004, "sun": 0.0, "fog_beta": 0.0, "dust": True, "flood": True},
}


# ------------------------------------------------- exact geometry for map eval


def distance_to_surface(points: np.ndarray, prims: list[dict]) -> np.ndarray:
    """Unsigned distance from (N,3) points to the nearest scene surface — exact."""
    p = np.asarray(points, float)
    best = np.full(len(p), np.inf)
    for pr in prims:
        if pr["kind"] in ("box", "room"):
            c, h = np.array(pr["center"]), np.array(pr["half"])
            q = np.abs(p - c) - h
            outside = np.linalg.norm(np.maximum(q, 0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            d = np.abs(outside + inside) if pr["kind"] == "room" else np.abs(outside + inside)
        elif pr["kind"] == "cylinder":
            cx, cy = pr["center_xy"]
            dr = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - pr["radius"]
            dz = np.maximum(pr["z0"] - p[:, 2], p[:, 2] - pr["z1"])
            d = np.abs(np.where(dz > 0, np.hypot(np.maximum(dr, 0), dz), np.maximum(dr, dz)))
        elif pr["kind"] == "ground":
            d = np.abs(p[:, 2] - pr["z"])
        else:
            continue
        best = np.minimum(best, d)
    return best


def surface_samples(prims: list[dict], n_per_prim: int = 400, seed: int = 0) -> np.ndarray:
    """Points sampled on the scene surfaces (for map completeness)."""
    rng = np.random.default_rng(seed)
    out = []
    for pr in prims:
        if pr["kind"] in ("box", "room"):
            c, h = np.array(pr["center"]), np.array(pr["half"])
            for _ in range(n_per_prim):
                f = rng.integers(0, 6)
                u, v = rng.uniform(-1, 1, 2)
                s = np.array([u, v, 1.0 if f % 2 == 0 else -1.0])
                s = np.roll(s, f // 2)
                out.append(c + s * h)
        elif pr["kind"] == "cylinder":
            cx, cy = pr["center_xy"]
            for _ in range(n_per_prim // 2):
                th = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(pr["z0"], min(pr["z1"], 6.0))     # visible part
                out.append([cx + pr["radius"] * np.cos(th), cy + pr["radius"] * np.sin(th), z])
        elif pr["kind"] == "ground":
            for _ in range(n_per_prim * 4):
                out.append([rng.uniform(-20, 20), rng.uniform(-14, 14), pr["z"]])
    return np.asarray(out)
