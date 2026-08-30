"""Rig description for podslam: cameras (lens model + extrinsics), IMU, topics.

Accepts both rig-yaml flavours used across the repo (single source of truth):
  - sim rigs (rigs/pod_*.yaml, rigs/oakdpro.yaml): pod-relative FLU `mount`s,
    rpy 0 = optical axis along +x — resolved with the same math as
    sim/isaac/workspace/rig_math.py (T_mount_optical: x right, y down, z fwd);
  - real rigs (rigs/tumvi_room1.yaml): Kalibr `T_cam_imu` (IMU -> camera
    optical), plus `topic` per camera / IMU.
Lens models come from tools/fisheye (KB4, Double Sphere, EUCM, pinhole).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from tools.fisheye.models import KannalaBrandt4, load_camera  # noqa: E402

from .geometry import T_from_Rt, euler_to_R, inv_T  # noqa: E402

# camera FLU mount frame -> optical frame (columns = optical axes in FLU)
R_MOUNT_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
T_MOUNT_OPTICAL = T_from_Rt(R_MOUNT_OPTICAL, [0, 0, 0])


class Pinhole:
    """Ideal pinhole (no distortion) with the same project/unproject contract."""

    def __init__(self, fx, fy, cx, cy, **_):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.max_theta = np.deg2rad(89.0)

    def project(self, points):
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        z = p[:, 2]
        valid = z > 1e-6
        zs = np.where(valid, z, 1.0)
        return np.stack([self.fx * p[:, 0] / zs + self.cx, self.fy * p[:, 1] / zs + self.cy], 1), valid

    def unproject(self, pixels):
        px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        b = np.stack([(px[:, 0] - self.cx) / self.fx, (px[:, 1] - self.cy) / self.fy, np.ones(len(px))], 1)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        return b, np.ones(len(px), dtype=bool)


@dataclass
class Camera:
    name: str
    model: object                 # KB4 / DS / EUCM / Pinhole with project()/unproject()
    size: tuple                   # (width, height)
    T_imu_cam: np.ndarray         # 4x4, camera optical -> IMU
    topic: str
    depth_topic: str | None = None
    fov_deg: float | None = None  # kb4 rigs: full field of view (image circle)
    fx: float = 0.0
    time_shift_s: float = 0.0     # Kalibr timeshift_cam_imu: t_imu = t_image + shift

    @property
    def T_cam_imu(self) -> np.ndarray:
        return inv_T(self.T_imu_cam)

    def circle_mask(self, shrink: float = 0.97) -> np.ndarray | None:
        """uint8 mask, 255 = valid, inside the image circle of an f-theta lens."""
        if self.fov_deg is None or not isinstance(self.model, KannalaBrandt4):
            return None
        w, h = self.size
        r = self.model.fx * np.deg2rad(self.fov_deg / 2.0) * shrink
        ys, xs = np.mgrid[0:h, 0:w]
        inside = (xs + 0.5 - self.model.cx) ** 2 + (ys + 0.5 - self.model.cy) ** 2 <= r * r
        return (inside * 255).astype(np.uint8)


@dataclass
class Imu:
    topic: str
    rate_hz: float
    gyro_noise_density: float
    gyro_random_walk: float
    accel_noise_density: float
    accel_random_walk: float
    T_body_imu: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class Rig:
    name: str
    cameras: list
    imu: Imu
    ground_truth_topic: str = "/uav1/ground_truth"

    def T_cam_cam(self, i: int, j: int) -> np.ndarray:
        """T_ci_cj: camera j optical -> camera i optical."""
        return self.cameras[i].T_cam_imu @ self.cameras[j].T_imu_cam


def load_rig(path: str) -> Rig:
    raw = yaml.safe_load(open(path))
    if "pods" in raw:
        raise ValueError("podslam consumes a single rig (split composite recordings with bench/split_bag.py)")
    cams_yaml = raw["cameras"]
    imu_yaml = raw.get("imu", {})
    real = all("T_cam_imu" in c for c in cams_yaml)
    prefix = "/" + imu_yaml.get("topic", "/uav1/sensor_pod/imu").strip("/").split("/")[0]
    if real:
        T_body_imu = np.eye(4)
        T_imu_cams = [inv_T(np.asarray(c["T_cam_imu"], dtype=np.float64)) for c in cams_yaml]
    else:
        # sim rig: everything pod-relative; the IMU frame is the pod IMU mount
        imu_mount = imu_yaml.get("mount", {"position": [0, 0, 0], "rpy_deg": [0, 0, 0]})
        T_pod_imu = T_from_Rt(euler_to_R(imu_mount["rpy_deg"]), imu_mount["position"])
        T_body_imu = T_pod_imu
        T_imu_cams = []
        for c in cams_yaml:
            T_pod_cam = T_from_Rt(euler_to_R(c["mount"]["rpy_deg"]), c["mount"]["position"]) @ T_MOUNT_OPTICAL
            T_imu_cams.append(inv_T(T_pod_imu) @ T_pod_cam)
    cameras = []
    for c, T_ic in zip(cams_yaml, T_imu_cams):
        model = Pinhole(**c["intrinsics"]) if c["model"] == "pinhole" else load_camera(c)
        cameras.append(Camera(
            name=c["name"], model=model, size=tuple(int(v) for v in c["resolution"]), T_imu_cam=T_ic,
            time_shift_s=float(c.get("time_shift_s", 0.0)),
            topic=c.get("topic", f"{prefix}/{c['name']}/color/image_raw"),
            depth_topic=c.get("depth_topic", f"{prefix}/{c['name']}/depth/image_raw") if c.get("depth") else None,
            fov_deg=float(c["fov_deg"]) if c["model"] == "kb4" and "fov_deg" in c else (190.0 if c["model"] == "kb4" else None),
            fx=float(c["intrinsics"]["fx"]),
        ))
    imu = Imu(
        topic=imu_yaml.get("topic", f"{prefix}/sensor_pod/imu"),
        rate_hz=float(imu_yaml.get("rate_hz", 200.0)),
        gyro_noise_density=float(imu_yaml.get("gyro_noise_density", 1.6e-4)),
        gyro_random_walk=float(imu_yaml.get("gyro_random_walk", 2.2e-5)),
        accel_noise_density=float(imu_yaml.get("accel_noise_density", 2.8e-3)),
        accel_random_walk=float(imu_yaml.get("accel_random_walk", 8.6e-4)),
        T_body_imu=T_body_imu,
    )
    return Rig(name=raw.get("name", Path(path).stem), cameras=cameras, imu=imu,
               ground_truth_topic=f"{prefix}/ground_truth")
