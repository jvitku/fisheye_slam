"""Pure-math rig helpers: loading, validation, pod-mount composition.

Deliberately free of Isaac/Pegasus imports so it is unit-testable on the host
(see bench/tests/test_rig_math.py). fisheye_rig.py builds Pegasus sensors on
top of this.

Conventions:
  - mount = {position: [x,y,z] m, rpy_deg: [roll, pitch, yaw] deg} in the
    parent frame (body FLU for top-level mounts, pod frame for pod children).
  - Rotation from rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
  - A rig may define a `pod:` section (self-contained sensor unit — cameras +
    its own FC IMU — mounted as one rigid piece on the drone). Camera and IMU
    mounts inside a pod rig are POD-relative; load_rig() resolves everything
    into body-frame `body_mount` entries.
"""

from __future__ import annotations

import numpy as np
import yaml

IDENTITY_MOUNT = {"position": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}


def euler_to_R(rpy_deg) -> np.ndarray:
    """[roll, pitch, yaw] degrees -> 3x3 rotation matrix (R = Rz @ Ry @ Rx)."""
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R: np.ndarray) -> list[float]:
    """3x3 rotation matrix -> [roll, pitch, yaw] degrees (inverse of euler_to_R)."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.abs(R[2, 0]) < 1.0 - 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: put all z-x rotation into roll
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return [float(np.rad2deg(roll)), float(np.rad2deg(pitch)), float(np.rad2deg(yaw))]


def compose_mount(outer: dict, inner: dict) -> dict:
    """Compose two mounts: inner expressed in outer's frame -> parent frame."""
    R_o = euler_to_R(outer["rpy_deg"])
    p_o = np.asarray(outer["position"], dtype=np.float64)
    R_i = euler_to_R(inner["rpy_deg"])
    p_i = np.asarray(inner["position"], dtype=np.float64)
    return {
        "position": (p_o + R_o @ p_i).tolist(),
        "rpy_deg": R_to_euler(R_o @ R_i),
    }


def load_rig(path: str) -> dict:
    """Load and validate a rig yaml; resolve all mounts into body frame.

    Adds `body_mount` to every camera and to rig['imu'] (composed with
    pod.mount when a pod section is present, pass-through otherwise).
    """
    with open(path) as f:
        rig = yaml.safe_load(f)

    pod_mount = rig.get("pod", {}).get("mount", IDENTITY_MOUNT)

    for cam in rig["cameras"]:
        # kb4 (ideal equidistant, k1..k4=0) renders as an exact f-theta match;
        # pinhole is for global-shutter stereo devices (rigs/oakdpro.yaml).
        if cam["model"] not in ("kb4", "pinhole"):
            raise ValueError(
                f"benchmark rigs must use kb4 or pinhole intrinsics (got "
                f"'{cam['model']}' for '{cam['name']}') — the sim renders "
                f"these exactly"
            )
        cam["depth"] = bool(cam.get("depth", False))
        cam["body_mount"] = compose_mount(pod_mount, cam["mount"])

    imu = rig.get("imu", {})
    imu["body_mount"] = compose_mount(pod_mount, imu.get("mount", IDENTITY_MOUNT))

    if "illuminator" in rig:
        ill = rig["illuminator"]
        ill["body_mount"] = compose_mount(pod_mount, ill.get("mount", IDENTITY_MOUNT))

    return rig
