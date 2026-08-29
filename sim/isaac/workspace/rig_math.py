"""Pure-math rig helpers: loading, validation, pod-mount composition.

Deliberately free of Isaac/Pegasus imports so it is unit-testable on the host
(see bench/tests/test_rig_math.py). fisheye_rig.py builds Pegasus sensors on
top of this; bench/ tools import it through bench/rigdef.py.

Conventions:
  - mount = {position: [x,y,z] m, rpy_deg: [roll, pitch, yaw] deg} in the
    parent frame (body FLU for top-level mounts, pod frame for pod children).
  - Rotation from rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
  - A rig may define a `pod:` section (self-contained sensor unit — cameras +
    its own FC IMU — mounted as one rigid piece on the drone). Camera and IMU
    mounts inside a pod rig are POD-relative; load_rig() resolves everything
    into body-frame `body_mount` entries.
  - A COMPOSITE rig (`pods:` list, e.g. rigs/pod3_oakdpro.yaml) mounts several
    pod rigs side by side on ONE drone so they record the same flight. Each
    member gets a namespace `ns`; its cameras are renamed `<ns>_<cam>` (Pegasus
    derives prim + topic names from the camera name) and its IMU is published
    on `/uavN/<ns>/imu`. bench/split_bag.py undoes the renaming so every
    member's bag matches its own single-rig contract again.

load_rig() output, for every rig kind:
  rig["cameras"]       list, each with body_mount / depth (+ pod_ns,
                       source_name in composites)
  rig["imus"]          list of IMU dicts with topic / body_mount / kind
                       ("pod" = the pod's own FC IMU, simulated by PodIMU;
                       "body" = the drone body IMU) (+ pod_ns, source_topic)
  rig["illuminators"]  list (possibly empty) with body_mount (+ pod_ns)
  rig["pods"]          composites only: [{ns, name, rig, mount, source}],
                       `source` = the member resolved as a stand-alone rig
                       (original camera names/topics) for downstream tools
  rig["imu"] / rig["illuminator"]   single rigs only (backwards compatible)
"""

from __future__ import annotations

import copy
import os
import re

import numpy as np
import yaml

IDENTITY_MOUNT = {"position": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}
CAMERA_MODELS = ("kb4", "pinhole")
GROUND_TRUTH_TOPIC = "ground_truth"   # under the vehicle prefix: /uav1/ground_truth

# Camera FLU mount frame (rpy 0 = optical axis along +x) -> optical frame
# (x right, y down, z forward). Columns = optical axes expressed in the mount
# frame. VERIFY-IN-SIM: image "up" must be world up for a level camera.
R_MOUNT_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


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


def R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [qx, qy, qz, qw] (Shepperd)."""
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_xyzw_to_R(q) -> np.ndarray:
    """Unit quaternion [qx, qy, qz, qw] -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


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


def mount_to_T(mount: dict) -> np.ndarray:
    """mount -> 4x4 homogeneous transform (child frame -> parent frame)."""
    T = np.eye(4)
    T[:3, :3] = euler_to_R(mount["rpy_deg"])
    T[:3, 3] = np.asarray(mount["position"], dtype=np.float64)
    return T


def topic_prefix(topic: str) -> str:
    """'/uav1/sensor_pod/imu' -> '/uav1' (the vehicle namespace)."""
    return "/" + topic.strip("/").split("/")[0]


def vehicle_prefix(rig: dict) -> str:
    return topic_prefix(rig["imus"][0]["topic"])


def camera_topic(rig: dict, cam: dict, stream: str = "color") -> str:
    """Topic of a camera stream as published by the sim: /uavN/<cam>/<stream>/image_raw."""
    return f"{vehicle_prefix(rig)}/{cam['name']}/{stream}/image_raw"


def recorded_topics(rig: dict) -> list[str]:
    """Everything bench/record.sh records for this rig (cams, depth, IMUs, GT)."""
    topics = [camera_topic(rig, c) for c in rig["cameras"]]
    topics += [camera_topic(rig, c, "depth") for c in rig["cameras"] if c["depth"]]
    topics += [imu["topic"] for imu in rig["imus"]]
    topics.append(f"{vehicle_prefix(rig)}/{GROUND_TRUTH_TOPIC}")
    return topics


def imu_sensor_config(imu: dict) -> dict:
    """Rig IMU entry -> LivoxIMU/PodIMU config (mount, rate, noise, topic).

    Noise densities come from the rig yaml so the two IMUs of a composite rig
    (pod PX4-FC vs OAK-D BMI270 class) differ the way the hardware does.
    LivoxIMU orientation is a ZYX list = [yaw, pitch, roll].
    """
    mount = imu["body_mount"]
    r, p, y = mount["rpy_deg"]
    cfg = {
        "position": list(mount["position"]),
        "orientation": [y, p, r],
        "update_rate": float(imu.get("rate_hz", 400)),
        "topic": imu["topic"],
    }
    gyro = {k: imu[f"gyro_{k}"] for k in ("noise_density", "random_walk") if f"gyro_{k}" in imu}
    accel = {k: imu[f"accel_{k}"] for k in ("noise_density", "random_walk") if f"accel_{k}" in imu}
    if gyro:
        cfg["gyroscope"] = gyro
    if accel:
        cfg["accelerometer"] = accel
    return cfg


def _resolve_single(rig: dict) -> dict:
    """Resolve a stand-alone (non-composite) rig dict in place."""
    is_pod = "pod" in rig
    pod_mount = rig.get("pod", {}).get("mount", IDENTITY_MOUNT)

    for cam in rig["cameras"]:
        # kb4 (ideal equidistant, k1..k4=0) renders as an exact f-theta match;
        # pinhole is for global-shutter stereo devices (rigs/oakdpro.yaml).
        if cam["model"] not in CAMERA_MODELS:
            raise ValueError(
                f"benchmark rigs must use kb4 or pinhole intrinsics (got "
                f"'{cam['model']}' for '{cam['name']}') — the sim renders "
                f"these exactly"
            )
        cam["depth"] = bool(cam.get("depth", False))
        cam["body_mount"] = compose_mount(pod_mount, cam["mount"])

    imu = rig.setdefault("imu", {})
    imu["body_mount"] = compose_mount(pod_mount, imu.get("mount", IDENTITY_MOUNT))
    imu["kind"] = "pod" if is_pod else "body"
    rig["imus"] = [imu]

    rig["illuminators"] = []
    if "illuminator" in rig:
        ill = rig["illuminator"]
        ill["body_mount"] = compose_mount(pod_mount, ill.get("mount", IDENTITY_MOUNT))
        rig["illuminators"] = [ill]

    return rig


def _resolve_composite(rig: dict, base_dir: str) -> dict:
    """Resolve a `pods:` composite: load members, namespace them, flatten."""
    cameras, imus, illuminators, pods = [], [], [], []
    seen = set()
    for entry in rig["pods"]:
        ns = entry["ns"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", ns):
            raise ValueError(f"pod ns '{ns}' must be alphanumeric (it prefixes prim/topic names)")
        if ns in seen:
            raise ValueError(f"duplicate pod ns '{ns}'")
        seen.add(ns)

        sub_path = entry["rig"]
        if not os.path.isabs(sub_path):
            sub_path = os.path.join(base_dir, sub_path)
        with open(sub_path) as f:
            sub = yaml.safe_load(f)
        if "pods" in sub:
            raise ValueError(f"{sub_path}: nested composite rigs are not supported")
        if "pod" not in sub:
            raise ValueError(
                f"{sub_path}: composite members must be pod rigs (have a `pod:` "
                f"section) — body-mounted rigs cannot share a drone"
            )
        if "mount" in entry:
            sub["pod"]["mount"] = entry["mount"]
        sub = _resolve_single(sub)
        source = copy.deepcopy(sub)   # the member as its own single-rig contract

        for cam in sub["cameras"]:
            cam["source_name"] = cam["name"]
            cam["name"] = f"{ns}_{cam['name']}"
            cam["pod_ns"] = ns
            cameras.append(cam)

        imu = sub["imu"]
        imu["source_topic"] = imu["topic"]
        imu["topic"] = f"{topic_prefix(imu['topic'])}/{ns}/imu"
        imu["pod_ns"] = ns
        imus.append(imu)

        for ill in sub["illuminators"]:
            ill["pod_ns"] = ns
            illuminators.append(ill)

        pods.append({
            "ns": ns,
            "name": sub.get("name", ns),
            "rig": sub_path,
            "mount": sub["pod"]["mount"],
            "source": source,
        })

    if not pods:
        raise ValueError("composite rig has an empty `pods:` list")
    rig.update(cameras=cameras, imus=imus, illuminators=illuminators,
               pods=pods, composite=True)
    return rig


def load_rig(path: str) -> dict:
    """Load and validate a rig yaml; resolve all mounts into body frame.

    Single rigs: adds `body_mount` to every camera, to rig['imu'] and to
    rig['illuminator'] (composed with pod.mount when a pod section is
    present, pass-through otherwise) and the list views `imus` /
    `illuminators`. Composite rigs (`pods:`): see module docstring.
    """
    with open(path) as f:
        rig = yaml.safe_load(f)
    if "pods" in rig:
        return _resolve_composite(rig, os.path.dirname(os.path.abspath(path)))
    return _resolve_single(rig)


def pod_imu(rig: dict, ns: str | None = None) -> dict:
    """The pod IMU entry of a rig (by ns for composites). Body rigs -> body IMU."""
    if rig.get("composite"):
        if ns is None:
            if len(rig["imus"]) == 1:
                return rig["imus"][0]
            raise ValueError(
                f"composite rig: pick a pod ns from {[p['ns'] for p in rig['pods']]}"
            )
        for imu in rig["imus"]:
            if imu["pod_ns"] == ns:
                return imu
        raise ValueError(f"no pod ns '{ns}' in rig (have {[p['ns'] for p in rig['pods']]})")
    if ns is not None:
        raise ValueError("--pod only applies to composite rigs")
    return rig["imus"][0]
