"""Host-side import shim for the sim's rig math (sim/isaac/workspace/rig_math.py).

rig_math is Isaac-free on purpose; bench tools (record topics, bag split,
ground-truth frames, candidate config generation) reuse it through this shim
so the rig yaml stays the single source of truth for sim and bench alike.
"""

import sys
from pathlib import Path

_WS = Path(__file__).resolve().parents[1] / "sim" / "isaac" / "workspace"
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

from rig_math import (  # noqa: E402
    R_MOUNT_OPTICAL,
    R_to_quat_xyzw,
    camera_topic,
    euler_to_R,
    imu_sensor_config,
    load_rig,
    mount_to_T,
    pod_imu,
    quat_xyzw_to_R,
    recorded_topics,
    vehicle_prefix,
)

__all__ = [
    "R_MOUNT_OPTICAL", "R_to_quat_xyzw", "camera_topic", "euler_to_R",
    "imu_sensor_config", "load_rig", "mount_to_T", "pod_imu", "quat_xyzw_to_R",
    "recorded_topics", "vehicle_prefix",
]
