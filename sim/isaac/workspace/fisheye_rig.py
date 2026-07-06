"""Fisheye camera rig support for the Isaac/Pegasus benchmark drone.

Loads a rig definition from rigs/*.yaml (mounted at /rigs in the container) and

  1. builds Pegasus MonocularCamera sensors for each camera entry (pod-relative
     mounts already resolved to body frame by rig_math.load_rig), and
  2. after the vehicle is spawned, rewrites each camera prim to an f-theta
     (fisheyePolynomial) projection matching the rig's kb4 intrinsics.

The benchmark rigs use ideal equidistant intrinsics (kb4 with k1..k4 = 0),
which maps exactly onto Isaac's f-theta model with polyA=0, polyB=1/fx,
polyC..E=0 — so the ground-truth calibration handed to the candidates is
exact and calibration error is eliminated as a benchmark variable.

Frame conventions:
  - rig mounts: position [m] + rpy_deg = [roll, pitch, yaw] in the parent FLU
    frame, rpy [0,0,0] = optical axis along +x (see rig_math.py).
  - Pegasus sensor `orientation` lists are consumed as
    Rotation.from_euler("ZYX", ori) — i.e. ori = [yaw, pitch, roll] — see
    livox_imu.py and the px4_drone.py examples ("forward" = [0,0,180],
    "down" = [0,90,180]). A forward camera needs a 180-deg ROLL flip
    (last element), which we apply as the optical-convention correction.
    VERIFY-IN-SIM: confirm image orientation for yawed cameras on first
    bring-up.
"""

import carb

from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera

from rig_math import load_rig  # re-exported for bench_drone.py

__all__ = ["load_rig", "make_cameras", "apply_fisheye_projections"]

# Pegasus/Isaac camera optical-convention correction (see module docstring).
_ROLL_CORRECTION_DEG = 180.0


def _pegasus_orientation(rpy_deg):
    """[roll, pitch, yaw] body mount -> Pegasus ZYX list [yaw, pitch, roll+180]."""
    roll, pitch, yaw = rpy_deg
    return [yaw, pitch, roll + _ROLL_CORRECTION_DEG]


def make_cameras(rig):
    """Build Pegasus MonocularCamera sensors from a rig loaded by load_rig()."""
    cameras = []
    for cam in rig["cameras"]:
        mount = cam["body_mount"]
        cameras.append(
            MonocularCamera(cam["name"], config={
                "position": list(mount["position"]),
                "orientation": _pegasus_orientation(mount["rpy_deg"]),
                "resolution": tuple(cam["resolution"]),
                "frequency": cam["rate_hz"],
                # Placeholder pinhole FOV; the prim is rewritten to f-theta by
                # apply_fisheye_projections() after spawn.
                "diagonal_fov": 120.0,
                "depth": False,
            })
        )
    return cameras


def apply_fisheye_projections(stage, body_path, rig):
    """Rewrite spawned camera prims to f-theta fisheye projection.

    Mirrors the post-spawn attribute-override pattern used for the lidar in
    px4_drone.py (apply_mid360_fov).
    """
    for cam in rig["cameras"]:
        prim_path = f"{body_path}/{cam['name']}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            carb.log_error(f"fisheye_rig: camera prim not found at {prim_path}")
            continue

        intr = cam["intrinsics"]
        width, height = cam["resolution"]

        def _set(name, val, p=prim):
            attr = p.GetAttribute(name)
            if attr and attr.IsValid():
                attr.Set(val)
            else:
                # Attribute may need creation on some Isaac versions
                carb.log_warn(f"fisheye_rig: attribute {name} missing on {p.GetPath()}")

        _set("cameraProjectionType", "fisheyePolynomial")
        _set("fthetaWidth", float(width))
        _set("fthetaHeight", float(height))
        _set("fthetaCx", float(intr["cx"]))
        _set("fthetaCy", float(intr["cy"]))
        _set("fthetaMaxFov", float(cam.get("fov_deg", 190.0)))
        # f-theta: angle_rad = A + B*r + C*r^2 + D*r^3 + E*r^4  (r in pixels)
        # ideal equidistant kb4: theta = r / fx
        _set("fthetaPolyA", 0.0)
        _set("fthetaPolyB", 1.0 / float(intr["fx"]))
        _set("fthetaPolyC", 0.0)
        _set("fthetaPolyD", 0.0)
        _set("fthetaPolyE", 0.0)

        carb.log_info(
            f"fisheye_rig: {cam['name']} -> f-theta fov={cam.get('fov_deg', 190.0)} "
            f"fx={intr['fx']} ({width}x{height})"
        )
