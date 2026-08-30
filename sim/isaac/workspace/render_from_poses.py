"""Offline render pass: cameras of a rig along a recorded trajectory, one at a time.

"Physics once, render offline" (bench/README.md). The flight was flown by
bench_drone.py with only IMU + ground truth recorded; this script re-creates
the rig on a KINEMATIC Xform (no vehicle, no PX4, no physics), steps it through
the sampled body poses (bench/pose_sampler.py) and renders each camera in
turn, so VRAM = renderer + ONE render product — it fits an 8 GB GPU — and
every rig sees byte-identical motion. Lighting (SIM_LIGHTING) and the pod IR
floods attach exactly as in the live sim, so day AND night can be rendered
from the same flight.

Env (compose passes them through):
  RIG_CONFIG      /rigs/<rig>.yaml            cameras/illuminators (composite ok)
  POSES           /render/<name>/poses.txt    TUM body poses at the frame rate
  RENDER_OUT      /render/<name>/frames       output root
  SIM_LIGHTING    day|night|half              lighting.py
  POD_IR_LIGHT    auto|on|off
  SIM_ENVIRONMENT environment name/path (bench_drone.py convention)
  RENDER_CAMERAS  "all" or comma list of camera names (split work across runs)
  RENDER_SETTLE   render calls per pose before capture (default 4; RTX
                  temporal accumulation)

Output per camera: <RENDER_OUT>/<cam>/frames.csv (index,t_ns), <index>.png
(RGB), <index>.depth.npy (float32 m) for cameras with `depth: true`.
bench/frames2bag.py turns this + the flight bag into the contract bag.

VERIFY-IN-SIM (never run — needs the Isaac image): the isaacsim.* imports on
5.1, the USD-camera-to-FLU orientation (R_FLU_USDCAM below) and that the
f-theta attributes survive Camera.initialize().
"""

import csv
import os
import sys
import time

import numpy as np

_headless = "--headless" in sys.argv or os.environ.get("HEADLESS", "true").lower() == "true"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": _headless, "renderer": "RayTracedLighting"})

import carb
import omni.usd
from pxr import Gf, Sdf, UsdGeom

try:                                      # Isaac Sim >= 4.5 namespaces
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim
    from isaacsim.core.utils.prims import delete_prim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.sensors.camera import Camera
except ImportError:                       # older layout (what bench_drone.py uses)
    from omni.isaac.core.world import World
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.utils.prims import delete_prim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.sensor import Camera

from pegasus.simulator.params import SIMULATION_ENVIRONMENTS

from fisheye_rig import apply_fisheye_projections
from lighting import attach_pod_ir_light, setup_lighting
from rig_math import R_to_quat_xyzw, euler_to_R, load_rig

RIG_PATH = "/World/rig"
RIG_CONFIG = os.environ.get("RIG_CONFIG", "/rigs/pod3_oakdpro.yaml")
POSES = os.environ.get("POSES", "/render/poses.txt")
RENDER_OUT = os.environ.get("RENDER_OUT", "/render/frames")
SIM_LIGHTING = os.environ.get("SIM_LIGHTING", "day")
POD_IR_LIGHT = os.environ.get("POD_IR_LIGHT", "auto")
RENDER_CAMERAS = os.environ.get("RENDER_CAMERAS", "all")
RENDER_SETTLE = int(os.environ.get("RENDER_SETTLE", "4"))

# USD cameras look along -Z with +Y up (+X right). Rig mounts are FLU with
# rpy 0 = optical axis along +x. Columns = USD camera axes expressed in FLU.
R_FLU_USDCAM = np.array([
    [0.0, 0.0, -1.0],   # usd +x (right) = flu -y ; usd +y (up) = flu +z ; usd +z (back) = flu -x
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
]).T


def _quat_wxyz(R):
    x, y, z, w = R_to_quat_xyzw(R)
    return np.array([w, x, y, z])


def load_environment():
    env = os.environ.get("SIM_ENVIRONMENT", "Curved Gridroom")
    path = env if (os.path.exists(env) or env.startswith(("http", "omniverse://"))) \
        else SIMULATION_ENVIRONMENTS[env]
    add_reference_to_stage(usd_path=path, prim_path="/World/environment")
    carb.log_warn(f"render: environment {env}")


def read_poses(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split()[:8]])
    return np.asarray(rows)


def place_camera(stage, cam):
    """Define the USD camera prim under the rig Xform at its FLU body mount."""
    path = f"{RIG_PATH}/{cam['name']}"
    prim = UsdGeom.Camera.Define(stage, path).GetPrim()
    mount = cam["body_mount"]
    R = euler_to_R(mount["rpy_deg"]) @ R_FLU_USDCAM
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*mount["position"]))
    w, x, y, z = _quat_wxyz(R)
    xf.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    return path


def render_camera(world, stage, rig, cam, poses, rig_xform):
    name = cam["name"]
    out_dir = os.path.join(RENDER_OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    width, height = cam["resolution"]

    path = place_camera(stage, cam)
    camera = Camera(prim_path=path, resolution=(int(width), int(height)))
    camera.initialize()
    if cam["depth"]:
        camera.add_distance_to_image_plane_to_frame()
    # exact projection (f-theta for kb4, exact-fx pinhole) — re-applied after
    # initialize() in case the Camera wrapper touched the prim
    apply_fisheye_projections(stage, RIG_PATH, {"cameras": [cam]})

    from PIL import Image
    t_start = time.time()
    with open(os.path.join(out_dir, "frames.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "t_ns"])
        for k, row in enumerate(poses):
            t_ns = int(round(row[0] * 1e9))
            qx, qy, qz, qw = row[4:8]
            rig_xform.set_world_pose(position=row[1:4], orientation=np.array([qw, qx, qy, qz]))
            for _ in range(RENDER_SETTLE):
                world.render()
            rgba = camera.get_rgba()
            if rgba is None or rgba.size == 0:
                carb.log_error(f"render: {name} frame {k}: empty image")
                continue
            Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(
                os.path.join(out_dir, f"{k:06d}.png"))
            if cam["depth"]:
                depth = np.asarray(camera.get_depth(), dtype=np.float32)
                np.save(os.path.join(out_dir, f"{k:06d}.depth.npy"), depth)
            writer.writerow([k, t_ns])
            if k % 100 == 0:
                carb.log_warn(f"render: {name} {k}/{len(poses)} ({time.time() - t_start:.0f}s)")
    delete_prim(path)
    carb.log_warn(f"render: {name} done -> {out_dir}")


def main():
    rig = load_rig(RIG_CONFIG)
    poses = read_poses(POSES)
    wanted = None if RENDER_CAMERAS == "all" else {n.strip() for n in RENDER_CAMERAS.split(",")}
    cams = [c for c in rig["cameras"] if wanted is None or c["name"] in wanted]
    carb.log_warn(f"render: rig {rig.get('name')} cameras {[c['name'] for c in cams]}, "
                  f"{len(poses)} poses, lighting {SIM_LIGHTING}")

    world = World(stage_units_in_meters=1.0)
    load_environment()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, RIG_PATH)
    rig_xform = XFormPrim(RIG_PATH)
    world.reset()

    setup_lighting(stage, SIM_LIGHTING)
    ir_on = POD_IR_LIGHT == "on" or (POD_IR_LIGHT == "auto" and SIM_LIGHTING != "day")
    if ir_on:
        attach_pod_ir_light(stage, RIG_PATH, rig)

    for cam in cams:
        render_camera(world, stage, rig, cam, poses, rig_xform)
    carb.log_warn("render: all cameras done")
    simulation_app.close()


if __name__ == "__main__":
    main()
