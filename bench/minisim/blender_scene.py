"""Blender realism lane: render a minisim sequence with Cycles/OptiX.

Same scenes/trajectories/conditions as the raycaster (bench.minisim.scene is
the single source of truth), but with a physically-based renderer: real light
falloff and shadows from the rig's IR floods (spot lights), volumetric fog,
noise-node procedural materials, native equidistant fisheye (= our KB4 with
k1..k4 = 0) and true pinhole cameras.  Dust splats stay raycaster-only for now.

Run inside Blender (headless):
    ~/tools/blender-4.2.0-linux-x64/blender -b --python bench/minisim/blender_scene.py -- \
        rigs/skydio3.yaml indoor night out_dir --start 0 --end 200 [--traj f.tum] [--samples 16]

Blender's bundled Python needs pyyaml once:
    ~/tools/blender-4.2.0-linux-x64/4.2/python/bin/python3.11 -m pip install pyyaml
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import bpy                                   # noqa: E402
import yaml                                  # noqa: E402

from bench.minisim import scene as S         # noqa: E402
from podslam.geometry import euler_to_R      # noqa: E402
from podslam.rig import T_MOUNT_OPTICAL      # noqa: E402

R_OPT_BCAM = np.diag([1.0, -1.0, -1.0])      # optical (+z fwd, +y down) -> blender cam (-z fwd, +y up)


def make_material(name, tex, night):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    noise = m.node_tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0 / max(tex.get("octaves", ((0.4, 0.25),))[0][0], 0.02)
    noise.inputs["Detail"].default_value = 6.0
    ramp = m.node_tree.nodes.new("ShaderNodeValToRGB")
    base = tex.get("base", 0.5)
    amp = sum(a for _, a in tex.get("octaves", ((0.4, 0.25),)))
    ramp.color_ramp.elements[0].color = (max(base - amp, 0.02),) * 3 + (1.0,)
    ramp.color_ramp.elements[1].color = (min(base + amp, 1.0),) * 3 + (1.0,)
    tc = m.node_tree.nodes.new("ShaderNodeTexCoord")
    m.node_tree.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    m.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    m.node_tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def build_scene(prims, cond):
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for i, pr in enumerate(prims):
        mat = make_material(f"m{i}", pr["tex"], cond["flood"])
        if pr["kind"] in ("box", "room"):
            bpy.ops.mesh.primitive_cube_add(location=pr["center"])
            ob = bpy.context.object
            ob.scale = pr["half"]
            if pr["kind"] == "room":
                mod = ob.modifiers.new("flip", "SOLIDIFY")
                mod.thickness = 0.0
                ob.display_type = 'TEXTURED'
                # normals point outward; for the interior view flip them
                bpy.context.view_layer.objects.active = ob
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.flip_normals()
                bpy.ops.object.mode_set(mode="OBJECT")
        elif pr["kind"] == "cylinder":
            cx, cy = pr["center_xy"]
            h = pr["z1"] - pr["z0"]
            bpy.ops.mesh.primitive_cylinder_add(radius=pr["radius"], depth=h,
                                                location=(cx, cy, pr["z0"] + h / 2))
            ob = bpy.context.object
        elif pr["kind"] == "ground":
            bpy.ops.mesh.primitive_plane_add(size=120, location=(0, 0, pr["z"]))
            ob = bpy.context.object
        else:
            continue
        ob.data.materials.append(mat)

    w = bpy.data.worlds["World"] if "World" in bpy.data.worlds else bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = cond["ambient"]
    bg.inputs["Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    if cond["sun"] > 0:
        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.object
        sun.data.energy = cond["sun"] * 3.0
        d = S.SUN_DIR if hasattr(S, "SUN_DIR") else np.array([0.35, 0.25, -0.9])
        sun.rotation_mode = "QUATERNION"
        import mathutils
        sun.rotation_quaternion = mathutils.Vector((0, 0, -1)).rotation_difference(
            mathutils.Vector((-d[0], -d[1], -d[2])))
    if cond["fog_beta"] > 0:
        vol = w.node_tree.nodes.new("ShaderNodeVolumePrincipled")
        vol.inputs["Density"].default_value = cond["fog_beta"]
        out = w.node_tree.nodes["World Output"]
        w.node_tree.links.new(vol.outputs["Volume"], out.inputs["Volume"])


def add_floods(raw, cond):
    lights = []
    if not cond["flood"]:
        return lights
    for il in raw.get("illuminators", []):
        m = il["mount"]
        bpy.ops.object.light_add(type="SPOT")
        sp = bpy.context.object
        sp.data.energy = float(il.get("intensity", 30000)) * 0.01
        sp.data.spot_size = math.radians(2 * il.get("cone_half_angle_deg", 60))
        sp.data.spot_blend = 0.5
        sp.data.shadow_soft_size = 0.02
        fd = euler_to_R(m["rpy_deg"]) @ np.array([1.0, 0.0, 0.0])
        lights.append((sp, np.array(m["position"], float), fd))
    return lights


def set_pose(ob, T):
    import mathutils
    ob.matrix_world = mathutils.Matrix([list(T[i]) for i in range(4)])


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("rig"); ap.add_argument("scene", choices=list(S.SCENES))
    ap.add_argument("condition", choices=list(S.CONDITIONS)); ap.add_argument("out")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--fps", type=float, default=20.0); ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--traj", default=None); ap.add_argument("--samples", type=int, default=16)
    a = ap.parse_args(argv)

    raw = yaml.safe_load(open(REPO / a.rig))
    cond = S.CONDITIONS[a.condition]
    scene_fn, traj_fn, default_dur = S.SCENES[a.scene]
    if a.traj:
        from bench.minisim.render import TumTrajectory
        traj = TumTrajectory(a.traj)
        dur = a.duration or traj.dur
    else:
        traj, dur = traj_fn, (a.duration or default_dur)
    n_frames = int(dur * a.fps)
    end = a.end or n_frames

    build_scene(scene_fn(), cond)
    floods = add_floods(raw, cond)

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type in ("OPTIX", "CUDA")
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "BW"

    # cameras
    cams = []
    T_pod_imu = np.eye(4)
    im = raw.get("imu", {}).get("mount")
    if im:
        T_pod_imu[:3, :3] = euler_to_R(im["rpy_deg"]); T_pod_imu[:3, 3] = im["position"]
    for c in raw["cameras"]:
        cam_data = bpy.data.cameras.new(c["name"])
        w_px, h_px = c["resolution"]
        if c["model"] == "kb4":
            cam_data.type = "PANO"
            cam_data.panorama_type = "FISHEYE_EQUIDISTANT"
            cam_data.fisheye_fov = math.radians(c.get("fov_deg", 190.0))
        else:
            cam_data.type = "PERSP"
            cam_data.lens_unit = "FOV"
            cam_data.angle = 2 * math.atan(w_px / (2 * c["intrinsics"]["fx"]))
        ob = bpy.data.objects.new(c["name"], cam_data)
        bpy.context.collection.objects.link(ob)
        T_pod_cam = np.eye(4)
        T_pod_cam[:3, :3] = euler_to_R(c["mount"]["rpy_deg"]); T_pod_cam[:3, 3] = c["mount"]["position"]
        T_pod_cam = T_pod_cam @ T_MOUNT_OPTICAL
        T_pod_cam[:3, :3] = T_pod_cam[:3, :3] @ R_OPT_BCAM
        cams.append((c, ob, T_pod_cam, (w_px, h_px)))
        (Path(a.out) / c["name"]).mkdir(parents=True, exist_ok=True)

    for k in range(a.start, end):
        t = k / a.fps
        R_wp, p_wp = traj(t)
        T_wp = np.eye(4); T_wp[:3, :3] = R_wp; T_wp[:3, 3] = p_wp
        for sp, fp, fd in floods:
            Tf = np.eye(4)
            z = -fd                                        # blender lights emit along -Z
            x = np.cross([0, 0, 1.0], z); x = x / (np.linalg.norm(x) + 1e-9) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
            y = np.cross(z, x)
            Tf[:3, :3] = np.stack([x, y, z], 1); Tf[:3, 3] = fp
            set_pose(sp, T_wp @ Tf)
        for c, ob, T_pod_cam, (w_px, h_px) in cams:
            out_png = Path(a.out) / c["name"] / f"{k:06d}.png"
            if out_png.exists():
                continue
            sc.camera = ob
            sc.render.resolution_x, sc.render.resolution_y = w_px, h_px
            set_pose(ob, T_wp @ T_pod_cam)
            sc.render.filepath = str(out_png)
            bpy.ops.render.render(write_still=True)
        if k % 20 == 0:
            print(f"blender frame {k}/{end}", flush=True)
    print("blender render done")


main()
