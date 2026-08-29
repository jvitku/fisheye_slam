"""Benchmark scene lighting + sensor-pod IR illuminator (Isaac Sim / UsdLux).

Lighting is a first-class benchmark axis (SIM_LIGHTING env):

  day    scene's own lighting untouched (baseline).
  night  every existing UsdLux light dimmed to ~0 + a faint ambient dome
         ("moonlight"). The pod's IR illuminator becomes the dominant source.
  half   night treatment + a large rect light over ONE half of the scene:
         the trajectory crosses a lit->dark boundary mid-flight (the
         day-to-darkness transition case).

The pod illuminator models an 850 nm IR flood LED at the camera-triangle
center, as seen by NoIR cameras: RTX lights are spectrally RGB, so NIR is
modeled as white light and NoIR imaging as grayscale conversion of the
rendered RGB (do the mono conversion at the candidate input / offline —
Isaac renders color). Crucially the illuminator is a REAL ray-traced light
rigidly attached to the drone: it moves with the vehicle and casts moving
shadows into the scene — precisely the failure mode that makes night VIO
hard (shadow edges are non-static "features").

All intensities are scaffold values — tune at GPU bring-up (VERIFY-IN-SIM).
"""

import carb
from pxr import Gf, Sdf, UsdGeom, UsdLux

BENCH_SCOPE = "/World/BenchLighting"


def _iter_scene_lights(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.LightAPI) if hasattr(UsdLux, "LightAPI") else prim.HasAPI(UsdLux.LightAPI):
            yield prim
        elif prim.GetTypeName() in (
            "DomeLight", "DistantLight", "SphereLight", "RectLight",
            "DiskLight", "CylinderLight",
        ):
            yield prim


def _dim_existing_lights(stage, factor):
    """Scale every pre-existing scene light's intensity by `factor`."""
    n = 0
    for prim in _iter_scene_lights(stage):
        if str(prim.GetPath()).startswith(BENCH_SCOPE):
            continue
        for attr_name in ("inputs:intensity", "intensity"):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                val = attr.Get()
                if val is not None:
                    attr.Set(float(val) * factor)
                    n += 1
                break
    carb.log_warn(f"lighting: dimmed {n} scene lights by x{factor}")


def setup_lighting(stage, mode: str):
    """Apply the benchmark lighting mode. Call AFTER the environment loads."""
    mode = (mode or "day").lower()
    if mode == "day":
        carb.log_warn("lighting: mode=day (scene default)")
        return

    _dim_existing_lights(stage, 0.0)
    UsdGeom.Scope.Define(stage, BENCH_SCOPE)

    # Faint ambient so "night" is not pitch black (moon/starlight class).
    dome = UsdLux.DomeLight.Define(stage, f"{BENCH_SCOPE}/night_ambient")
    dome.CreateIntensityAttr(3.0)
    dome.CreateColorAttr(Gf.Vec3f(0.6, 0.7, 1.0))

    if mode == "half":
        # A large rect light hovering over the +y half of the scene: the -y
        # half stays night-dark. Size/height are Curved-Gridroom-scale
        # scaffold values (VERIFY-IN-SIM per environment).
        rect = UsdLux.RectLight.Define(stage, f"{BENCH_SCOPE}/half_day")
        rect.CreateWidthAttr(12.0)
        rect.CreateHeightAttr(12.0)
        rect.CreateIntensityAttr(8000.0)
        xf = UsdGeom.Xformable(rect.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 8.0, 6.0))
        # face downward (rect lights emit along -z)
        carb.log_warn("lighting: mode=half (lit +y half, dark -y half)")
    else:
        carb.log_warn("lighting: mode=night (ambient only; pod IR dominates)")


def attach_pod_ir_light(stage, body_path: str, rig) -> int:
    """Attach every pod IR flood illuminator of the rig to the vehicle body.

    One shadow-casting cone light per rig['illuminators'] entry (pod-relative
    mounts already resolved to body frame by rig_math.load_rig), emitting
    along that pod's camera boresight. Composite rigs get one light per
    member pod (prim <ns>_ir_light): with both devices on the drone both
    floods are on, exactly as in hardware. Returns the number of lights.
    """
    n = 0
    for ill in rig.get("illuminators", []):
        name = f"{ill['pod_ns']}_ir_light" if "pod_ns" in ill else "pod_ir_light"
        mount = ill["body_mount"]
        light = UsdLux.SphereLight.Define(stage, f"{body_path}/{name}")
        light.CreateRadiusAttr(0.005)
        light.CreateIntensityAttr(float(ill.get("intensity", 30000.0)))
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))  # NIR modeled as white
        light.CreateNormalizeAttr(True)

        # Cone shaping: flood beam along the pod boresight.
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(float(ill.get("cone_half_angle_deg", 60.0)))
        shaping.CreateShapingConeSoftnessAttr(0.3)

        xf = UsdGeom.Xformable(light.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*mount["position"]))
        # Shaped lights emit along -z; pitch -90 maps -z -> body +x, composed with
        # the mount's own rotation. VERIFY-IN-SIM on first night-mode bring-up.
        r, p, y = mount["rpy_deg"]
        xf.AddRotateZYXOp().Set(Gf.Vec3f(r, p - 90.0, y))

        carb.log_warn(
            f"lighting: IR illuminator '{name}' at {mount['position']} "
            f"(cone {ill.get('cone_half_angle_deg', 60.0)} deg, "
            f"intensity {ill.get('intensity', 30000.0)})"
        )
        n += 1
    return n
