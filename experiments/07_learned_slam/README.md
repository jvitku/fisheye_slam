# Track F — Hybrid SLAM exploiting the 3-camera + IMU pod (Orin Nano Super)

**Goal:** candidates that *maximally use the pod's geometry* — the triangle is
three stereo pairs sharing one IMU — with a hard deployability gate (inference
on **Orin Nano Super**) and a **commercial-license** gate. General monocular
systems are out of scope for the comparison (they waste the rig).

## Candidates

### F1 — cuVSLAM / Isaac ROS Visual SLAM  ← headline product-path candidate
`NVIDIA-ISAAC-ROS/isaac_ros_visual_slam` (Apache-2.0 wrapper, cuVSLAM binary
under NVIDIA terms — commercial use on NVIDIA hardware, which is our target
anyway).
- **Native fit:** cuVSLAM consumes **up to 16 stereo pairs + IMU** — the
  triangle pod maps directly to 2–3 stereo pairs sharing the pod FC IMU. It
  auto-falls back to IMU (~1 s) then constant-velocity (~0.5 s) when vision
  degrades — robustness behavior we otherwise have to build.
- Built *for* Orin; GPU-accelerated; ROS 2 Humble.
- **Caveats:** closed-source core (no fixes, black-box failures); fisheye
  support to verify — may need rectification of the 190° lenses to ~120°
  virtual pinholes (loses peripheral FOV; quantify the cost in the benchmark).

### F2 — In-house permissive hybrid (XFeat + LightGlue + BSD backend)
The fully-permissive lane: Apache-2.0 learned front-end (XFeat every frame,
LightGlue on keyframes) over a BSD backend (Basalt-derived, or Kimera-VIO
`MIT-SPARK/Kimera-VIO`, BSD, stereo+IMU + mesh output) extended to the 3-cam
pod. Most integration work, zero license risk, full control. This is the
fallback if F1's black box disappoints and the GPL lane stays blocked.

### F3 — AirSLAM (benchmark yardstick; GPL-3)
`sair-lab/AirSLAM` (TRO 2025): hybrid CNN point+line front-end (PLNet) +
classical backend, **stereo + optional IMU**, TensorRT-deployed, **40 Hz on a
Jetson Orin embedded** — and built specifically for illumination robustness,
i.e. our night+IR axis. **GPL-3**: same bucket as OpenVINS — benchmark and
architecture reference, not product code without a licensing decision.

## Deferred / removed from the comparison
- **MASt3R-SLAM** — removed per decision 2026-07-06: CC BY-NC weights
  (commercially unusable) + monocular (doesn't exploit the rig). Clone stays
  in `candidates/` as an offline-tooling / future-reference option only.
- **DPVO** — MIT, but monocular VO; doesn't use the 3-cam+IMU geometry.
  Optional night-axis probe at best; not a matrix row.

## What we test

- [ ] F1: Isaac ROS Visual SLAM on the pod bags — 1/2/3 stereo-pair configs
      (cam0+cam1, +diagonals), pod IMU on/off, fisheye vs rectified inputs
- [ ] F1 vs Track A (OpenVINS 3-cam) on identical day + night+IR bags
- [ ] F3: AirSLAM stereo(+IMU) on the front pair, day vs night+IR
- [ ] F2: XFeat front-end grafted into the best permissive backend (Phase 3)
- [ ] Orin Nano Super: fps / latency / VRAM for F1 and F3

## Notes
- cuVSLAM wants ROS 2; bridge the benchmark ROS1 bags (or record ROS2 bags in
  parallel — the sim is ROS2-native, cheapest path).
- Depth for the map stage from the triangle's 3 baselines: permissive learned
  stereo (HITNet / CREStereo, Apache-2.0 — verify per repo at integration)
  → nvblox (Apache-2.0). See `experiments/06_dense_mapping/`.
