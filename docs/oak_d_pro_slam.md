# OAK-D Pro SLAM — robust real-time state estimation + dense mapping

*2026-07-10 · research report + implementation plan · experiments/08_oakdpro_slam*

Goal: the most robust real-time stack on a **Luxonis OAK-D Pro** producing
(1) a real-time 6-DoF state estimate and (2) a dense 3D map (TSDF / OctoMap)
from the depth stream — i.e. a camera-native alternative to FAST-LIO+OctoMap.
Every option below is implemented for testing in `experiments/08_oakdpro_slam/`
against **one bag contract** that is produced identically by Isaac Sim
(`rigs/oakdpro.yaml`) and by the real device
(`experiments/08_oakdpro_slam/hw/record_oak.py`) — same runners, both worlds.

## TL;DR

| # | Stack | Odometry | Dense map | License | Platform | Why |
|---|---|---|---|---|---|---|
| 1 | **Spectacular AI SDK** | proprietary VISLAM (HybVIO lineage) | SDK Mapping API → .ply / feed TSDF | free **non-commercial**; paid product | x86 / Jetson, CPU | most robust out of the box, built *for* OAK-D |
| 2 | **cuVSLAM + nvblox** | GPU stereo-inertial VIO | nvblox GPU TSDF/ESDF/mesh | free, closed-source (NVIDIA) | Jetson Orin / x86+GPU | the FAST-LIO+OctoMap analog; Track F1 product lane |
| 3 | **OpenVINS + RTAB-Map** | OpenVINS MSCKF (GPL yardstick) | RTAB-Map → **OctoMap** + cloud + 2D grid | open source | CPU (Pi-5-class up) | fully open; our Round-1 data says OpenVINS is the night-robust core |

Recommendation: prototype **#2 first** (it doubles as the Track F1 eval with
real hardware), run **#1 alongside as the robustness yardstick** (license only
if it wins), keep **#3 as the open fallback** — per
[EVAL_2026-07-06](EVAL_2026-07-06_day_night_hybrid.md), OpenVINS is the most
*predictably* robust of the classical cores at night.

## 1. Why the OAK-D Pro is a friendlier target than the pod

Compared to `rigs/pod_*.yaml` (unsynced rolling-shutter fisheyes), the OAK-D
Pro is an easy SLAM rig: hardware-synchronized **global-shutter** stereo
(OV9282 1280×800, 7.5 cm baseline, HFOV ≈ 80°), on-device stereo depth, an
onboard IMU (BMI270/BNO086), and *both* active illuminators:

- **IR laser dot projector** — dense, reliable depth on textureless
  surfaces (active stereo). The dots move with the camera, so they **poison
  feature tracking**; DepthAI supports [per-frame alternation of projector and
  flood LED](https://luxonis-depthai-python.readthedocs-hosted.com/en/develop/samples/MonoCamera/mono_preview_alternate_pro/):
  dot frames → depth, flood frames → VIO features, each at half FPS.
- **IR flood LED** — night feature tracking. This is the same concept as the
  pod's 850 nm flood illuminator, factory-integrated; the benchmark's lighting
  axis (`SIM_LIGHTING=day|night|half`) applies unchanged.

The known weak point is the **IMU path through depthai-ros**: batched
delivery, timestamp discrepancies, ~150–160 Hz effective vs 400 Hz nominal
([depthai-ros #461](https://github.com/luxonis/depthai-ros/issues/461)).
Naive VINS-Fusion-on-OAK attempts drift for exactly this reason
([VINS-Fusion #241](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/issues/241)).
Options that own the OAK IMU handling (Spectacular AI, our own recorder using
device timestamps) sidestep it; ROS-live pipelines must use device timestamps
and online time-offset calibration (OpenVINS has it).

Luxonis' [officially recommended VIO/SLAM paths](https://docs.luxonis.com/overview/toplevel-features/vio_and_slam)
match this report: Spectacular AI, PyCuVSLAM, RTAB-Map (ROS 2), and a native
Basalt+RTAB-Map DepthAI v3 example.

## 2. The options

### Option 1 — Spectacular AI SDK (most robust out of the box)

[Spectacular AI](https://spectacularai.github.io/docs/sdk/wrappers/oak.html)
was built for OAK-D (close Luxonis partnership); its engine is the successor
of [HybVIO](https://github.com/SpectacularAI/HybVIO)
([WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Seiskari_HybVIO_Pushing_the_Limits_of_Real-Time_Visual-Inertial_Odometry_WACV_2022_paper.pdf)),
which posted the best real-time EuRoC results of its generation on an
embedded CPU. It absorbs the OAK IMU sync/calibration mess internally, runs
on Jetson and x86, ships a ROS 2 wrapper, and is actively maintained
([v1.52.1, 2026-06](https://pypi.org/project/spectacularai/)). The Mapping
API emits keyframe poses + point clouds (→ .ply, or feed any TSDF/OctoMap
backend), with loop closure and relocalization.

**Catches.** Free for **non-commercial use only** (Track F commercial-license
matrix applies); closed box — its degradation *signature* (works-harder vs
silently-does-less, see EVAL round 1) can only be measured from outside.

**Here:** `run_spectacularai.sh` — bag → SAI recording format (`bag2sai.py`)
→ Replay API (`sai_replay.py`) → TUM trajectory + keyframe clouds. On real HW
the SDK can also run **live** on the device for the lowest-latency estimate.

### Option 2 — cuVSLAM + nvblox (the FAST-LIO+OctoMap analog)

- [cuVSLAM / Isaac ROS Visual SLAM](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)
  — GPU stereo-inertial odometry, few-ms/frame on Orin, loop closure,
  camera-agnostic (rectified stereo + IMU). Luxonis explicitly lists
  **[PyCuVSLAM](https://github.com/NVlabs/PyCuVSLAM)** (ROS-free Python API)
  as a supported OAK path — it also sidesteps the depthai-ros IMU plumbing.
- [nvblox](https://github.com/nvidia-isaac/nvblox) — GPU TSDF/ESDF/mesh in
  real time on Jetson from depth + pose (any pose source), outputs the ESDF a
  planner wants and a Nav2 costmap. Dramatically faster than the CPU lineage
  (voxblox/OctoMap); [standalone ROS 2 ports](https://github.com/donceykong/nvblox_ros2)
  exist outside the Isaac container ecosystem.

**Catches.** Needs an NVIDIA GPU (Orin Nano Super ✓, Pi 5 ✗). Closed-source
odometry; night behavior is the Track F1 unknown — but flood-lit
global-shutter frames are a much friendlier night input than the pod proxy.

**Here:** `run_cuvslam.sh` — bag → PyCuVSLAM (`cuvslam_track.py`) → TUM
trajectory; depth + poses → nvblox fuser → mesh/ESDF (Track E interface,
`experiments/06_dense_mapping`).

### Option 3 — OpenVINS + RTAB-Map (fully open source)

[RTAB-Map](http://introlab.github.io/rtabmap/) natively outputs **OctoMap**,
dense cloud, and 2D occupancy grid, with loop closure, and has a native
depthai driver (incl. on-device SuperPoint) for live HW use. Its built-in
odometry is its weak point → feed external odometry. Round-1 data picks the
source: **OpenVINS was night-immune (ATE 7.4→7.6 cm) while Basalt silently
degraded 2.6×**. OpenVINS' online time-offset calibration also absorbs
residual OAK IMU timing error. All CPU — the only lane that reaches
Pi-5-class hardware.

**Here:** `run_rtabmap.sh` — ROS1: bag replay → (odometry: RTAB-Map
stereo odom, or OpenVINS via `3dfe/openvins` on the shared roscore) →
`rtabmap` → OctoMap `.bt` + cloud `.ply` + TUM poses.

## 3. How robust can it get? (honest expectations)

- **State estimate**: 0.3–1 % of trajectory drift-class VIO (5–10 cm on
  ~100 m indoor EuRoC/TUM-VI-scale runs for all three), pose at camera rate,
  IMU-rate propagation available. Flood LED + global shutter makes night
  genuinely viable — Round 1 showed KLT-style VIO barely degrades under
  flood-lit darkness.
- **Dense map**: limiter is the 7.5 cm baseline — depth error grows
  quadratically; good to ~4–5 m, usable to ~8–10 m. The dot projector makes
  depth dense on blank walls/floors where passive stereo returns holes.
- **vs FAST-LIO+OctoMap**: won't match LiDAR range (30–100 m), lighting
  independence beyond flood reach (~5–8 m outdoors at night), or long
  featureless corridors. Residual failure modes: aggressive rotation over
  textureless scenes, heavy dynamics in view, IR-absorbing surfaces.

## 4. One contract, two worlds (sim ↔ real HW)

Everything runs from a **ROS1 bag with fixed topics**; only the producer
differs:

```
topic                          type                 sim producer                real-HW producer
/uav1/cam0/color/image_raw    sensor_msgs/Image     Isaac (rigs/oakdpro.yaml)   record_oak.py (rectified left)
/uav1/cam1/color/image_raw    sensor_msgs/Image     Isaac                       record_oak.py (rectified right)
/uav1/cam0/depth/image_raw    Image 32FC1 [m]       Isaac depth annotator       record_oak.py (OAK stereo depth, mm→m)
/uav1/sensor_pod/imu          sensor_msgs/Imu       PodIMU @200 Hz              record_oak.py (device timestamps)
/uav1/ground_truth            nav_msgs/Odometry     mrs_bridge                  (absent — use mocap/none)
```

- **Sim**: `rigs/oakdpro.yaml` models the device as a pinhole global-shutter
  stereo pod (+depth on cam0, + flood illuminator for the night axis); rig
  support for `model: pinhole` and `depth: true` lives in
  `sim/isaac/workspace/{rig_math,fisheye_rig}.py`. Record with
  `bench/record.sh rigs/oakdpro.yaml <name>`. Sim depth is ideal (RTX
  ground-truth depth) — an optimistic upper bound; the
  [luxonis-isaac-sim](https://github.com/luxonis/luxonis-isaac-sim) noise
  parameters (baseline/disparity/noise for OAK-D Pro) are the Phase-2 layer
  to add.
- **Real HW**: `hw/record_oak.py` (pure depthai + rosbags, no ROS install)
  records the identical bag: rectified mono pair, on-device depth aligned to
  cam0 (mm→32FC1 m), IMU with **device** timestamps, factory calibration
  dumped to `calib_<serial>.yaml` next to the bag. Flags: `--fps`, `--dot`
  (projector), `--flood` (LED). For VIO runs keep `--dot 0` or use frame
  interleaving (v2) — constant dots poison feature tracking.
- **Live HW mode** (after bag-mode parity is established): Spectacular AI
  runs natively on the device; cuVSLAM live via PyCuVSLAM's OAK example or
  depthai-ros → isaac_ros_visual_slam; RTAB-Map live via its depthai driver
  or depthai-ros.

Evaluation: `bench/evaluate.py gt.txt est.txt` unchanged (ATE / RPE /
coverage / gaps); map quality vs Isaac scene mesh is the Track E metric
(`experiments/06_dense_mapping`, accuracy/completeness — Phase 2).

## 5. Test matrix (extends bench/README.md)

```
options {spectacularai, cuvslam(+nvblox), openvins+rtabmap}
  × source {sim bag, HW bag, HW live}
  × lighting {day, night+flood, half}      (sim axis; HW: lab lights off)
  × projector {dot off, dot on}            (depth quality vs VIO poisoning)
```

Priority rows: all three options × sim day/night (bag) first — that fills
the comparison table with GT; then HW bag day; then HW night (flood); then
live latency measurements on Orin.

## 6. Status

- [x] Research + this report
- [x] `rigs/oakdpro.yaml` + pinhole/depth rig support + unit tests
- [x] Runners + converters in `experiments/08_oakdpro_slam/` (bag-mode, all 3 options)
- [x] Dockerfiles: `docker/{spectacularai,cuvslam,rtabmap,nvblox}` — **built and smoke-tested 2026-08-29** on the dev laptop (RTX 4070 8 GB) with `docker/build_oakd_lane.sh` (sequential, under `bench/guard.sh`), on a synthetic full-resolution OAK-D contract bag (noise images, so no real tracking — the API/plumbing is what was verified):
  - cuVSLAM: `cuvslam` **17.0.0** wheel (cu12, cp310) from the GitHub release page; `cuvslam_track.py` rewritten against the wheel's real API (`Tracker.OdometryConfig`/`OdometryMode.Inertial`, `Distortion.Model.Pinhole`, `track()` → `(PoseEstimate, slam_pose)`); rig origin = cam0 optical, output re-expressed at the pod IMU; 12/12 frames tracked in Inertial mode on the GPU
  - cuVSLAM `--map` → nvblox: depth dump (frames grouped by stamp in any bag order) → `fuse_3dmatch` (upstream root build, renderer/tests/torch off, 3-channel color PNGs, 0-based contiguous frames) → `mesh.ply` (125 k vertices from 12 noise frames)
  - Spectacular AI: `bag2sai.py` recording accepted by the SDK `Replay`, 9 VIO poses from 12 frames
  - RTAB-Map: `rgbd_odometry` + `rtabmap` + export pipeline runs headless; `octomap_saver` fixed (`-f` means *full map*, not file); OctoMap service availability in the apt build still to confirm
- [x] `hw/record_oak.py` recorder (needs a physical OAK-D Pro to validate)
- [x] Side-by-side with the fisheye pod on one drone: `rigs/pod3_oakdpro.yaml` (composite rig), `bench/split_bag.py`, `bench/compare_rigs.sh` — same flight, same GT, OpenVINS on both + cuVSLAM (bench/README.md "Side-by-side reference")
- [ ] Sim bring-up of the oakdpro rig (depth topic name VERIFY-IN-SIM; needs the ≥24 GB GPU host)
- [ ] First matrix row: 3 options × sim day bag
- [ ] SAI recording-format validation against a real SDK replay (format doc: https://spectacularai.github.io/docs/sdk/recording.html)
- [x] PyCuVSLAM wheel/install pin: `cuvslam==17.0.0+cu12` (ARG `CUVSLAM_VERSION` in `docker/cuvslam/Dockerfile`; Orin = same tag family, aarch64 wheel, JetPack 6)
- [ ] HW validation pass + live-mode wiring
- [ ] Projector interleaving (dot↔flood per frame) in `record_oak.py` (v2)

## Sources

[Luxonis VIO/SLAM overview](https://docs.luxonis.com/overview/toplevel-features/vio_and_slam) ·
[Spectacular AI OAK wrapper](https://spectacularai.github.io/docs/sdk/wrappers/oak.html) ·
[spectacularai PyPI](https://pypi.org/project/spectacularai/) ·
[HybVIO paper](https://openaccess.thecvf.com/content/WACV2022/papers/Seiskari_HybVIO_Pushing_the_Limits_of_Real-Time_Visual-Inertial_Odometry_WACV_2022_paper.pdf) ·
[isaac_ros_visual_slam](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam) ·
[PyCuVSLAM](https://github.com/NVlabs/PyCuVSLAM) ·
[nvblox](https://github.com/nvidia-isaac/nvblox) ·
[nvblox_ros2 standalone](https://github.com/donceykong/nvblox_ros2) ·
[RTAB-Map](http://introlab.github.io/rtabmap/) ·
[DepthAI projector/flood alternation](https://luxonis-depthai-python.readthedocs-hosted.com/en/develop/samples/MonoCamera/mono_preview_alternate_pro/) ·
[depthai-ros IMU timestamps #461](https://github.com/luxonis/depthai-ros/issues/461) ·
[VINS-Fusion on OAK-D Pro W #241](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/issues/241) ·
[luxonis-isaac-sim](https://github.com/luxonis/luxonis-isaac-sim)
