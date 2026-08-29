# Unified benchmark: same world, same flight, same data — for every candidate

All candidates are quantified in the **exact same environment**: one Isaac Sim
world, one flown trajectory, one recording per rig. The only variables are
*candidate*, *camera count* (2/3/6) and *timestamp-skew profile*.

## The contract

```
                    Isaac Sim (sim/isaac, copied from swarm_stack)
  RIG_CONFIG=/rigs/rig_{2,3,6}cam.yaml  ->  bench_drone.py spawns the fisheye
  SIM_ENVIRONMENT=<world>                   rig on the drone (exact f-theta
                                            intrinsics = ground-truth calib)
                    |
        fly trajectory (MRS stack / PX4, same goto script every run)
                    |
        bench/record.sh rig.yaml <name>     ROS1 bag: cams + /uav1/imu
                    |                       + /uav1/ground_truth
        bench/skew_bag.py                   offline unsync variants
                    |                       (per-camera offset + jitter)
        candidate runners (experiments/*)   consume the SAME bag
                    |
        bench/evaluate.py gt.txt est.txt    ATE / RPE / coverage / gaps
```

Key design decisions:

- **Cameras are recorded perfectly synced; unsync is injected offline** by
  `skew_bag.py`. One recording → any number of deterministic skew profiles,
  and every candidate sees byte-identical images. This makes the central
  question of the project ("how much does lack of sync hurt, per candidate?")
  a controlled, repeatable experiment.
- **Rig yamls are the single source of truth** (`rigs/rig_{2,3,6}cam.yaml`):
  the sim spawns cameras from them, `record.sh` derives topics from them, and
  candidate configs are generated from them (`gen_openvins_config.py`). All
  three rigs share identical intrinsics/rates — camera count and placement
  are the only variables.
- **Ideal equidistant (kb4, k1..k4=0) intrinsics** rendered exactly by Isaac's
  f-theta camera → calibration error is eliminated as a benchmark variable.
  (Realistic distortion + calibration noise can be added later as another
  controlled axis.)
- **Robustness is a first-class metric**: `evaluate.py` reports coverage and
  dropout gaps, not just ATE. A candidate that diverges quietly scores worse
  than one that dies loudly and recovers.

## The sensor pod (the thing we're actually building)

Besides the body-mounted survey rigs (`rig_{2,3,6}cam.yaml`), the benchmark
models the target product: a **self-contained sensor pod** mounted on top of
the drone — N coplanar fisheye cameras facing the same direction plus the
pod's **own PX4 FC used as the IMU source** (independent of the drone's flight
FC). Two variants:

| Rig | Layout |
|---|---|
| `rigs/pod_2cam.yaml` | horizontal stereo pair, baseline 0.12 m |
| `rigs/pod_3cam_triangle.yaml` | equilateral triangle (side 0.12 m) — adds vertical/diagonal baselines |

In sim the pod FC IMU is `PodIMU` (`sim/isaac/workspace/pod_imu.py`): rigid
mounted at the pod origin with correct offset physics, published on
`/uav1/sensor_pod/imu` at 400 Hz; the drone's body IMU stays on `/uav1/imu`
as reference. Camera/IMU mounts in pod rigs are pod-relative and resolved to
the body frame by `rig_math.py` (unit-tested on the host).

Pod **output** = position (winning VIO candidate on pod cams + pod IMU) and a
dense 3D map (voxblox/nvblox TSDF) — that stage lives in
`experiments/06_dense_mapping/`.

## Side-by-side reference: the pod and an OAK-D Pro on one drone

`rigs/pod3_oakdpro.yaml` is a **composite rig**: a `pods:` list mounting the
3-cam fisheye pod and the OAK-D Pro (`rigs/oakdpro.yaml`, unchanged) on the
same drone, so one flight produces one recording with both devices — same
motion, same lighting, same ground truth. Sim topics are namespaced per
member (`/uav1/pod_cam0/...`, `/uav1/pod/imu`, `/uav1/oakd_cam0/...`,
`/uav1/oakd/imu`); each member keeps its own IMU model (PX4-FC vs BMI270
noise from the yaml) and its own IR flood LED.

```bash
# GPU host: both devices on the drone
RIG_CONFIG=/rigs/pod3_oakdpro.yaml SIM_LIGHTING=day ./sim/isaac/start_all.sh -d
# fly the benchmark trajectory, record everything
bench/record.sh rigs/pod3_oakdpro.yaml combo_day 120
# split -> per-device contract bags + GT at each device's IMU, run OpenVINS on
# both (same estimator, rig is the only variable) + cuVSLAM on the OAK-D,
# evaluate, print the table
bench/compare_rigs.sh datasets/data/sim/combo_day.bag rigs/pod3_oakdpro.yaml
```

Pieces (all host-side, unit-tested, no ROS install):

| Tool | Does |
|---|---|
| `bench/split_bag.py` | composite bag → `combo_day_pod.bag`, `combo_day_oakd.bag` following each member's single-rig contract (`/uav1/cam0/...`, `/uav1/sensor_pod/imu`) so every existing runner works unchanged |
| `bench/gt_extract.py` | `/uav1/ground_truth` → TUM, **expressed at the pod IMU** (`--rig`, `--pod`): a pod 12 cm above body center at 15° pitch is 3 cm off — the size of the ATE numbers — and SE3 alignment cannot absorb a lever arm |
| `bench/gen_openvins_config.py` | rig yaml → OpenVINS config dir (N cameras, `T_cam_imu` from the mounts, kb4/equidistant or pinhole, image-circle masks, IMU noise/rate/topic) |
| `bench/run_sim_candidate.sh` | one candidate × one sim bag × generated config → `bench/results/<name>/<run>/`. OpenVINS: 1–2 cameras use the deterministic serial bag reader; **3+ cameras use the live node + `rosbag play`** (`ros1_serial_msckf` supports only 1–2 cams), real-time by default (`BAG_RATE`) |
| `bench/collect_results.py` | now prefers a per-run `gt.txt` (each device has its own frame) |

The convergence loop this enables: iterate on the fisheye pod's stack
(learned front-end / depth allowed — see the design note on what is geometry
vs perception) until its row matches the OAK-D row on the same recording.

## Resource guards (don't kill the workstation)

Everything heavy runs under **`bench/guard.sh`** — `compare_rigs.sh` and
`run_sim_candidate.sh` re-exec themselves under it, `sim/isaac/start_all.sh`
runs its preflight, and every `docker run` in the runners appends
`$GUARD_DOCKER_ARGS` (memory / cpus / pids caps + a label the guard can kill).

```bash
bench/guard.sh --mem 16G --cpus 12 --disk-floor 20G -- bench/compare_rigs.sh combo.bag rigs/pod3_oakdpro.yaml
bench/guard.sh --check --vram-need 7G          # preflight only
GUARD_OPTS="--mem 8G --cpus 8" bench/run_sim_candidate.sh openvins rigs/oakdpro.yaml x.bag out/
```

| Stage | What it does |
|---|---|
| preflight | refuses to start when free disk (repo + docker root), available RAM, free VRAM (`--vram-need`), CPU package temperature or load average say the host can't afford it |
| limits | host processes in a systemd user scope with `MemoryMax`, no swap, pinned to the first `--cpus` cores, `nice`/`ionice`; containers via `--memory --memory-swap --cpus --pids-limit`; the Isaac compose file caps the sim at `ISAAC_MEM`/`ISAAC_CPUS` (20G / 16) |
| watchdog | every 5 s: kills the job (scope + labelled containers) if available RAM, free disk or free VRAM drop below the floors, the CPU package reaches `--temp-max` (90 °C), or the run exceeds `--timeout` (4 h). Log: `bench/results/guard/<id>.log` |

Defaults are for a shared workstation (12 GB, half the cores). The 8 GB-VRAM
laptop this was written on cannot run Isaac Sim with five rendered cameras
safely — `start_all.sh` says so and stops; `FORCE=1` overrides.

## Flying the benchmark trajectory

`bench/fly_trajectory.py` streams PX4 offboard setpoints (LOCAL_NED, 20 Hz)
along a time-parametrized path so every recording gets the same commanded
flight: `slow_scan` (lawn-mower survey, 0.6 m/s) or `fast_yaw` (figure-8,
1.5 m/s, continuous yaw). PX4 SITL publishes its offboard link to UDP 14540
on the host, which the tool listens on.

```bash
bench/record.sh rigs/pod3_oakdpro.yaml combo_day 150 &
uv run python -m bench.fly_trajectory --pattern slow_scan --duration 120
```

## Metrics (bench/evaluate.py)

| Metric | Meaning |
|---|---|
| ATE rmse/mean/median/max | absolute error after SE3 (or Sim3 with `--scale`) alignment |
| RPE rmse/max @ 1s | local drift, aggressive-motion sensitivity |
| coverage | fraction of GT time span with estimates (robustness) |
| gaps / longest_gap | tracking dropouts > 0.5 s |
| (per runner) CPU %, peak RSS | measured by the candidate runner scripts |

## Lighting axis (day / night / transition)

Isaac's RTX lighting is fully scriptable, so illumination is a controlled
benchmark variable (`SIM_LIGHTING` env, implemented in
`sim/isaac/workspace/lighting.py`):

| Mode | Scene | Pod IR illuminator (`POD_IR_LIGHT=auto`) |
|---|---|---|
| `day` | environment's own lighting | off |
| `night` | all scene lights ~0 + faint ambient | **on** — dominant light source |
| `half` | one half of the scene lit, other half night-dark; trajectory crosses the boundary | **on** |

The pod cameras are **NoIR** (no IR-cut filter) with an **850 nm IR flood LED
at the triangle center** (`illuminator:` in the pod rig yamls). In sim the
illuminator is a real shadow-casting RTX cone light rigidly attached to the
drone: at night it both *enables* the cameras and *hurts* VIO — the light and
its shadows move with the vehicle, so shadow edges are non-static features and
illumination is never constant between frames. That trade-off is exactly what
the night rows of the matrix measure. NIR is modeled as white light; NoIR
imaging = grayscale conversion at candidate input.

## The matrix

candidates {openvins, basalt*, openmavis, cuvslam, airslam*, dba-fusion} ×
rigs {2cam, 3cam, 6cam*, pod_2cam, pod_3cam_triangle} ×
trajectories {slow_scan, fast_yaw} ×
lighting {day, night+IR, half-transition} ×
skew {sync, 15ms, 40ms, 15ms+2ms-jitter}

(*) basalt and airslam are stereo(+IMU)-only → 2cam / pod front-pair columns;
openmavis is the only 6cam-ready open candidate; cuvslam consumes the pod as
1–3 stereo pairs + IMU (its native input model). MASt3R-SLAM and DPVO were
removed from the matrix 2026-07-06 (license / monocular — see
`experiments/07_learned_slam/`). Run what fits, report the holes honestly.
The full cross-product is large — prioritize:
{openvins, cuvslam} × {pod rigs} × {day, night} × {sync, 15ms} first.
License lanes: GPL systems (openvins, openmavis, airslam, dba-fusion) are
benchmark yardsticks only; the product lane is cuvslam / permissive hybrid
(docs/REPORT.md §2b).

## Getting ground truth into TUM format

`python -m bench.gt_extract bag.bag gt.txt [--rig rig.yaml] [--pod ns]` —
`/uav1/ground_truth` (nav_msgs/Odometry) → TUM, optionally re-expressed at a
pod's IMU frame (see the side-by-side section for why that matters).

## Status

- [x] Isaac setup copied from swarm_stack (`sim/isaac`, provenance in README there)
- [x] `bench_drone.py` + `fisheye_rig.py` (untested scaffold — needs GPU host bring-up)
- [x] `skew_bag.py` + `evaluate.py` (unit-tested offline)
- [ ] Sim bring-up: verify fisheye rendering, camera orientation convention, rates
- [x] Scripted benchmark trajectories (`fly_trajectory.py`, PX4 offboard; VERIFY-IN-SIM)
- [x] Resource guard for every heavy step (`guard.sh`)
- [x] GT extraction (`gt_extract.py`, pod-frame aware) + OpenVINS config generation from rig yamls (`gen_openvins_config.py`)
- [x] Composite rig (pod + OAK-D Pro side by side) + bag split + one-command comparison (`compare_rigs.sh`)
- [ ] First full matrix row: openvins × 3 rigs × slow_scan × sync
