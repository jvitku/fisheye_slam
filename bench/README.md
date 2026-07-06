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
  candidate configs must be generated from them (generator tool is Phase 1
  work). All three rigs share identical intrinsics/rates — camera count and
  placement are the only variables.
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

candidates {openvins, basalt*, openmavis, dba-fusion, mast3r-slam†, dpvo} ×
rigs {2cam, 3cam, 6cam*, pod_2cam, pod_3cam_triangle} ×
trajectories {slow_scan, fast_yaw} ×
lighting {day, night+IR, half-transition} ×
skew {sync, 15ms, 40ms, 15ms+2ms-jitter}

(*) basalt is stereo-only → 2cam column only; openmavis is the only 6cam-ready
candidate today; openvins 6cam needs a config experiment. (†) mast3r-slam /
dpvo are monocular (evaluate with `--scale`); see `experiments/07_learned_slam/`.
Run what fits, report the holes honestly. The full cross-product is large —
prioritize: {openvins, dpvo} × {pod rigs} × {day, night} × {sync, 15ms} first.

## Getting ground truth into TUM format

`/uav1/ground_truth` (nav_msgs/Odometry) → TUM text. Extraction helper is
Phase 1 work alongside the candidate runners (one-liner with `rosbags` — see
`skew_bag.py` for the API pattern).

## Status

- [x] Isaac setup copied from swarm_stack (`sim/isaac`, provenance in README there)
- [x] `bench_drone.py` + `fisheye_rig.py` (untested scaffold — needs GPU host bring-up)
- [x] `skew_bag.py` + `evaluate.py` (unit-tested offline)
- [ ] Sim bring-up: verify fisheye rendering, camera orientation convention, rates
- [ ] Scripted benchmark trajectories (MRS goto sequence, identical every run)
- [ ] GT extraction + candidate config generation from rig yamls
- [ ] First full matrix row: openvins × 3 rigs × slow_scan × sync
