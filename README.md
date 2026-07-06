# 3d_fisheye

Exploratory evaluation of open-source SLAM/VIO for multi-fisheye + IMU rigs
(2–3 cameras now, 6 later) on cheap rolling-shutter cameras, targeting
Raspberry Pi 5 / Jetson Orin Nano Super.

**Start here: [docs/REPORT.md](docs/REPORT.md)** — candidate overview, decision
rationale, and the phased evaluation plan.

## Quick start

```bash
# Fetch all candidate systems (shallow clones into candidates/)
./candidates/clone.sh

# Lens-model library + tests (needs uv: https://docs.astral.sh/uv/)
uv run pytest

# Download TUM-VI eval sequences (~few GB)
./datasets/download_tumvi.sh room1

# Per-track experiments
ls experiments/
```

## Tracks

| Track | System | Role |
|---|---|---|
| A | [OpenVINS](https://github.com/rpng/open_vins) | Primary N-camera VIO candidate (online time-offset calib) |
| B | [Basalt](https://gitlab.com/VladyslavUsenko/basalt) | Calibration backbone + lean stereo baseline |
| C | [OpenMAVIS](https://github.com/MAVIS-SLAM/OpenMAVIS) | Multi-camera (>2) SLAM with loop closure |
| D | [XFeat](https://github.com/verlab/accelerated_features) / [LightGlue](https://github.com/cvg/LightGlue) / [DBA-Fusion](https://github.com/GREAT-WHU/DBA-Fusion) | Learned front-end + hybrid ceiling |

Docker images per track live in `docker/`. All are exploratory — see the
per-experiment READMEs for status.

## Unified simulation benchmark

All candidates are quantified on identical data: one Isaac Sim world, one
flown trajectory, camera rigs of **2 / 3 / 6 fisheyes** mounted on the drone
(`rigs/rig_{2,3,6}cam.yaml`), unsynchronized-camera variants generated
deterministically offline. See **[bench/README.md](bench/README.md)** for the
contract and metrics. The Isaac Sim + Pegasus setup in `sim/isaac/` is copied
from `swarm_stack/tools/isaac` (provenance in its CLAUDE.md).

The target product is modeled as a **sensor pod**: N coplanar same-direction
fisheye cameras + the pod's own PX4-FC IMU, mounted on top of the drone —
`rigs/pod_2cam.yaml` (stereo pair) and `rigs/pod_3cam_triangle.yaml`
(triangle). Pod output = position + dense 3D map
(`experiments/06_dense_mapping/`, voxblox/nvblox).

```bash
# sim (GPU host): fisheye benchmark drone with the 3-cam rig
cd sim/isaac && RIG_CONFIG=/rigs/rig_3cam.yaml ./start_all.sh -d

# record a run, generate an unsync variant, evaluate a candidate output
./bench/record.sh rigs/rig_3cam.yaml slow_scan_3cam
uv run python -m bench.skew_bag in.bag out.bag --offset /uav1/cam1/color/image_raw=0.015
uv run python -m bench.evaluate gt.txt est.txt
```
