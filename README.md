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
