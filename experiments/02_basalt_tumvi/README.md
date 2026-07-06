# Track B — Basalt on TUM-VI (Double Sphere stereo VIO)

**Goal:** the lean CPU baseline (most realistic Pi 5 candidate) and, more
importantly, the **calibration backbone** for the whole project
(`basalt_calibrate` / `basalt_calibrate_imu` with the Double Sphere model).

## Status
- [ ] Docker image builds (`docker/basalt/`)
- [ ] room1 VIO run (Basalt ships TUM-VI configs + Double Sphere calib)
- [ ] CPU/RSS comparison vs Track A on identical sequences
- [ ] Calibration dry-run on our own recordings (Phase 2)

## Run

```bash
docker build -t 3dfe/basalt docker/basalt
./experiments/02_basalt_tumvi/run.sh room1
```

Notes:
- Basalt reads TUM-VI natively (rosbag reader built in, no ROS needed at runtime).
- Upstream ships calibration for TUM-VI in `data/tumvi_512_ds_calib.json` and a
  VIO config in `data/tumvi_512_config.json` (names may drift — check the clone
  under `candidates/basalt/data/`).
- Calibration workflow reference: upstream `doc/Calibration.md`.
