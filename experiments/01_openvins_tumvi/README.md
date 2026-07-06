# Track A — OpenVINS on TUM-VI (fisheye stereo + IMU)

**Goal:** establish the primary-candidate baseline; then stress the property we
actually care about — tolerance to **unsynchronized** cameras via online
time-offset calibration.

## Status
- [ ] Docker image builds (`docker/openvins/`)
- [ ] room1 runs, ATE computed with `evo`
- [ ] Time-offset stress test (skew cam1 timestamps by 5/15/40 ms, `calib_cam_timeoffset: true`)
- [ ] 3-cam config dry-run (OpenVINS `max_cameras: 3` on simulated rig)

## Run

```bash
# from repo root
./datasets/download_tumvi.sh room1
docker build -t 3dfe/openvins docker/openvins
./experiments/01_openvins_tumvi/run.sh room1
```

Notes:
- OpenVINS ships a TUM-VI config at `config/tum_vi/` in the upstream repo
  (kb4/equidistant intrinsics); the runner uses it directly.
- Key config knobs for our use case: `max_cameras`, `calib_cam_timeoffset`,
  `calib_cam_extrinsics`, `calib_cam_intrinsics`, `use_klt`.
