# Track G — OAK-D Pro SLAM: three stacks, one bag contract, sim ↔ real HW

Implements the three options from **[docs/oak_d_pro_slam.md](../../docs/oak_d_pro_slam.md)**
(read that first — rationale, robustness expectations, sources):

| Option | Runner | Odometry | Dense map | Image |
|---|---|---|---|---|
| 1 Spectacular AI | `run_spectacularai.sh` | SAI VISLAM (replay) | keyframe cloud `map.ply` | `3dfe/spectacularai` |
| 2 cuVSLAM + nvblox | `run_cuvslam.sh [--map]` | PyCuVSLAM (GPU) | nvblox TSDF `mesh.ply` | `3dfe/cuvslam`, `3dfe/nvblox` |
| 3 OpenVINS/RTAB-Map | `run_rtabmap.sh [rtabmap\|external]` | rgbd_odometry or external | **OctoMap** `.bt` + `cloud.ply` | `3dfe/rtabmap` (+`3dfe/openvins`) |

Every runner consumes the **same ROS1 bag** (contract in
docs/oak_d_pro_slam.md §4: `/uav1/cam{0,1}/color/image_raw`,
`/uav1/cam0/depth/image_raw` 32FC1 [m], `/uav1/sensor_pod/imu`) and emits
`out/<option>/<bag>/est.tum` for `bench/evaluate.py`.

## Getting a bag

**Sim (Isaac):**
```bash
# on the GPU host, from sim/isaac:
RIG_CONFIG=/rigs/oakdpro.yaml SIM_LIGHTING=day ./start_all.sh -d
# fly the benchmark trajectory, then:
../../bench/record.sh rigs/oakdpro.yaml oakdpro_day
```
`rigs/oakdpro.yaml` models the device (pinhole stereo 1280×800, 7.5 cm
baseline, depth on cam0, flood-LED illuminator for `SIM_LIGHTING=night`).
Sim depth is ideal RTX depth — an optimistic upper bound.

**Real HW (no ROS needed):**
```bash
pip install depthai rosbags numpy pyyaml
python3 hw/record_oak.py oakdpro_lab.bag --fps 20 --duration 120 \
    --flood 0.8   # night runs; keep --dot 0 for VIO (dots poison tracking)
```
Also writes `calib_<serial>.yaml` (factory calibration) — pass it to runners:
`./run_spectacularai.sh oakdpro_lab.bag --calib calib_<serial>.yaml`.

## Running

```bash
# build images once (exploratory — expect first-build fixes, like voxblox)
docker build -t 3dfe/spectacularai ../../docker/spectacularai
docker build -t 3dfe/cuvslam      ../../docker/cuvslam      # NVIDIA GPU
docker build -t 3dfe/rtabmap      ../../docker/rtabmap
docker build -t 3dfe/nvblox       ../../docker/nvblox       # NVIDIA GPU

./run_spectacularai.sh <bag>          # option 1: est.tum + map.ply
./run_cuvslam.sh <bag> --map          # option 2: est.tum + mesh.ply
./run_rtabmap.sh <bag>                # option 3: est.tum + octomap.bt + cloud.ply

# evaluate against sim GT (extract gt.tum from the bag's /uav1/ground_truth):
uv run python -m bench.evaluate gt.tum out/<option>/<bag>/est.tum
```

External-odometry variant of option 3 (the report's recommended open lane —
OpenVINS odometry per EVAL round 1): start `3dfe/openvins` live on the host
network remapping its odometry to `/uav1/odom`, then
`./run_rtabmap.sh <bag> external`.

## Live HW mode (after bag-mode parity)

- **Spectacular AI**: runs natively on-device (`spectacularAI.depthaiPlugin`),
  lowest-latency estimate; same Mapping API outputs.
- **cuVSLAM**: PyCuVSLAM OAK example, or depthai-ros → isaac_ros_visual_slam
  (+ isaac_ros_nvblox) on Orin.
- **RTAB-Map**: native depthai driver (`rtabmap --driver depthai`, on-device
  SuperPoint optional) or depthai-ros.

## Status / gotchas

- [ ] All three stacks are **written, not yet run** — each script marks its
  VERIFY-ON-FIRST-RUN points (SAI recording format fields, PyCuVSLAM API
  surface + wheel pin, rtabmap-export flags, fuse_3dmatch args).
- [ ] `record_oak.py` needs a physical OAK-D Pro pass (device-timestamp
  anchoring, IMU extrinsics EEPROM presence, IR intensity API names).
- IMU: we bypass depthai-ros entirely (its batching/timestamp issues are the
  documented OAK weak point) — the recorder uses device timestamps directly.
- The projector axis ({dot off, dot on} × VIO quality) is part of the test
  matrix; per-frame dot/flood interleaving is v2.
