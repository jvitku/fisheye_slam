# podslam — the in-house visual-inertial odometry (product lane)

*Started 2026-08-30 after the cuVSLAM diagnosis (docs/CUVSLAM_DIAGNOSIS_2026-08-30.md): a closed
core with no control over its feature detector, outlier model or re-initialisation policy is not
a product we can own. podslam is permissively licensed end to end (Apache-2.0 for our code;
OpenCV Apache-2.0, GTSAM BSD, PyTorch BSD) and is designed to be extended with learned components
inside the loop, not only in front of it.*

## What it is

A multi-camera (stereo today, N-camera by construction) visual-inertial odometry:

```
images ─► conditioning ─► Frontend ─► Tracker ─► Backend ─► pose (IMU frame, TUM)
              │              │           │           │
              │              │           │           └ GTSAM fixed-lag smoother: pose/velocity/bias
              │              │           │             per keyframe, combined IMU factors, projection
              │              │           │             factors on normalized coordinates, Huber
              │              │           └ keyframe policy, stereo triangulation, landmark lifecycle
              │              └ KLT (OpenCV) | XFeat (learned, Apache-2.0) | yours: bearings + ids out
              └ norm | clahe | gamma | nlmeans | enhance:<torchscript>  and masks: circle | sat | learned:<pt>
imu ─► static init (gravity, gyro bias) ─► preintegration ─► gyro rotation prediction for the front-end
```

- **Lens-model agnostic.** Front-ends emit unit bearings (KB4 / Double Sphere / EUCM / pinhole from
  `tools/fisheye`); the back-end only sees normalized coordinates. Any camera the rig yaml can
  describe works; the same rig yamls drive the sim, the bench and podslam.
- **Same contract as everything else.** Reads the bag contract (`/uav1/<cam>/color/image_raw`,
  `/uav1/sensor_pod/imu` — or the topics named in a real-rig yaml such as `rigs/tumvi_room1.yaml`),
  writes `est.tum` in the IMU frame + `frames.csv`, so `bench/compare_rigs.sh`,
  `bench/sweep_summary.py` and `bench/collect_results.py` work unchanged.

## The robustness policy (the cuVSLAM lessons, built in)

1. **The world frame is never re-created.** A visual loss is bridged by IMU propagation; the smoother
   is rebuilt around the last estimate if it ever fails numerically ("soft reset"), never at a new origin.
2. **Motion prediction comes from the gyro**, not an internal constant-velocity model: the KLT
   initial flow is the IMU-rotated previous bearing, so 3 rad/s turns stay inside the search window.
3. **Photometric normalisation by default** (`--preprocess norm`): auto-exposure steps do not break
   brightness constancy.
4. **Masks everywhere**: image circle (f-theta lenses), per-frame saturation blobs, learned masks —
   applied to detection, tracking and stereo.
5. **No landmark without parallax**: stereo-triangulated points need ≥ 0.5° between the rays
   (a parallel-ray point has no depth information and makes the linear system singular — the
   first real bug the synthetic test caught).

## ML extension points

| Seam | Interface | Shipped today |
|---|---|---|
| `podslam.frontend.Frontend` | `process(t_ns, images, masks, dR_imu) -> FrameFeatures` (bearings + persistent ids, stereo ids shared) | `klt` (GFTT + pyramidal LK), `xfeat` (detector/descriptor + MNN matching on GPU) |
| `podslam.conditioning.build_conditioner` | `f(img) -> img` chain | norm, clahe, gamma, nlmeans, `enhance:<torchscript>` (U-Net trained by `experiments/09_cuvslam_robustness/train_enhancer.py`) |
| `podslam.conditioning.build_mask_provider` | `f(img, static_mask) -> mask` (255 = valid) | `sat`, `learned:<torchscript>` |
| `podslam.backend.Backend` | keyframes / landmarks / observations / optimize | GTSAM `IncrementalFixedLagSmoother` |

A learned depth prior (for landmark initialisation without stereo parallax) and a learned feature
reliability weight (per-observation noise scaling) are the next two seams; both are local changes
in `Tracker._add_landmarks_and_factors` / `Backend.add_observation`.

## Running

```bash
uv run python -m podslam.track_bag datasets/data/tumvi/room1_day.bag out/ --rig rigs/tumvi_room1.yaml
bench/sweep_podslam.sh night klt --preprocess clahe          # -> bench/results/room1_sweep/podslam_klt_night
uv run python -m bench.sweep_summary                          # ATE / early-late RMSE / RPE / obs per frame
uv run pytest podslam/tests                                   # rig conventions + synthetic end-to-end VIO
docker run --gpus all 3dfe/podslam ...                        # torch + GTSAM + OpenCV for the learned parts
```

The synthetic test (`podslam/tests/test_synthetic.py`) flies a KB4 stereo rig through a random point
cloud with IMU derived from the trajectory and a perfect front-end: it must recover the trajectory to
< 3 cm with zero solver resets (currently 0.7 cm), which pins down every convention in the chain.

## Status / roadmap

- [x] Estimator core validated synthetically; KLT and XFeat front-ends; conditioning + masks; bag CLI
- [x] Real data, first complete flights (2026-08-30, KLT, untuned): TUM-VI room1 day / night /
      transition ATE **19.6 / 16.5 / 22.6 cm** (OpenVINS 7.4 / 7.6 / 6.1, cuVSLAM 11.4 / 13.1 / 17.2).
      The first 37 s of the day flight alone — where cuVSLAM loses tracking three times — score 2.9 cm;
      the whole-flight numbers are dominated by a heading drift in the last 40 s that coincides with the
      remaining solver soft resets (11 / 26 / 14 per flight, each dropping every landmark). Scale is
      right (path-length ratio 0.99–1.00). `bench/reports/room1_review.py` renders the comparison.
- [ ] Remove the reset path: rebuild the smoother *with* the healthy landmarks, mono landmark
      initialisation from motion parallax so the window never starves
- [ ] Learned enhancer and learned masks trained on room2, evaluated on room1
- [ ] Bearing-only factors for the fisheye rim (> 80° off-axis is currently unused)
- [ ] Mono landmark initialisation from parallax over time (today: stereo only)
- [ ] N-camera pod (triangle) and the sim recordings; ROS 2 live node; Orin timing
