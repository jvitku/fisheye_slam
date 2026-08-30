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
bench/sweep_podslam.sh night enh --preprocess enhance:models/enhancer_night.pt+norm   # room1 conditions
bench/run_hilti.sh exp14_basement_2 stereo --cams cam0,cam1 --kf-every 6              # Hilti 5-cam rig, subset
PYTHONPATH=. uv run python bench/sweep_summary.py            # room1: ATE / first-last 20 s / RPE / obs per frame
PYTHONPATH=. uv run python bench/hilti_summary.py            # Hilti: ATE SE3 / Sim3 / scale / coverage
PYTHONPATH=. uv run python bench/drift_analysis.py <run>     # yaw / tilt / scale / position drift over time
bench/enhance_bag.py <bag> <out.bag> models/enhancer_night.pt   # GPU pass of a TorchScript conditioner
uv run pytest podslam/tests                                   # rig conventions + synthetic end-to-end VIO
```

Knobs that matter (all in `TrackerConfig`, exposed by `track_bag`): `imu_noise_scale` (estimator-side
inflation of the rig's IMU noise; Basalt's TUM-VI values 5.7/1.8/1.2/4.5 are the default and are
**IMU-specific** — on the Hilti BMI085 the measured Allan values without inflation are better),
`marg_mode` (`all` = MSCKF-style absorption of every landmark seen from a marginalised keyframe,
`ended`, legacy `pin`), `chi2_gate`, `noise_gate` (detector response ≥ k × frame median), `kf_every`
+ optional motion-adaptive keyframes, `lag_s` / `max_window_kf`, `px_sigma`, `stereo_passes`.
Rig yaml: cameras with Kalibr `T_cam_imu`, KB4/DS/EUCM intrinsics, per-camera `time_shift_s`; IMU
noise densities, `accel_scale` (measured |a| at rest vs local g).

The synthetic test (`podslam/tests/test_synthetic.py`) flies a KB4 stereo rig through a random point
cloud with IMU derived from the trajectory and a perfect front-end: it must recover the trajectory to
< 3 cm with zero solver resets (currently 0.35 cm), which pins down every convention in the chain.

## Status / roadmap

- [x] Estimator core validated synthetically; KLT and XFeat front-ends; conditioning + masks; bag CLI
- [x] Smart-factor backend with real marginalisation (Schur complement kept as LinearContainerFactor,
      hard gauge anchor on the first pose, absolute LM stop test), chi² outlier gate, IMU noise inflation:
      TUM-VI room1 day / night / transition **8.8 / 16.2 / 13.4 cm**, zero resets and zero failed solves
      (2026-08-30; OpenVINS 7.4 / 7.6 / 6.1, Basalt 10.8 / 28.5 / 20.9, cuVSLAM 11.4 / 13.1 / 17.2).
      Full history with every accepted and rejected change: `bench/results/campaign.json`.
- [x] Learned low-light enhancer (0.12 M-param U-Net, room2-trained with the night degradation model,
      3.7 ms/frame on GPU): night 16.2 → **13.7 cm**, tracker 69 → 39 ms/frame.
- [x] Real 5-camera benchmark (Hilti-Oxford 2022 exp14, `rigs/hilti2022.yaml`, `bench/run_hilti.sh`,
      OpenVINS baselines in `bench/configs/openvins_hilti2022`): podslam stereo 21.1 cm SE3 / 7.7 Sim3
      (scale 0.976, tilt 1.2°) vs OpenVINS stereo 5.4 cm — the gap is scale + tilt, not tracking.
- [ ] Per-IMU noise setting (online residual-based) and online camera–IMU extrinsic / time-offset
      refinement — what OpenVINS has and we lack on the Hilti rig
- [ ] 5-camera podslam rows (Hilti) and the N-camera pod sim recordings (needs the GPU host)
- [ ] Bearing-only factors for the fisheye rim (> 80° off-axis is currently unused)
- [ ] Learned masks; enhancer generalisation to real low light (trained on synthetic night only)
- [ ] C++ port of KLT + preintegration + smart-factor window for Jetson real time (Python: 45–70 ms/frame
      single-threaded on a laptop core); ROS 2 live node
