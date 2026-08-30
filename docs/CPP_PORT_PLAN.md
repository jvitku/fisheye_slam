# podslam C++/Jetson port plan (Phase D)

*2026-08-31. Target: Orin Nano Super, 20 Hz × 3 cameras, ≤ 50 ms per frame set,
identical results to the Python reference (which stays as the research harness
and the test oracle).*

## Where the time goes today (Python, one laptop core)

Measured on TUM-VI room1 day, 400 frames, after the vectorised stereo check
(`bench/` profile, 2026-08-31):

| stage | share | notes |
|---|---|---|
| GTSAM Levenberg-Marquardt solve | **53 %** (62 ms / keyframe) | window of ≤ 32 keyframes, ~200 smart rig factors rebuilt per solve |
| KLT (2 × `calcOpticalFlowPyrLK` temporal + 2 × stereo) | 13 % | already C++ inside OpenCV |
| detector (`goodFeaturesToTrack` + response gate) | 8 % | C++ inside OpenCV |
| factor construction (87 k `SmartProjectionRigFactor` builds) | 6 % | pure wrapper overhead |
| marginalisation (`eliminatePartialMultifrontal` + containers) | 3 % | |
| bookkeeping, conditioning, bag I/O | rest | vanishes in a live node |

Whole-tracker wall time today: 33 ms/frame stereo, ~55–70 ms five cameras
(single thread). The algorithm is already bounded-compute (fixed window, fixed
feature budget); the port is an engineering translation, not a redesign.

## Port strategy

1. **Keep GTSAM, native.** The backend is plain GTSAM C++ (`SmartProjectionRigFactor`,
   `CombinedImuFactor`, `LinearContainerFactor`, partial elimination) — the Python
   wrapper is the overhead, not the library. Port `backend_smart.py` 1:1 into a
   `podslam::Window` class; reuse the exact parameterisation (gauge anchor,
   absolute LM stop, χ² gate, marginalisation modes). GTSAM builds on aarch64.
   Factor caching (rebuild only factors whose measurement set changed) is a free
   2–5 ms once in C++.
2. **Front-end on the GPU.** `cv::cuda::SparsePyrLKOpticalFlow` +
   `goodFeaturesToTrack` (CUDA) — the IMU-predicted initial flow and the
   response-floor gate carry over unchanged. The KB4/DS/EUCM projections are a
   header (`tools/fisheye/models.py` transcribed; the KB4 path is verified
   bit-equal to OpenCV fisheye).
3. **One thread per stage, bounded queues.** camera driver → conditioner
   (optional TensorRT enhancer, 3.7 ms at 512²) → tracker → window solve.
   Keyframe solves (7 Hz) overlap the next frames' tracking; the pose output at
   frame rate stays IMU-propagated exactly as in Python.
4. **The Python harness is the oracle.** Every ported stage gets a golden test:
   same bag in, per-stage outputs compared against the Python reference
   (tracks: identical ids/px within 1e-3; window solutions within 1e-6). The
   synthetic end-to-end test (0.35 cm) is ported first and gates every commit.
5. **ROS 2 node last**: subscriptions per camera + IMU, the rig yaml unchanged,
   TUM + TF outputs; the bag runner stays for benchmarks.

## Budget on Orin Nano Super (estimate, to be measured in Phase D2)

| stage | today (laptop, 1 core, Python) | Orin target |
|---|---|---|
| conditioning + enhancer (TensorRT, optional) | 200 ms CPU / 3.7 ms GPU | ≤ 4 ms GPU |
| detect + track, 3 cams | ~20 ms | ≤ 10 ms GPU |
| window solve (7 Hz keyframes, amortised) | 62 ms / kf ≈ 20 ms/frame | ≤ 25 ms / kf on 2 A78 cores ≈ 8 ms/frame |
| total per frame set | ~55 ms | **≤ 25 ms** (goal ≤ 50) |

## Order of work

1. ~~`podslam-cpp/` skeleton: KB4/DS/EUCM models + golden tests~~ **done** (≤1.7e-9 px).
2. ~~Window/backend port + replay parity~~ **done** (1.7e-9 m over 114 keyframes,
   identical event counts, production LM settings; GTSAM 4.3a2 pinned, TBB-free).
3. ~~KLT front-end + track parity~~ **done at the honest criterion**: teacher-forced
   per-frame parity 0.69 % unmatched / 1.25 % extra / ≤0.5 px (OpenCV 5.0.0+IPP
   pinned to the wheel; free-running sets diverge chaotically from threshold-edge
   corners — documented, not chased).
4. ~~Tracker in C++, all benchmark rows~~ **done**: full pipeline end-to-end on
   dumped frames — ATE vs GT, C++ (Python): room1 day **7.7** (8.8), night **15.9**
   (16.2), transition **13.3** (14.5), room2 **11.8** (12.0), Hilti exp14 5-cam
   **11.8** (9.4) cm; full coverage, zero failed solves, event counts within 2 %.
   35.6 ms/frame single-core unthrottled. Remaining: bag/ROS 2 I/O.
5. Orin build + timing; ROS 2 node; then the learned parts as TensorRT engines.

Risks: GTSAM version drift between the Python wheel (4.3a2) and the C++ build —
pin the same commit; `SmartProjectionRigFactor` bool-threshold trap does not
exist in C++ (typed API), but the EPI/cheirality behaviour must be mirrored
(EPI off).
