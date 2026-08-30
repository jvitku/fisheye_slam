# cuVSLAM 17 on real fisheye data — what goes wrong, and why we stopped tuning it

*2026-08-30 · TUM-VI room1 (512×512 KB4 stereo, 200 Hz IMU, mocap GT) · day / night-IR proxy / transition ·
runner `experiments/08_oakdpro_slam/cuvslam_track.py`, sweep `bench/sweep_cuvslam.sh`, `bench/sweep_summary.py`*

## Findings (systematic, one knob at a time)

1. **Day error is one event, not drift.** ATE 11.4 cm comes from the first 25 s: cuVSLAM loses tracking
   three times (t = 11.6, 23.3, 25.0 s), each followed by a fresh detection burst, and after the last loss it
   *re-initialises its frame* — the trajectory before t ≈ 25 s sits ~19° rotated relative to the rest
   (`rot20°` column). Two losses coincide with the fastest rotations of the flight (2.5–3 rad/s, p95),
   one with a 60 % auto-exposure brightness step. Night, same motion, no loss — so it is image-driven.
2. **Night error is drift in the darkest 40 s** (last-20 s RMSE 19 cm vs 10 cm early), with *more*
   features than daytime (426 vs 282 per frame — noise texture), not starvation.
3. **Mask polarity**: 255 = ignore (an all-255 mask yields zero observations).
4. **Fixes that work (day):** `--no-motion-model` → 8.0 cm, no losses, no re-init (the internal pose
   prediction fights the fast rotations); `--preprocess clahe` → 6.7 cm, RPE p95 4.2 cm (photometric
   normalisation defeats exposure steps); per-frame saturation mask → 8.0 cm. Denoising, multicam mode,
   SLAM mode, IMU noise scaling: no effect or worse. Without the IMU: 4× worse.
5. **Night:** the image-circle mask made it worse (coverage 0.90); denoising/CLAHE/gamma pending in the
   same sweep table (`bench/results/room1_sweep/summary.json`).

| run | ATE cm | first 20 s | last 20 s | rot20° | RPE p95 | losses |
|---|---:|---:|---:|---:|---:|---:|
| base_day | 11.4 | 23.4 | 8.4 | 19.6 | 21.6 | 3 |
| nomm_day | 8.0 | 7.1 | 7.5 | 1.2 | 6.0 | 0 |
| clahe_day | 6.7 | 13.8 | 5.8 | 7.3 | 4.2 | 0 |
| sat_day | 8.0 | 17.6 | 7.0 | 8.7 | 6.1 | 0 |
| base_night | 10.9 | 10.2 | 19.2 | 1.7 | 4.6 | 0 |

## Why we stopped here

The closed core exposes no feature detector, no outlier model, no re-initialisation policy — the
frame reset after a loss is the single biggest error source and it cannot be turned off. With a
commercial-license requirement and the need for 100 % control (and ML extension points inside the
loop, not only in front of it), the decision on 2026-08-30 was to build the in-house lane instead:
**`podslam/`** — permissive components only (OpenCV, GTSAM/BSD, PyTorch), own front-end and policy,
same bag contract and outputs, learned front-end / conditioning / masks as first-class plug-ins.
The requirements above are its acceptance criteria: no frame re-initialisation on loss (IMU carries
the frame), IMU-predicted tracking through 3 rad/s, photometric normalisation, per-frame masks.
