# Evaluation Round 1 — Day / Night / Hybrid comparison

*2026-07-06 · 3d_fisheye Phase 1 (first executable slice) · autonomous run*

## TL;DR

Two classical VIO candidates (OpenVINS, Basalt) and two front-ends (ORB,
XFeat) were run on the same real fisheye+IMU sequence under **day**, **night+IR
proxy**, and **day→night transition** conditions.

1. **Nobody died, but they degrade very differently.** OpenVINS (KLT
   front-end + online calibration) is nearly night-immune here: ATE 7.4 → 7.6 cm
   (+3%), at the price of **+55% compute**. Basalt degrades **2.6×**
   (10.8 → 28.5 cm) yet keeps full coverage — the IMU carries it through.
2. **The degradation signature matters more than the score.** OpenVINS *works
   harder* at night (slower, same accuracy). Basalt *silently does less* (3×
   faster at night = fewer features surviving its gates → weaker vision
   constraint). For a flying robot, the second failure mode is the dangerous
   one.
3. **The hybrid front-end case is now quantified**: day→night, ORB loses 69%
   of its verified matches and lands in tracking-failure territory (<30
   inliers) in 35% of frame pairs; **XFeat loses only 26% and never falls
   below 30 inliers** — on CPU. Descriptor-matching systems (the
   ORB-SLAM3/OpenMAVIS lineage) are the most exposed at night; a drop-in
   learned front-end removes most of that exposure.
4. Caveat up front: night here is a **photometric proxy** (camera-colocated
   IR flood + darkness + sensor noise). It does **not** include moving cast
   shadows — the Isaac lighting axis exists for that and still needs a ≥24 GB
   GPU host. These results are a lower bound on night difficulty.

## 1. What was actually run

**Data.** TUM-VI `room1` (512×512 fisheye stereo + 200 Hz IMU, ~146 m indoor
trajectory, mocap ground truth). The TUM mirror no longer serves the `.bag`
exports, so the EuRoC tar was converted with `bench/euroc2bag.py` (16-bit PNGs
mapped linearly to mono8, matching what cv_bridge did with the original bags).

**Conditions.** Three variants of the identical sequence:

| Condition | Generation |
|---|---|
| `day` | original imagery |
| `night` | photometric night+IR-flood proxy (`bench/degrade_bag.py`): image re-lit as reflectance × (4% ambient + gaussian beam vignette centered on the optical axis), plus high-gain sensor noise. Models the pod's camera-colocated 850 nm flood LED in darkness. |
| `transition` | day → night linear ramp over the middle 20% of the sequence (the "half day / half darkness" flight) |

| Day (frame 1500) | Night proxy (same frame) |
|---|---|
| ![day](assets/room1_day_frame1500.png) | ![night](assets/room1_night_frame1500.png) |

**Candidates.**

- **OpenVINS** (Track A, GPL yardstick) — stereo-inertial MSCKF, TUM-VI config,
  online camera-IMU calibration on, docker `3dfe/openvins`.
- **Basalt** (Track B, BSD product lane) — stereo-inertial optimization VIO,
  Double Sphere calibration, pinned pre-vcpkg commit `24acebe`, docker `3dfe/basalt`.
- **Hybrid front-end datapoint** (Track F2 evidence) — XFeat (Apache-2.0,
  CPU) vs ORB on identical frame pairs from the day and night bags:
  matching survival under the night+IR conditions.

**Metrics.** `bench/evaluate.py`: ATE (SE3-aligned RMSE), RPE@1s, coverage
(fraction of GT time span with estimates — the robustness number), dropout
gaps. Wall/CPU/RSS via in-container `timer_wrap.py`.

## 2. Results — VIO candidates × conditions

TUM-VI room1 (146 m, 141 s), stereo fisheye + IMU, mocap ground truth. ATE =
SE3-aligned RMSE; RPE@1s = median windowed drift after alignment; coverage =
fraction of the sequence with pose output.

| Candidate | Condition | ATE rmse [m] | ATE max [m] | RPE@1s [m] | Coverage | Wall [s] | CPU | RSS [MB] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenVINS | day | **0.074** | 0.146 | 0.018 | 0.95 | 100 | 129% | 102 |
| OpenVINS | night | **0.076** | 0.166 | 0.024 | 0.95 | 155 | 166% | 98 |
| OpenVINS | transition | **0.061** | 0.122 | 0.023 | 0.95 | 97 | 132% | 100 |
| Basalt | day | 0.108 | 0.268 | 0.020 | 1.00 | 54 | 1202% | 142 |
| Basalt | night | 0.285 | 0.575 | 0.042 | 1.00 | 16 | 1183% | 129 |
| Basalt | transition | 0.209 | 0.481 | 0.028 | 1.00 | 15 | 1360% | 144 |

Both day baselines match published TUM-VI room1 results (≈0.06–0.11 m), which
validates the pipeline end-to-end (EuRoC→bag conversion, configs, evaluator).
OpenVINS coverage is 0.95 in all conditions — that is its ~7 s initialization
window, not a tracking loss. Timing was measured on a shared, loaded machine
(a Gazebo sim ran throughout): treat wall/CPU as relative, not absolute;
Basalt's CPU% is high by design (TBB parallelism).

## 3. Results — hybrid front-end (XFeat vs ORB)

Identical 40 frame pairs (0.5 s temporal baseline) from cam0, day vs night,
RANSAC (fundamental-matrix) verification:

| Front-end | Condition | Matches | Inliers (mean) | Inlier ratio | Pairs < 30 inliers |
|---|---|---:|---:|---:|---:|
| ORB (2048) | day | 396 | 211 | 49.8% | 2/40 |
| **XFeat** (2048) | day | 890 | **396** | 41.9% | **0/40** |
| ORB (2048) | night | 204 | 66 | 29.0% | **14/40** |
| **XFeat** (2048) | night | 815 | **293** | 33.7% | **0/40** |

Day → night, ORB loses **69%** of its inliers and drops below 30 inliers
(tracking-failure territory for a feature-based VIO) in **35%** of the pairs.
XFeat loses only **26%** and never falls below 30 inliers. That is the
clearest quantitative argument so far for the hybrid lane (learned front-end
over a classical backend) under the pod's night+IR operating mode.

## 4. Interpretation

**Day.** OpenVINS is more accurate (0.074 vs 0.108 m) and lighter (1.3 vs ~12
cores); Basalt is 2× faster wall-clock by throwing cores at it. Both are
solid; this is the expected filter-vs-optimization picture.

**Night.** The interesting result is *how* each degrades:

- **OpenVINS** tracks KLT patches (photometric, no descriptors) and
  re-detects aggressively. Under beam-lit-center/dark-periphery imagery it
  keeps enough center features to stay at day-level accuracy — but wall time
  rises 55% and CPU 29%: the robustness is *bought with compute*. On an edge
  board that margin must exist. Its online calibration also stayed converged
  (fx within 0.5% of truth) despite the degraded imagery.
- **Basalt** got *3× faster* at night — the tell that its feature pipeline
  passes far fewer candidates, so the optimizer does less work and vision
  constrains the solution more weakly (ATE 2.6× worse, RPE 2×). It never
  reports a problem: coverage stays 1.0. **Silent degradation** is exactly the
  failure mode a flight stack must detect (feature-count / covariance
  monitoring, not pose-output monitoring).
- **Transition** ranks between the extremes for Basalt (0.209 m — the night
  half dominates the error) and is a non-event for OpenVINS (0.061 m).

**Hybrid (Track F2 evidence).** The front-end study isolates *why* night
hurts: binary-descriptor matching (ORB) collapses (69% inlier loss, 35% of
pairs in failure territory), while the learned detector-descriptor (XFeat)
barely notices (26% loss, zero failure pairs) — at CPU speeds compatible
with the pod. Implications:

1. The ORB-SLAM3/OpenMAVIS lineage (descriptor-based) carries the highest
   night risk of the candidate pool — to be confirmed when Track C runs.
2. A KLT-based classical core (OpenVINS-style) is already decent at night;
   the learned front-end's value concentrates in **relocalization/loop
   closure under illumination change** and **feature-starved scenes** —
   which this well-textured lab sequence does not probe.
3. For the pod product lane: cuVSLAM's night behavior is unknown (closed
   box) — measuring its degradation *signature* (does it work harder or do
   less?) matters as much as its scores.

## 5. Honest limitations

1. **The night condition is a photometric proxy**, not the Isaac lighting
   axis: the flood illumination and noise are modeled, but **moving cast
   shadows** (drone-mounted light sweeping the scene) are not — those need the
   GPU sim bring-up. Night results here are a *lower bound* on difficulty.
2. **Real data, wrong rig**: TUM-VI is a synchronized global-shutter stereo
   rig, not the pod triangle with unsynchronized rolling-shutter cameras. The
   camera-count and skew axes of the matrix still need the sim benchmark.
3. **Timing measured on a loaded shared machine** (a Gazebo sim was running
   throughout) — wall/CPU numbers are indicative, not benchmarks.
4. Single sequence (`room1`), single run per cell — no variance estimates.
5. cuVSLAM (Track F1 headline) not yet run: needs the ROS 2 pipeline; AirSLAM
   (GPL yardstick) needs a TensorRT build. Both are the next cells to fill.

## 6. Next steps

- Fill the cuVSLAM and AirSLAM rows (day/night) — the product-lane decision
  needs them.
- Isaac lighting-axis bring-up on a ≥24 GB GPU host → replace the proxy with
  real moving-shadow night sequences + pod rigs (2/3 cams) + skew axis.
- Wire XFeat into an OpenVINS-style front-end (Track F2) and rerun the night
  column — the front-end study predicts a large coverage gain.
