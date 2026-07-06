# Multi-Fisheye + IMU SLAM — Evaluation Plan & Rationale

*Project: `3d_fisheye` — exploratory evaluation of open-source SLAM/VIO for 2–3 (later 6)
cheap rolling-shutter fisheye cameras + IMU, targeting Pi 5 / Jetson Orin Nano Super.*

---

## 1. What we are optimizing for

From the research review, the requirements ranked by how much they constrain the choice:

1. **Robustness first** — recovery from visual degradation matters more than benchmark accuracy.
2. **2–3 cameras now, 6 later** — the architecture must not be hard-wired to stereo.
3. **Cheap, rolling-shutter, unsynchronized cameras** — online time-offset estimation and
   tolerance to imperfect sync is a hard requirement, not a nice-to-have.
4. **Real-time on edge** (Pi 5 → Orin Nano Super) — bounded compute; filter-based or
   fixed-budget optimization back-ends preferred.
5. **Dense local 3D map** — navigational TSDF/ESDF, not photogrammetry.
6. **IR/NoIR + active NIR illumination** — favors front-ends robust to illumination change
   (learned features help here).

Key conclusion of the review, which this plan adopts: **no single open-source system does all
of this**. So we evaluate a small portfolio of candidates as *modules* and converge on a
modular stack: classical bounded VIO core + surgical learned components + separate local
dense mapper.

## 2. Candidate shortlist and decision

Systems considered and **rejected for now**:

| System | Why not (for the first round) |
|---|---|
| ORB-SLAM3 | Mature but stereo/mono-centric, GPL, heavy back-end; covered indirectly via OpenMAVIS which is built on it. |
| VINS-Fisheye | Stereo-only for fisheye, aging deps (TX2-era), no path to 3+ cams. Kept as a Jetson fallback only. |
| BAMF-SLAM | Closest conceptual match, **but no open-source release**. We test its nearest open relative instead (DBA-Fusion). |
| Sphere-VIO | Best long-term architecture on paper; too new/immature. Re-check the repo situation in ~3 months; treat as design template. |
| NICE-SLAM / Co-SLAM / neural implicit | Offline/desktop only; irrelevant to edge flight loop. |

**Selected for exploratory implementation — four tracks:**

### Track A — OpenVINS (primary candidate)
*MSCKF filter VIO, BSD-ish (GPL-3 for core), github.com/rpng/open_vins*

- **Why:** the only mature open system that natively supports **N cameras** in its config
  (`max_cameras`), **online camera–IMU time-offset calibration** (`calib_cam_timeoffset`) and
  online intrinsics/extrinsics refinement — exactly the knobs an *unsynchronized* cheap-camera
  rig needs. Filter-based ⇒ bounded compute ⇒ best Pi 5 / Orin fit. Fisheye via
  equidistant/Kannala-Brandt model.
- **Role:** the default VIO core we expect to ship. Also the natural host for injecting a
  learned front-end later (its tracker is a pluggable KLT/descriptor front-end).
- **What we test:** 2-cam TUM-VI baseline → 3-cam on our own rig recording → time-offset
  robustness (artificially skew timestamps, measure degradation).

### Track B — Basalt (calibration backbone + lean stereo baseline)
*Optimization VIO + full calibration toolchain, BSD, gitlab.com/VladyslavUsenko/basalt*

- **Why:** best-in-class **Double Sphere** fisheye model + the calibration tools
  (`basalt_calibrate`, `basalt_calibrate_imu`) we will need regardless of which VIO wins.
  Very CPU-efficient — the most realistic Pi 5 stereo candidate.
- **Role:** calibration pipeline for *all* tracks + the lowest-compute 2-cam fallback.
- **What we test:** TUM-VI stereo baseline, calibration of our own cameras, CPU usage vs OpenVINS.

### Track C — OpenMAVIS (multi-camera SLAM, 3–6 cams)
*Multi-camera ORB-SLAM3 reimplementation of the Hilti-2023-winning MAVIS, GPL-3, github.com/MAVIS-SLAM/OpenMAVIS*

- **Why:** the only public system today that does **true multi-camera (>2) visual-inertial
  SLAM** with loop closure. This is the "does more-than-stereo actually help robustness?"
  experiment.
- **Risks:** compact reimplementation (not the original Hilti code), missing preprocessing/IMU
  intrinsic compensation, ORB-SLAM3-class compute (Orin-only, not Pi).
- **What we test:** 4-cam sequences (Hilti / Newer College multi-cam), then our 3-cam rig.
  Judge robustness in degraded segments vs Track A, not just ATE.

### Track D — Hybrid / learned components
1. **XFeat** (`verlab/accelerated_features`, Apache-2.0) — CPU-real-time learned features.
   Role: front-end replacement where KLT/ORB starves (low light, active-IR imagery).
2. **LightGlue** (`cvg/LightGlue`, Apache-2.0) — keyframe-only matcher for loop
   candidates/relocalization (too slow for every frame on edge CPU).
3. **DBA-Fusion** (`GREAT-WHU/DBA-Fusion`) — DROID-SLAM-style recurrent dense bundle
   adjustment tightly fused with IMU via GTSAM factor graph. This is the **closest
   open-source analog to BAMF-SLAM**. Desktop-GPU only for now; we evaluate it as the
   "hybrid ceiling" — how much robustness a learned dense front-end buys — to decide if a
   distilled/trimmed variant is worth pursuing on Orin.

- **What we test:** XFeat vs ORB/KLT matching quality on fisheye + synthetic low-light/NIR-ish
  degradation (`experiments/04_xfeat_lightglue/`); DBA-Fusion on TUM-VI on the desktop GPU.

### Track F — Learned/hybrid SLAM on edge (added 2026-07-06)
*The Skydio-style candidate: learned SLAM as a real runtime contender, not just a ceiling.*

1. **MASt3R-SLAM** (`rmurai0610/MASt3R-SLAM`) — real-time dense monocular SLAM on
   MASt3R two-view pointmap priors. Dense output doubles as the pod map; strong in
   low-texture/low-light — the natural fit for the night+IR axis. Risks: 4090-class
   paper runtime (edge gate = TensorRT on **Orin NX 16GB**, ≥5 keyframe-Hz or it drops
   to offline-refiner role), monocular/no-IMU, and **CC BY-NC weights** (fine for
   benchmarking, a product blocker unless retrained).
2. **DPVO** (`princeton-vl/DPVO`) — deep patch VO, the DROID lineage made light; the
   realistic **Orin Nano Super** runtime candidate. No IMU by default → evaluated as
   VO first; EKF fusion with the pod IMU if it wins the night rows.

Track F's decisive experiment: night+IR sequences vs Tracks A/B/C on identical bags.
If the classical stacks hold up under moving IR shadows, we stay classical; if they
collapse and Track F doesn't, the hybrid path gets promoted. Details:
`experiments/07_learned_slam/`.

### Dense mapping (deferred, decided in principle)
Not part of round 1. Decision already clear from the review: **voxblox** (CPU/Pi) or
**nvblox** (Orin) TSDF/ESDF fed by depth from selected adjacent fisheye pairs. It plugs into
whichever VIO wins, so evaluating it now would be premature.

## 3. Why this portfolio is the robust choice

- **A vs B** gives us filter-vs-optimization on identical 2-cam data — the classic
  robustness/compute trade — with either one a shippable Pi 5 fallback.
- **C** answers the only question that justifies >2 cameras: does surround coverage measurably
  reduce tracking failures in turns/low-texture, given our compute budget?
- **D** bounds the value of learning: XFeat is the cheap incremental win (drop-in front-end);
  DBA-Fusion is the expensive ceiling. If DBA-Fusion is not *dramatically* more robust than
  OpenVINS+XFeat, we stay classical and save months.
- Everything chosen is **open source and buildable today** (all repos verified reachable at
  project creation).

The expected end-state (matching the review's recommendation):
**OpenVINS-derived multi-cam VIO + XFeat-augmented front-end + adjacent-pair depth + nvblox on
Orin Nano Super, 3 cameras first.** The evaluation exists to confirm/refute this cheaply.

## 4. Plan

**Phase 0 — Scaffolding (this commit)**
- Repo layout, clone scripts, Dockerfiles per track, dataset download scripts.
- `tools/fisheye`: Python implementations of the three lens models that matter
  (Kannala-Brandt-4, Double Sphere, EUCM) with round-trip unit tests. These are the shared
  geometry vocabulary for calibration sanity checks, rig simulation, and later depth work.
- Example 3-cam rig definition (`rigs/`).

**Phase 0.5 — Unified simulation benchmark (added 2026-07-06)**
- Isaac Sim + Pegasus setup copied from `swarm_stack/tools/isaac` into `sim/isaac/`;
  `bench_drone.py` spawns an N-fisheye rig on the drone from `rigs/rig_{2,3,6}cam.yaml`
  (identical intrinsics across rigs — **camera count is the only variable**), rendered
  with exact f-theta = ideal kb4 intrinsics (calibration error eliminated as a variable).
- One recording per rig/trajectory (cams + IMU + `/uav1/ground_truth`);
  **unsynchronized-camera variants are generated offline** by `bench/skew_bag.py`
  (deterministic per-camera offsets/jitter) so every candidate sees byte-identical
  images under controlled timing degradation.
- `bench/evaluate.py`: ATE, RPE@1s, coverage and dropout gaps (robustness first-class).
- Matrix: candidates × rigs {2,3,6} × trajectories × **lighting {day, night+IR,
  half-transition}** × skew profiles — see `bench/README.md`.
- **Lighting axis** (added 2026-07-06): Isaac's RTX lights are scriptable
  (`sim/isaac/workspace/lighting.py`, `SIM_LIGHTING=day|night|half`). `night` kills the
  scene lights; `half` lights one half of the scene so the trajectory crosses a
  lit→dark boundary. The pod's NoIR cameras + **850 nm IR flood LED at the triangle
  center** (shadow-casting RTX cone light rigidly attached to the drone) then become
  the dominant illumination — enabling the cameras while creating the moving-shadow /
  non-constant-illumination conditions that make night VIO hard. This is the benchmark
  axis Track F exists for.
- **Sensor pod** (added 2026-07-06): the target product is modeled explicitly — a
  self-contained unit of N coplanar, same-direction fisheye cameras + its own PX4-FC
  IMU (`/uav1/sensor_pod/imu`, simulated with rigid-offset physics), mounted on top of
  the drone which keeps its own flight FC. Variants: `pod_2cam` (stereo pair) and
  `pod_3cam_triangle` (equilateral triangle → adds vertical/diagonal baselines for
  depth on horizontal structure). Pod output = position + dense TSDF map; the mapping
  stage (voxblox now, nvblox on Orin) is scaffolded in `experiments/06_dense_mapping/`.

**Phase 1 — Baselines on public data + sim bring-up (desktop, ~1–2 weeks)**
- Build Tracks A–C in Docker; run TUM-VI `room1/room4` (A, B, D) and a multi-cam sequence (C).
- Bring up the sim benchmark: verify fisheye rendering/orientation, script benchmark
  trajectories, add GT extraction + rig-yaml→candidate-config generation.
- Metrics: `bench/evaluate.py` (+ `evo` cross-check), CPU load, peak RSS.
- Deliverable: comparison table + per-track "gotchas" notes.

**Phase 2 — Own rig, 2–3 cameras (~2–3 weeks)**
- 2× (then 3×) NoIR wide/fisheye cams + IMU; locked exposure/gain; best-effort software
  timestamping. Calibrate with Basalt/Kalibr (intrinsics → extrinsics → cam-IMU temporal).
- Record a dataset (normal / fast yaw / low light + active IR) and replay through A, B, C.
- Key experiment: **timestamp-skew sensitivity** — this decides how much sync hardware we need.

**Phase 3 — Hybridization + edge deployment (~3–4 weeks)**
- XFeat front-end into the winning VIO; keyframe LightGlue relocalization.
- Cross-compile / deploy on Orin Nano Super (and Pi 5 for the 2-cam config); measure real-time
  envelopes.
- Start dense mapping track (nvblox + adjacent-pair depth).

**Go/no-go gates:** after Phase 1 drop any track that fails to build or is grossly
outperformed; after Phase 2 pick ONE core VIO; Phase 3 only hybridizes the winner.

## 5. Risks / open issues

- **Rolling shutter is unmodeled** in all Track A–C candidates. Mitigation: short exposure,
  moderate motion during eval; revisit continuous-time RS-aware methods only if Phase 2 shows
  RS as the dominant error source.
- **OpenMAVIS integration debt** — it's a reimplementation; expect config archaeology.
- **Unsynchronized 3-cam capture on Pi 5** is physically constrained (2 MIPI ports) — Phase 2
  recording likely happens on Orin or a desktop capture rig; that's fine for evaluation.
- **DBA-Fusion GPU memory** — DROID-style backends are VRAM-hungry; desktop-only, by design.
- Dockerfiles in this scaffold are written from upstream docs and **not yet built/tested**;
  Phase 1 starts by making them build.

## 6. Repo map

```
3d_fisheye/
├── docs/REPORT.md              ← this file
├── candidates/clone.sh         ← shallow-clones all upstream repos into candidates/
├── docker/                     ← one Dockerfile per track (A: openvins, B: basalt,
│                                  C: openmavis, D: hybrid python/torch)
├── datasets/download_tumvi.sh  ← TUM-VI calibrated sequences
├── sim/isaac/                  ← Isaac Sim + Pegasus (copied from swarm_stack) +
│                                  bench_drone.py / fisheye_rig.py rig support
├── bench/                      ← unified benchmark: record.sh, skew_bag.py,
│                                  evaluate.py (+ tests); contract in README.md
├── rigs/                       ← rig_{2,3,6}cam.yaml benchmark rigs (single source
│                                  of truth) + rig_3cam_real_example.yaml (hardware)
├── tools/fisheye/              ← lens models (KB4 / Double Sphere / EUCM) + tests
└── experiments/
    ├── 01_openvins_tumvi/      ← Track A runner
    ├── 02_basalt_tumvi/        ← Track B runner
    ├── 03_openmavis_multicam/  ← Track C notes/runner
    ├── 04_xfeat_lightglue/     ← Track D: fisheye matching demo + degradation eval
    └── 05_dba_fusion/          ← Track D: hybrid ceiling (desktop GPU)
```
