# podslam campaign — handoff notes (2026-08-31, updated same day on the desktop)

Written by the campaign agent for its future instance on a stronger machine.
**Desktop era has begun** (i7-13700K / 62 GB / RTX 3080): see "Desktop-era status"
at the bottom for what's already done here — the laptop-era notes below stand.
Everything below is committed; results live in `results/minisim/*/{ate.json,map_metrics.json}`,
history in `git log` (one commit per accepted/rejected experiment).

## Mission and standing rules (user-issued, all in force)

1. **Goal**: in-house multi-camera VIO/SLAM (`podslam/`) replacing closed cuVSLAM;
   iterate until **SOTA from a 3–6 cam setup in every aspect, "but also 10x better"**.
2. Run **as fast as possible**; target **realtime on Jetson class** (C++ port exists, `podslam-cpp/`, ~35 ms/frame single core, full parity validated).
3. **Easily adaptable to different cameras (SWIR, thermal) in sim** — lens/sensor models are per-rig yaml.
4. **Robustness is the main priority.**
5. **Resource guards on every heavy job** (CPU/GPU/RAM/HDD/temp) — `bench/guard.sh`; never endanger the host.
6. **Hourly summaries**: camera setup used + improvement vs baselines and previous version. HTML progress artifact: https://claude.ai/code/artifact/8881efbd-089a-4be2-9a76-22916c884e1a
7. **Commercial closed-source product**: every shipped dependency must be permissive
   (gtsam BSD, OpenCV Apache-2.0, rosbags Apache — clean). Ideas from GPL systems allowed, **never code**.
   Sim/tooling (Isaac Sim EULA, PX4 BSD, Blender GPL-as-tool) is fine as long as nothing links into the product.
8. **Build everything in Docker containers** — host must stay clean. *Current host-side debt to
   containerize on the new machine*: `~/tools/px4venv`, `~/tools/px4` (PX4 SITL build), `~/tools/isaacvenv`
   (Isaac Sim 4.5 pip), `~/tools/blender-4.2.0-linux-x64`. Existing images: `3dfe/cuvslam-ml` (CUDA torch,
   render+enhancer), `fisheye/gtsam-dev` (C++ toolchain), `3dfe/openvins`, `3dfe/podslam`, `ros-noetic-kalibr`.
9. **Smallest possible sim2real gap** — headline numbers should come from the most realistic lane
   available: Isaac RTX renders (native f-theta fisheye verified), PX4-flown trajectories (not analytic
   splines), measured sensor models (Allan IMU, Kalibr extrinsics, per-rig px_sigma). The analytic
   raycaster ("minisim") remains the exact-GT **regression anchor**, not the realism claim.
10. Renderer decisions (user): **Unreal DITCHED**; custom minisim continues; **Isaac Sim adopted**
    (proven working even on the 8 GB laptop via `pip install "isaacsim[all]==4.5.0"
    --extra-index-url https://pypi.nvidia.com`, `OMNI_KIT_ACCEPT_EULA=YES`, no NGC login).
11. **Dynamic scenes are a requirement** (moving trees, movers) — implemented, see below.

## The three rigs (user-specified, `rigs/`)

| rig | cameras | notes |
|---|---|---|
| `oakdpro.yaml` | 2× OV9282 1280×800 pinhole, 7.5 cm baseline, depth stream, 850 nm flood | OAK-D Pro-identical; device sim contract (color 20 Hz, depth 10 Hz z-depth 32FC1) |
| `skydio3.yaml` | 3× 200° KB4 512², coplanar top set, axes 40° off vertical @ az 0/120/240° + top flood | Skydio X10-style single trinocular plane |
| `skydio6.yaml` | top set + mirrored bottom set (az 60/180/300°), 2 floods | full X10 arrangement; **the winner rig** |

## Results (indoor scan flight 110 s & forest loop, minisim renders, per-camera frontend)

| sequence | ATE rmse | map median / inlier@10cm / completeness |
|---|---|---|
| **skydio6 · PX4-flown indoor (27 s, static init)** | **2.2 cm** | **4.7 cm / 69% / 39%** |
| oakdpro · indoor day / night | 9.2 / 15.7 cm | 4.5/91/79% · 7.2/61/81% (dense depth fusion) |
| skydio6 · indoor day / night+IR | 10.4 / 61.6 cm | 10.6/48/74% · 20.8/26/61% |
| skydio6 · indoor dyn (2 movers) / fog β0.16 / night+dust | 17.3 / 27.8 / 60.2 cm | 14.2/38/74 · 14.2/39/73 · 24/21/57% |
| skydio6 · forest day / night+IR / dyn (61 swaying trees) | 31.9 / 41.0 / 81.2 cm | 58/17/31 · 27.5/27/22 · 137/9/17% (@20cm) |
| skydio3 · indoor day / night | 24.1 / 120 cm | 3 up-cams = weak geometry; reference only |

Baselines: established table stands (TUM-VI room1 day 8.8 cm vs OpenVINS 7.4 / cuVSLAM 11.4 / Basalt 10.8;
Hilti exp14 8.3 vs OpenVINS 5.4). OpenVINS **on minisim**: fails to initialize on jerk-free C² starts
(needs accel jerk; with thresh 0.08 it inits mid-motion at t=3.6 s and diverges to km) — open item;
fly PX4 missions with real takeoff jerk for fair external baselines (podslam handles both).

## Key findings of the campaign (each = one commit)

1. **Frontend v2 (`multiklt`) is the breakthrough**: the classic KLT is stereo-centric (tracks only cam0,
   other cams are LK cross-match targets) → on divergent rigs cams 1..N contribute ZERO (diagnosed via
   per-cam obs logging: 300/0/0/0/0/0). `MultiKltFrontend` = independent per-camera KLT, disjoint id
   spaces (stride 1e8); tracker landmark loop generalized over all cams with a seen-set (stereo-centric
   rigs stay bit-identical). skydio6: day 23.1→10.4 cm, night 2.0 m→0.62 m. Use `--frontend multiklt`
   for divergent rigs; default `klt` for stereo rigs (OAK-D, TUM-VI, Hilti).
2. **C² trajectories or death**: any discontinuity in pose ramps (or piecewise-linear TUM interpolation
   differentiated at 1e-4!) becomes a 1e7 m/s² IMU spike → km-scale fake divergence. `scene.py` uses
   quintic smoothstep ramps; `render.py::synth_imu` uses fd_dt=0.012 for TUM trajectories. Verify any
   new trajectory with the max|gyro|/|accel| scan.
3. **Map eval needs gauge alignment**: maps are built in the estimator frame (first pose ≈ origin);
   `map_eval --est` runs SE3 Umeyama est→gt before scoring (was hiding OAK-D's true 4.5 cm map).
4. **Depth convention**: device depth = z-depth, not ray range (17% edge error at 80° HFOV).
5. **Dynamic scenes**: transient movers are near-free (indoor 10.4→17.3 cm; fwd/bwd+gates catch them);
   **coherent wind sway is the hard case** (forest 31.9→81.2 cm — passes χ² gate). Accepted mitigation:
   px_sigma 2.5 (IMU-anchoring) → dyn 0.50 m but static 0.32→0.50 m (trade-off; both ≈ 0.50). Proper fix
   queued: motion-aware adaptive weighting / landmark temporal-consistency check. REJECTED: lag 2.0 s (2.5 m — worse).
6. **Night**: indoors, floods light only the ceiling patch → weak geometry (61.6 cm vs 10.4 day);
   outdoors floods act as a **range filter** to parallax-rich near field → night ≈ day (41 vs 32 cm)
   and the night map is BETTER. night+propwash dust ≈ free (60.2 vs 61.6 cm). Learned TUM-VI night
   enhancer does NOT transfer to minisim (negative result, recorded).
7. **PX4 SITL works on-laptop** (SIH lockstep): real missions → GT + real dynamics. **2.2 cm ATE** on the
   flown window with static ground start. **Moving-platform init is the capability gap** (3.49 m when the
   window starts mid-flight) — build dynamic init (velocity/gravity from vision+IMU) on the strong PC.
   SIH altitude estimate converges slowly (true z overshoots to 4.6-5 m on a 1.7 m command early flight)
   — use late windows or wait for EKF convergence; trim tooling in `trajs/` + `px4_fly.py`.

## What to do first on the stronger PC (priority order)

1. **Containerize everything** (rule 8): images for px4-sitl, isaac-sim, minisim-render, podslam-run;
   compose or one driver script. Kill the host-venv debt.
2. **Isaac Sim at scale** (rule 9): port `bench/minisim/scene.py` prims → USD builder
   (pattern proven in scratchpad `isaac_fisheye.py`: `rep.create.camera(projection_type=
   "fisheye_polynomial", fisheye_max_fov=200, ...)` + Replicator rgb annotator); render the full matrix
   at 1024²+; PX4-flown trajectories; compare vs minisim numbers (sim2real gap estimate between lanes).
3. **Moving-platform initialization** (the 3.49 m gap) — mandatory for real drone deployments.
4. **Night-indoor iteration**: 1024² fisheye (fx 293) attacks the 10 mrad angular noise ceiling
   (rig yaml + render only — no code changes); exposure/gain model; NN night front-end (user allows ML).
5. **Adaptive dynamic-scene weighting** (finding 5) + landmark temporal-consistency gate.
6. **Mapping v2**: fuse depth at marginalization-time (refined) poses instead of newest-pose;
   fisheye semi-dense (grid-seeded cross-camera stereo); TSDF option for the product map.
7. **OpenVINS/Basalt/VINS baselines on identical bags** (PX4 trajectories with real jerk).
8. **C++ port catch-up**: port multiklt + mapping to podslam-cpp; Jetson/Orin build + timing (rule 2).
9. **Full matrix at scale**: 3 rigs × {indoor,forest} × {day,night,fog,dust,dyn…} × {analytic,PX4} with
   run_matrix.sh (env: FRONTEND=multiklt, TRAJ=...); overnight sweeps are trivial without the thermal grind.

## Operating notes (this laptop; general traps in ALL CAPS)

- Thermal: i9-13900HX rides 95–100 °C under any sustained load; guard profile `--temp-max 98 --temp-hold 6`
  (sustained rule; Tjmax 100 throttle is the backstop). ONE heavy job at a time. Renders grind in 1–5 min
  chunks with `until temp<63°C` waits between (see run_matrix.sh / the chain scripts in session scratchpad).
- GUARD KILLS LEAVE DOCKER ORPHANS: the scope kill hits the docker CLI, the container keeps running
  unguarded (twice kept the CPU at 98 °C; once quietly finished a whole render). `docker ps` +
  `docker kill` after every kill, or next guard start reaps by label `fisheye_guard=*`.
- PKILL SELF-MATCH: `pkill -f PATTERN` kills your own shell if PATTERN appears anywhere in its cmdline
  (heredocs included!). Exit 144. Use `-x` exact names or run kills from a cmdline that cannot match.
- BASH READS SCRIPTS INCREMENTALLY: never edit a script an active process is executing.
- PIPES MASK EXIT CODES: `cmd | tail` returns tail's rc; use `set -o pipefail` before && chains.
- PNG writes are atomic now (tmp+rename); pre-fix corrupt frames from kills were real (PIL-verify sweep).
- rosbags Writer partial files are unreadable ("Bag is not indexed") — check freshness AND validity.
- PX4: shallow clones need local git tags; keep px4 stdin open; venv on subprocess PATH; pkill -x
  mavsdk_server before connect; retry arm ~30×2 s. Full recipe in memory `project-minisim-px4-ops`.
- Isaac on 8 GB: guard `--mem 20G` (12G scope OOM-kills shader compile); first full init ~255 s, then fast.
- gtsam python: bool-bound smart-factor thresholds, EPI throws in LM — see memory `reference-gtsam-python-pitfalls`.

## Repo map

- `podslam/` — the product: tracker.py (orchestration), frontend/{klt,multiklt,xfeat}.py,
  backend_smart.py (smart rig factors, marginalisation, χ² gate), mapping.py (voxel map), track_bag.py (CLI).
- `podslam-cpp/` — validated C++ port (window parity 1.7e-9 m; needs multiklt port).
- `bench/minisim/` — scene.py (single source of truth: prims/trajectories/conditions/dust/dynamics +
  exact distance_to_surface), render.py (GPU raycaster), blender_scene.py (Cycles lane),
  px4_fly.py (SIH missions), flightbag.py, map_eval.py, run_render.sh, run_matrix.sh.
- `bench/` — guard.sh, evaluate.py (ATE/RPE), frames2bag.py, gen_openvins_config.py (+env
  OV_INIT_IMU_THRESH/OV_INIT_MAX_DISP), run_sim_candidate.sh (OpenVINS docker), enhance_bag.py.
- `rigs/` — the three rigs + tumvi/hilti benchmark rigs.
- `results/minisim/<seq>/` — est.tum, ate.json, map.npz/.ply, map_metrics.json per experiment.
- `trajs/` — PX4-flown trajectories (indoor_px4b.tum + _trim).
- Memory (agent-side): project-minisim-px4-ops, project-dev-machine-limits,
  project-podslam-estimator-lessons, reference-gtsam-python-pitfalls, project-campaign-rules.


## Desktop-era status (2026-08-31 evening, this machine)

Done today (each a commit, all pushed):
1. **Containerized pipeline**: one lean image `fisheye/podslam` (6.5 GB, docker/podslam,
   pytorch cu126 base + pyproject deps). run_matrix.sh / run_render.sh fully dockerised
   (host uv debt gone), map_eval gets `--est` (finding 3), GUARD_DOCKER_ARGS expansion
   deferred into the guarded child (the laptop's unkillable-orphan bug — fixed).
   Env baseline: px4b window 2.79 cm here vs 2.22 laptop (lib drift; per-machine A/B only).
2. **Moving-platform init LANDED** (priority 3): podslam/init_dynamic.py.
   Mid-flight PX4 window 13.38 m -> 0.101 m (auto), map 784->8.7 cm median.
   Rotation-warp cross-cam matching (raw LK 1.2% -> warped ~90-200 pts/frame @ 37 ms).
   `--init-mode static|auto|dynamic`; static default bit-exact (verified); auto on
   static start 2.25 vs 2.79 cm (track warm-up helps). 8 solver unit tests.
   Candidate: flip default to auto after full-matrix A/B.
3. **Dynamic-scene weighting ACCEPTED** (priority 5): `--dyn-weight` wins every
   sequence tried — forest dyn 9.41 -> 0.416 m (22.6x), forest static 0.398 -> 0.324,
   px4b2 2.79 -> 2.60 cm, indoor day 0.228 -> 0.165. px2.5 stopgap obsolete (1.367 dyn
   / 0.560 static on the same bags). --px-adapt-up kept as secondary (0.467 dyn, exactly
   neutral static). Default OFF until a TUM-VI/Hilti no-harm pass (datasets not on this
   box). Also: multiklt + circle masks + generalized landmark loop ported to podslam-cpp
   (C++ 1.96 vs Py 2.85 cm on the px4b window slice; 78 ms/frame for 6 cams single-core;
   build_deps.sh = pinned GTSAM 4.3a2 + OpenCV 5.0.0 recipe). px4c windows: takeoff
   3.16 cm; mid-flight static 0.094 vs auto 0.090 (auto never worse anywhere).
   NEW BACKLOG (robustness): analytic indoor-day early-window fragility — ±2 features
   swings ATE {0.221, 0.228, 0.733}, bad from the first 30 s; fix = delayed/quality-gated
   gauge anchoring. Ops: background A/B tasks were externally killed 4x -> run long
   sweeps in foreground chunks; kill only by full guard-id label, never bare
   fisheye_guard; pkill -f self-match hit us again (use -x).
4. **px4-sitl container** (docker/px4-sitl, PX4 v1.15.4 SIH prebuilt). Traps hit and
   fixed in the Dockerfile: shallow NuttX needs a local nuttx-* tag; pip needs pyyaml.
   bench/minisim/trim_traj.py replaces the lost ad-hoc window trimming.
5. **skydio6hd** rig (1024^2, fx 293.4) — night angular-noise attack in progress.

Machine notes: disk is the scarce resource (shared box; arrived 100% full, safe docker
prunes freed ~42 G; ~44 G free at last check, guard floor 15 G). Big levers (423 GB
docker volumes of other projects, 133 GB Downloads) are the USER's call — asked.
Thermal is a non-issue (guarded peaks ~61 C, temp-max 97 kept). Isaac at scale
(priority 2) still needs the disk decision; OpenVINS baseline on px4 bags is next
(docker/openvins build + containerised run_sim_candidate.sh ready).
