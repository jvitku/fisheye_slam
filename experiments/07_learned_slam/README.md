# Track F — Learned/hybrid SLAM on edge (the Skydio-style candidate)

**Goal:** evaluate learned-SLAM systems as *real candidates* (not just a
ceiling like Track D2), with a hard deployability gate: must realistically run
on **Jetson Orin Nano Super**, or **Orin NX 16GB** at worst. Learned front-ends
are also the natural fit for the night/IR benchmark axis — classical
photometric tracking suffers most under the pod's moving IR illumination and
its moving shadows.

## Candidates

### MASt3R-SLAM (`rmurai0610/MASt3R-SLAM`)
Real-time dense monocular SLAM on MASt3R two-view 3D-reconstruction priors.
Attractive: dense pointmaps *are* the pod's map output; strong robustness to
low texture/illumination; works without precise calibration (relevant to
cheap fisheye).
- **Reality check:** paper real-time is on a desktop 4090-class GPU with a
  ViT-Large backbone; monocular RGB, no IMU fusion, pinhole-centric.
- **License flag:** MASt3R weights are CC BY-NC (non-commercial) — fine for
  benchmarking, a blocker for the product unless relicensed/retrained.
- **Edge gate:** TensorRT/INT8 export of the MASt3R encoder at reduced
  resolution on Orin NX 16GB. If it cannot hold ≥5 keyframe-Hz there, it is
  out as a runtime candidate (may survive as an offline map refiner).

### DPVO (`princeton-vl/DPVO`)
Deep patch visual odometry (sparse learned patches + recurrent update; the
DROID-SLAM lineage made light). Far smaller than MASt3R — the realistic
**Orin Nano Super** candidate; DPV-SLAM adds loop closure.
- No IMU by default → evaluate as VO; IMU fusion (e.g. feeding its poses into
  a light EKF with the pod IMU) is a Phase-3 integration if it wins the
  night-mode comparison.

## What we test

- [ ] Desktop baselines on TUM-VI + sim day sequences (accuracy sanity)
- [ ] **Night + IR sequences** (`SIM_LIGHTING=night`, pod illuminator on):
      compare vs Tracks A/B/C on identical bags — this is Track F's
      raison d'être. Also the `half` lit→dark transition.
- [ ] Fisheye handling: native (distorted) vs rectified-to-pinhole inputs
- [ ] Orin feasibility: TensorRT export, resolution/precision sweeps,
      fps + VRAM on Orin Nano Super (DPVO) and Orin NX (MASt3R-SLAM)

## Notes
- Both consume the same benchmark bags (mono conversion of the NoIR/IR
  imagery at input). Evaluate with `bench/evaluate.py --scale` (monocular).
- Environment: `docker/hybrid/` image covers both (PyTorch + CUDA); DPVO needs
  its CUDA extensions compiled in-image; MASt3R-SLAM pulls its own checkpoints.
