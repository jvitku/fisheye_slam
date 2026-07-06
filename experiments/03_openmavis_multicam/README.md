# Track C — OpenMAVIS (multi-camera visual-inertial SLAM, 3–6 cams)

**Goal:** answer the core architecture question — *does >2-camera surround
coverage measurably reduce tracking failures* (turns, low texture, occlusion)
versus the stereo/N-cam VIO tracks? OpenMAVIS is the only public system doing
true multi-camera VI-SLAM with loop closure (reimplementation of the
Hilti-2023-winning MAVIS, built on ORB-SLAM3).

## Status
- [ ] Docker image builds (`docker/openmavis/`) — expect the most friction here
- [ ] Reproduce upstream demo on a 4-cam sequence (Hilti 2023 or Newer College multi-cam)
- [ ] Adapt config to our 3-cam rig layout (`rigs/rig_3cam_example.yaml`)
- [ ] Robustness comparison vs Track A on identical degraded segments

## Notes / expectations
- GPL-3, ORB-SLAM3-class compute: **Orin-only** target, never Pi 5.
- Upstream is candid that it's a compact reimplementation: preprocessing and IMU
  intrinsic compensation from the paper are missing. Budget integration time.
- Judge on **robustness** (failure counts, relocalization behavior), not just ATE —
  that's the only reason to pay the multi-camera complexity tax.
- Build recipe: follow `candidates/OpenMAVIS/README.md` (ORB-SLAM3-style deps:
  Pangolin, OpenCV 4, Eigen, DBoW2/g2o vendored). Keep everything in the docker
  image; do not pollute the host.
