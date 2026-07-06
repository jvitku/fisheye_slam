# Track D2 — DBA-Fusion (hybrid ceiling, desktop GPU only)

**Goal:** measure the robustness ceiling of the BAMF-SLAM-style hybrid approach
using its closest open-source relative: **DBA-Fusion** (GREAT-WHU) — DROID-SLAM
recurrent dense bundle adjustment tightly fused with IMU through a GTSAM factor
graph.

BAMF-SLAM itself (the review's closest conceptual match to our end goal) has no
open release; DBA-Fusion is the deployable stand-in for evaluating the concept.

## Status
- [ ] Environment builds (`docker/hybrid/`, CUDA desktop)
- [ ] TUM-VI room sequence runs end-to-end
- [ ] Side-by-side vs Track A on the *hardest* Phase 2 segments (fast yaw, low light)
- [ ] Decision memo: is the robustness delta worth pursuing a trimmed/distilled
      variant for Orin? (VRAM + latency numbers included)

## Notes
- Expect heavy VRAM use (DROID-style dense BA). Desktop GPU only — this track is
  intentionally *not* edge-deployable; it exists to bound what learning buys.
- Monocular + IMU by design; do not compare trajectory accuracy against stereo
  tracks directly — compare *failure behavior* on degraded segments.
- Build/run recipe: `candidates/DBA-Fusion/README.md`.
