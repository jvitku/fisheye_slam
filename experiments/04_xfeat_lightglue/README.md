# Track D1 — XFeat (+ LightGlue) learned front-end on fisheye

**Goal:** quantify what a cheap learned front-end buys us on *fisheye* imagery
and under *low-light / active-NIR-like* degradation, before wiring it into the
winning VIO. XFeat is the every-frame candidate (CPU real-time); LightGlue is
keyframe-only (loop candidates / relocalization).

## Status
- [ ] `match_demo.py` runs on a TUM-VI image pair (visual sanity check)
- [ ] Degradation sweep: darken/blur/noise ramp, matching-survival curves
      XFeat vs ORB (`degradation_eval.py`, to be written)
- [ ] Same sweep on real NoIR + active-IR captures (Phase 2 data)
- [ ] Latency on target: XFeat CPU (Pi 5 class) and TensorRT (Orin)

## Run

```bash
# needs a GPU-less torch install; run in docker/hybrid or a local venv
uv run --with torch --with opencv-python --with kornia \
    python experiments/04_xfeat_lightglue/match_demo.py img0.png img1.png out.png
```

Note: `match_demo.py` pulls XFeat weights via torch.hub on first run (network
required). Untested scaffold — first item on the Phase 1 checklist.
