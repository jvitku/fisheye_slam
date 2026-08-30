#!/usr/bin/env python3
"""Where does the whole-flight error come from?  Align a run on its first 20 s only,
then report the growth of position / yaw / tilt / scale error over the flight.

    PYTHONPATH=. uv run python bench/drift_analysis.py podslam_smart_day podslam_smart_night [...]

Prints per-run: yaw drift (deg) and position error (cm) at 20-s marks, the yaw-drift
rate (deg/min), and the path-length ratio est/gt per 30-s window (scale)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.evaluate import associate, load_tum, quat_to_rot, umeyama  # noqa: E402


def yaw_of(R: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))


def analyze(gt: np.ndarray, est: np.ndarray, align_s: float = 20.0) -> dict:
    gi, ei = associate(gt[:, 0], est[:, 0], 0.02)
    tt = est[ei, 0] - gt[0, 0]
    sel = tt < align_s
    s, R, t = umeyama(est[ei][sel, 1:4], gt[gi][sel, 1:4], with_scale=False)
    al = (R @ est[ei, 1:4].T).T + t
    pe = np.linalg.norm(al - gt[gi, 1:4], axis=1)
    Re = np.einsum("ij,njk->nik", R, quat_to_rot(est[ei, 4:8]))
    Rg = quat_to_rot(gt[gi, 4:8])
    Rrel = np.einsum("nij,nkj->nik", Re, Rg)          # est * gt^T, in the world frame
    ang = np.degrees(np.arccos(np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    yaw = yaw_of(Rrel)
    yaw = np.degrees(np.unwrap(np.radians(yaw)))
    # tilt = rotation of the gravity axis
    tilt = np.degrees(np.arccos(np.clip(Rrel[:, 2, 2], -1, 1)))
    marks = np.arange(20, tt.max() + 1e-9, 20)
    rows = []
    for m in marks:
        w = (tt > m - 5) & (tt <= m)
        if not np.any(w):
            continue
        rows.append((m, np.median(yaw[w]), np.median(tilt[w]), np.sqrt(np.mean(pe[w] ** 2)) * 100))
    # yaw drift rate by a linear fit over the flight
    A = np.stack([tt, np.ones_like(tt)], 1)
    rate = np.linalg.lstsq(A, yaw, rcond=None)[0][0] * 60.0
    # scale: est/gt path length per 30 s window
    scales = []
    for w0 in np.arange(0, tt.max(), 30):
        w = (tt >= w0) & (tt < w0 + 30)
        if w.sum() < 20:
            continue
        pg = np.sum(np.linalg.norm(np.diff(gt[gi][w, 1:4], axis=0), axis=1))
        pe_ = np.sum(np.linalg.norm(np.diff(al[w], axis=0), axis=1))
        scales.append((w0, pe_ / pg if pg > 0 else np.nan))
    return {"marks": rows, "yaw_rate_deg_min": rate, "scales": scales, "ang_med": float(np.median(ang))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gt", default="bench/results/room1/gt.txt")
    ap.add_argument("--dir", default="bench/results/room1_sweep")
    a = ap.parse_args()
    gt = load_tum(a.gt)
    for r in a.runs:
        p = Path(a.dir) / r / "est.tum"
        if not p.exists():
            p = Path(a.dir) / r / "trajectory.tum"
        res = analyze(gt, load_tum(str(p)))
        print(f"== {r}: yaw drift {res['yaw_rate_deg_min']:+.2f} deg/min, median rot residual {res['ang_med']:.2f} deg")
        print("   t[s]   yaw[deg]  tilt[deg]  pos[cm]   (aligned on first 20 s)")
        for m, y, tl, pcm in res["marks"]:
            print(f"   {m:5.0f}   {y:+7.2f}   {tl:6.2f}   {pcm:7.1f}")
        print("   scale est/gt per 30 s:", " ".join(f"{s:.3f}" for _, s in res["scales"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
