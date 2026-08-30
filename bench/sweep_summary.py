"""Summarize cuVSLAM sweep runs (bench/results/room1_sweep) against the room1 ground truth.

Per run: ATE rmse, RPE@1s median/p95, RMSE of the first 20 s and last 20 s
(after one global SE3 alignment — the early transient and the late drift
are the two failure signatures on room1), coverage, tracked frames, mean
observations per camera, ms/frame, wall time.

Usage: python -m bench.sweep_summary [--json out.json] [--dir bench/results/room1_sweep]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from bench.evaluate import associate, load_tum, quat_to_rot, umeyama


def analyze(gt: np.ndarray, est_path: Path) -> dict:
    est = load_tum(str(est_path))
    gi, ei = associate(gt[:, 0], est[:, 0], 0.02)
    if len(gi) < 10:
        return {"status": "too few associations"}
    s, R, t = umeyama(est[ei, 1:4], gt[gi, 1:4], with_scale=False)
    al = (s * (R @ est[ei, 1:4].T)).T + t
    e = np.linalg.norm(al - gt[gi, 1:4], axis=1)
    tt = est[ei, 0] - gt[0, 0]
    step = max(1, int(round(1.0 / np.median(np.diff(est[ei, 0])))))
    rpe = np.array([np.linalg.norm((al[i + step] - al[i]) - (gt[gi[i + step], 1:4] - gt[gi[i], 1:4]))
                    for i in range(0, len(ei) - step, max(1, step // 2))])
    span = gt[-1, 0] - gt[0, 0]
    cov = float(np.clip((est[-1, 0] - est[0, 0]) / span, 0, 1))
    # orientation residual after the (position-based) global alignment: a large
    # value early on = the estimator's frame was still tilting/yawing (init)
    Re = quat_to_rot(est[ei, 4:8]); Rg = quat_to_rot(gt[gi, 4:8])
    Ra = np.einsum("ij,njk->nik", R, Re)
    Rrel = np.einsum("nij,nkj->nik", Ra, Rg)
    ang = np.degrees(np.arccos(np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return {
        "status": "ok",
        "rot_first20_deg": float(np.median(ang[tt < 20])) if np.any(tt < 20) else None,
        "rot_med_deg": float(np.median(ang)),
        "ate_cm": float(np.sqrt(np.mean(e ** 2)) * 100),
        "ate_max_cm": float(e.max() * 100),
        "first20_cm": float(np.sqrt(np.mean(e[tt < 20] ** 2)) * 100) if np.any(tt < 20) else None,
        "last20_cm": float(np.sqrt(np.mean(e[tt > tt.max() - 20] ** 2)) * 100),
        "rpe_med_cm": float(np.median(rpe) * 100),
        "rpe_p95_cm": float(np.percentile(rpe, 95) * 100),
        "rpe_max_cm": float(rpe.max() * 100),
        "coverage": cov,
        "poses": int(len(est)),
    }


def stats(run_dir: Path) -> dict:
    out = {}
    f = run_dir / "frames.csv"
    if f.exists():
        rows = np.genfromtxt(f, delimiter=",", names=True)
        if rows.size:
            out["tracked_frac"] = float(np.mean(rows["tracked"]))
            out["obs0_mean"] = float(np.mean(rows["n_obs0"][rows["n_obs0"] >= 0])) if np.any(rows["n_obs0"] >= 0) else None
            out["obs0_p05"] = float(np.percentile(rows["n_obs0"][rows["n_obs0"] >= 0], 5)) if np.any(rows["n_obs0"] >= 0) else None
            out["ms_mean"] = float(np.mean(rows["ms"]))
    log = run_dir / "run.log"
    if log.exists():
        m = re.search(r"wall_s=([\d.]+)", log.read_text(errors="replace"))
        if m:
            out["wall_s"] = float(m.group(1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="bench/results/room1_sweep")
    ap.add_argument("--gt", default="bench/results/room1/gt.txt")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    gt = load_tum(args.gt)
    rows = {}
    for d in sorted(Path(args.dir).iterdir()):
        if not d.is_dir():
            continue
        try:
            r = analyze(gt, d / "est.tum") if (d / "est.tum").exists() and (d / "est.tum").stat().st_size > 0 else {"status": "no trajectory"}
            r.update(stats(d))
        except Exception as e:                       # a truncated run must not hide the others
            r = {"status": f"error: {type(e).__name__}"}
        rows[d.name] = r
    f = lambda v, p=1: "—" if v is None else f"{v:.{p}f}"
    print("| run | ATE | max | first20 | last20 | rot20° | RPE med | RPE p95 | cov | tracked | obs0 | obs0 p5 | ms/frame | wall |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in rows.items():
        print(f"| {name} | {f(r.get('ate_cm'))} | {f(r.get('ate_max_cm'), 0)} | {f(r.get('first20_cm'))} | {f(r.get('last20_cm'))} | {f(r.get('rot_first20_deg'))} "
              f"| {f(r.get('rpe_med_cm'), 2)} | {f(r.get('rpe_p95_cm'))} | {f(r.get('coverage'), 2)} | {f(r.get('tracked_frac'), 3)} "
              f"| {f(r.get('obs0_mean'), 0)} | {f(r.get('obs0_p05'), 0)} | {f(r.get('ms_mean'))} | {f(r.get('wall_s'), 0)} |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
