#!/usr/bin/env python3
"""Summarise every run under bench/results/hilti2022 against the dense IMU-frame ground
truth: SE3 ATE, Sim3 ATE + scale, coverage.  Markdown table + summary.json.

    PYTHONPATH=. uv run python bench/hilti_summary.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.evaluate import associate, load_tum, umeyama  # noqa: E402


def ate(gt, est, with_scale):
    gi, ei = associate(gt[:, 0], est[:, 0], 0.02)
    if len(gi) < 10:
        return None
    s, R, t = umeyama(est[ei, 1:4], gt[gi, 1:4], with_scale=with_scale)
    al = (s * (R @ est[ei, 1:4].T)).T + t
    e = np.linalg.norm(al - gt[gi, 1:4], axis=1)
    cov = float(np.clip((est[-1, 0] - est[0, 0]) / (gt[-1, 0] - gt[0, 0]), 0, 1))
    return {"rmse_cm": float(np.sqrt(np.mean(e ** 2)) * 100), "max_cm": float(e.max() * 100), "scale": float(s), "coverage": cov}


def main() -> int:
    root = Path("bench/results/hilti2022"); gts = {}
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"(.+?)_(exp\d+_.+)$", d.name)
        if not m:
            continue
        name, seq = m.group(1), m.group(2)
        gt_path = Path("datasets/data/hilti2022/ground_truth") / f"{seq}_imu.txt"
        if not gt_path.exists() or not (d / "est.tum").exists():
            continue
        gts.setdefault(seq, load_tum(str(gt_path)))
        est = load_tum(str(d / "est.tum"))
        a, b = ate(gts[seq], est, False), ate(gts[seq], est, True)
        if a is None:
            continue
        ms = None
        log = d / "run.log"
        if log.exists():
            mm = re.search(r"wall_s=([0-9.]+)", log.read_text(errors="replace"))
            if mm and (d / "frames.csv").exists():
                ms = float(mm.group(1)) * 1000 / max(1, sum(1 for _ in open(d / "frames.csv")) - 1)
        rows.append({"run": name, "seq": seq, "ate_se3_cm": a["rmse_cm"], "max_cm": a["max_cm"], "ate_sim3_cm": b["rmse_cm"],
                     "scale": b["scale"], "coverage": a["coverage"], "ms_per_frame": ms})
    (root / "summary.json").write_text(json.dumps(rows, indent=1))
    print("| run | seq | ATE SE3 | max | ATE Sim3 | scale | cov | ms/frame |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run']} | {r['seq']} | {r['ate_se3_cm']:.1f} | {r['max_cm']:.0f} | {r['ate_sim3_cm']:.1f} | {r['scale']:.3f} | {r['coverage']:.2f} | {r['ms_per_frame'] and round(r['ms_per_frame']) or '—'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
