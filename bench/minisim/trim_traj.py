"""Trim a flown TUM trajectory to a render window (replaces the laptop-era ad-hoc trimming).

    python -m bench.minisim.trim_traj traj.tum --info
        per-second motion envelope (speed, altitude) to pick windows by eye
    python -m bench.minisim.trim_traj traj.tum --t0 35.45 --t1 62.33 -o traj_trim.tum
        cut [t0, t1]; timestamps are preserved (render.py re-zeroes internally)
    python -m bench.minisim.trim_traj traj.tum --takeoff -o traj_trim.tum
        start at the first sustained motion minus --margin (static ground start
        with the real takeoff jerk — what OpenVINS-class initialisers need)
"""
from __future__ import annotations

import argparse

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--takeoff", action="store_true", help="auto t0: first sustained |v| > --v-thr, minus --margin")
    ap.add_argument("--v-thr", type=float, default=0.25, help="sustained-motion speed threshold [m/s]")
    ap.add_argument("--margin", type=float, default=2.0, help="seconds of pre-takeoff standstill to keep")
    ap.add_argument("--duration", type=float, default=None, help="window length from t0 (alternative to --t1)")
    ap.add_argument("--info", action="store_true")
    a = ap.parse_args(argv)

    d = np.loadtxt(a.traj)
    t, p = d[:, 0], d[:, 1:4]
    dt = np.gradient(t)
    v = np.linalg.norm(np.gradient(p, axis=0) / dt[:, None], axis=1)

    if a.info:
        print(f"{a.traj}: {len(t)} poses, t [{t[0]:.2f}, {t[-1]:.2f}] ({t[-1]-t[0]:.1f} s)")
        print("  sec    |v| m/s   z m")
        for s in np.arange(np.floor(t[0]), t[-1], 5.0):
            m = (t >= s) & (t < s + 5.0)
            if m.any():
                print(f"  {s:6.1f}  {np.median(v[m]):7.2f}  {np.median(p[m, 2]):6.2f}")
        return 0

    t0 = a.t0
    if a.takeoff:
        # sustained = above threshold for >= 1 s
        w = int(round(1.0 / np.median(dt)))
        sus = np.convolve((v > a.v_thr).astype(float), np.ones(w) / w, "same") > 0.9
        idx = np.nonzero(sus)[0]
        if len(idx) == 0:
            raise SystemExit("no sustained motion found")
        t0 = max(t[idx[0]] - a.margin, t[0])
        print(f"takeoff at ~{t[idx[0]]:.2f} s -> t0 {t0:.2f}")
    if t0 is None:
        raise SystemExit("need --t0, --takeoff or --info")
    t1 = a.t1 if a.t1 is not None else (t0 + a.duration if a.duration else t[-1])
    m = (t >= t0) & (t <= t1)
    if m.sum() < 10:
        raise SystemExit(f"window [{t0}, {t1}] holds {m.sum()} poses")
    out = a.out or a.traj.replace(".tum", "_trim.tum")
    np.savetxt(out, d[m], fmt="%.9f")
    print(f"{out}: {m.sum()} poses, [{t[m][0]:.2f}, {t[m][-1]:.2f}] ({t[m][-1]-t[m][0]:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
