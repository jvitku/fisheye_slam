"""Sample a recorded body trajectory at a fixed camera rate (SE3 interpolation).

"Physics once, render offline" (bench/README.md): the flight is flown once
(PX4 SITL + physics; only IMU + ground truth are recorded), then every camera
is rendered offline along that trajectory. This tool turns the ground truth
(TUM `t x y z qx qy qz qw`, any rate, BODY frame — bench/gt_extract.py without
--rig) into the exact frame times + poses of the render pass:

    t_k = t0 + k / rate,  pose(t_k) = SE3 interpolation of the two bracketing
    GT samples (position linear, rotation slerp, shortest arc)

Every rig's cameras are placed relative to this body pose, so the fisheye pod
and the OAK-D see byte-identical motion, and timestamps are exact multiples of
the frame period (perfectly synchronized; bench/skew_bag.py adds the unsync).

Usage:
    python -m bench.pose_sampler gt_body.txt poses.txt [--rate 20]
        [--start 0.0] [--duration 120]      (offsets in seconds from the first GT sample)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from bench.evaluate import load_tum


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """Shortest-arc spherical interpolation of unit quaternions [x y z w]."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:                       # take the short way round
        q1, d = -q1, -d
    if d > 0.9995:                    # nearly parallel: lerp + renormalize
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(d)
    return (np.sin((1 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)


def sample(poses: np.ndarray, rate: float, start: float = 0.0,
           duration: float | None = None) -> np.ndarray:
    """(N,8) GT -> (M,8) poses at `rate` Hz from t_first+start (for `duration` s)."""
    t = poses[:, 0]
    t0 = t[0] + start
    t1 = t[-1] if duration is None else min(t[-1], t0 + duration)
    if t1 < t0:
        raise ValueError("start offset beyond the end of the trajectory")
    n = int(np.floor((t1 - t0) * rate + 1e-9)) + 1
    ts = t0 + np.arange(n) / rate
    out = np.empty((n, 8))
    out[:, 0] = ts
    idx = np.clip(np.searchsorted(t, ts, side="right"), 1, len(t) - 1)
    for k, (tk, i) in enumerate(zip(ts, idx)):
        ta, tb = t[i - 1], t[i]
        s = 0.0 if tb <= ta else float(np.clip((tk - ta) / (tb - ta), 0.0, 1.0))
        out[k, 1:4] = poses[i - 1, 1:4] + s * (poses[i, 1:4] - poses[i - 1, 1:4])
        out[k, 4:8] = slerp(poses[i - 1, 4:8], poses[i, 4:8], s)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt", help="ground truth, TUM, body frame")
    ap.add_argument("out", help="sampled poses, TUM")
    ap.add_argument("--rate", type=float, default=20.0, help="frame rate [Hz]")
    ap.add_argument("--start", type=float, default=0.0, help="skip this many seconds")
    ap.add_argument("--duration", type=float, default=None, help="render this many seconds")
    args = ap.parse_args(argv)
    out = sample(load_tum(args.gt), args.rate, args.start, args.duration)
    np.savetxt(args.out, out, fmt="%.9f",
               header=f"t x y z qx qy qz qw  ({args.rate} Hz samples of {args.gt})")
    print(f"{len(out)} poses at {args.rate} Hz ({out[-1, 0] - out[0, 0]:.1f} s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
