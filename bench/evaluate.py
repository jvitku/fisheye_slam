"""Trajectory evaluation for the 3d_fisheye benchmark.

Compares an estimated trajectory against ground truth (both TUM format:
`t x y z qx qy qz qw` per line) and reports the benchmark metrics:

- ATE (SE3 Umeyama alignment): rmse / mean / median / max [m]
- RPE translation over a time delta (default 1 s): rmse / max [m]
- coverage: fraction of the ground-truth time span covered by estimates
  (the robustness metric — a tracker that dies half-way scores 0.5, even if
  its surviving half is accurate)
- gap count / longest gap: estimator dropouts > gap threshold

Pure numpy on purpose: runs anywhere (host, CI, containers) with no ROS or
evo dependency. Cross-check against `evo_ape` when available.

Usage:
    python -m bench.evaluate gt.txt est.txt [--delta 1.0] [--max-diff 0.02]
        [--gap 0.5] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load_tum(path: str) -> np.ndarray:
    """Load a TUM-format trajectory -> (N, 8) array [t x y z qx qy qz qw]."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 8:
                raise ValueError(f"{path}: expected 8 columns, got {len(parts)}: {line!r}")
            rows.append([float(x) for x in parts[:8]])
    if not rows:
        raise ValueError(f"{path}: no trajectory rows")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[np.argsort(arr[:, 0])]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N,4) quaternions [qx qy qz qw] -> (N,3,3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def associate(t_ref: np.ndarray, t_query: np.ndarray, max_diff: float):
    """Match each query timestamp to the nearest reference timestamp.

    Returns (ref_idx, query_idx) for pairs within max_diff seconds.
    """
    idx = np.searchsorted(t_ref, t_query)
    idx = np.clip(idx, 1, len(t_ref) - 1)
    left, right = t_ref[idx - 1], t_ref[idx]
    nearest = np.where(np.abs(t_query - left) <= np.abs(t_query - right), idx - 1, idx)
    ok = np.abs(t_ref[nearest] - t_query) <= max_diff
    return nearest[ok], np.nonzero(ok)[0]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Least-squares similarity transform: s*R @ src + t ≈ dst. Returns (s, R, t)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / xs.var(axis=0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate_stats(gt_xyz: np.ndarray, est_xyz: np.ndarray, with_scale: bool = False) -> dict:
    s, R, t = umeyama(est_xyz, gt_xyz, with_scale)
    err = np.linalg.norm((s * (R @ est_xyz.T)).T + t - gt_xyz, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "scale": s,
        "pairs": int(len(err)),
    }


def rpe_stats(gt: np.ndarray, est: np.ndarray, delta: float) -> dict:
    """Relative pose (translation) error over time windows of `delta` seconds.

    gt/est are time-associated (N,8) arrays with identical row correspondence.
    """
    t = gt[:, 0]
    j = np.searchsorted(t, t + delta)
    keep = j < len(t)
    i, j = np.nonzero(keep)[0], j[keep]
    if len(i) == 0:
        return {"rmse": None, "max": None, "pairs": 0}

    Rg, Re = quat_to_rot(gt[:, 4:8]), quat_to_rot(est[:, 4:8])
    # relative translations expressed in the frame at time i
    dg = np.einsum("nij,ni->nj", Rg[i].transpose(0, 2, 1), gt[j, 1:4] - gt[i, 1:4])
    de = np.einsum("nij,ni->nj", Re[i].transpose(0, 2, 1), est[j, 1:4] - est[i, 1:4])
    err = np.linalg.norm(dg - de, axis=1)
    return {"rmse": float(np.sqrt(np.mean(err**2))), "max": float(err.max()), "pairs": int(len(err))}


def coverage_stats(t_gt: np.ndarray, t_est: np.ndarray, gap_threshold: float) -> dict:
    """How much of the GT time span the estimator actually covered."""
    t0, t1 = t_gt[0], t_gt[-1]
    te = t_est[(t_est >= t0) & (t_est <= t1)]
    if len(te) < 2:
        return {"coverage": 0.0, "gaps": 0, "longest_gap": float(t1 - t0)}
    # boundaries count as gaps too
    stamps = np.concatenate([[t0], te, [t1]])
    dt = np.diff(stamps)
    gaps = dt[dt > gap_threshold]
    covered = float((t1 - t0) - gaps.sum())
    return {
        "coverage": covered / float(t1 - t0),
        "gaps": int(len(gaps)),
        "longest_gap": float(dt.max()),
    }


def evaluate(gt_path: str, est_path: str, max_diff: float = 0.02,
             delta: float = 1.0, gap_threshold: float = 0.5,
             with_scale: bool = False) -> dict:
    gt, est = load_tum(gt_path), load_tum(est_path)
    gi, ei = associate(gt[:, 0], est[:, 0], max_diff)
    if len(gi) < 10:
        raise ValueError(f"only {len(gi)} associated pairs (max_diff={max_diff}s) — clocks aligned?")
    g, e = gt[gi], est[ei]
    return {
        "ate": ate_stats(g[:, 1:4], e[:, 1:4], with_scale),
        "rpe": rpe_stats(g, e, delta),
        **coverage_stats(gt[:, 0], est[:, 0], gap_threshold),
        "gt_duration": float(gt[-1, 0] - gt[0, 0]),
        "est_poses": int(len(est)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt", help="ground-truth trajectory (TUM format)")
    ap.add_argument("est", help="estimated trajectory (TUM format)")
    ap.add_argument("--max-diff", type=float, default=0.02, help="association tolerance [s]")
    ap.add_argument("--delta", type=float, default=1.0, help="RPE window [s]")
    ap.add_argument("--gap", type=float, default=0.5, help="dropout gap threshold [s]")
    ap.add_argument("--scale", action="store_true", help="Sim(3) alignment (monocular candidates)")
    ap.add_argument("--json", help="also write metrics to this json file")
    args = ap.parse_args(argv)

    m = evaluate(args.gt, args.est, args.max_diff, args.delta, args.gap, args.scale)
    print(json.dumps(m, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(m, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
