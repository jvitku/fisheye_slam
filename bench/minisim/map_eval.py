"""Evaluate map precision against the minisim's EXACT analytic geometry.

Accuracy: unsigned distance of every map point to the nearest true surface.
Completeness: fraction of observable true-surface samples (within reach of the
flown trajectory) that have a map point within tau.

    python -m bench.minisim.map_eval run/map.npz indoor --gt frames/gt.tum
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.minisim import scene as S    # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map_npz"); ap.add_argument("scene", choices=list(S.SCENES))
    ap.add_argument("--gt", default=None, help="gt.tum of the flight (observability mask)")
    ap.add_argument("--tau", type=float, default=0.10, help="completeness threshold [m]")
    ap.add_argument("--reach", type=float, default=15.0, help="observable = within this of the path")
    ap.add_argument("--out", default=None, help="write metrics json here")
    a = ap.parse_args(argv)

    m = np.load(a.map_npz)
    pts, kind = m["points"], m["kind"]
    prims = S.SCENES[a.scene][0]()
    res: dict = dict(n_points=int(len(pts)), n_dense=int((kind == 0).sum()),
                     n_landmarks=int((kind == 1).sum()))
    if len(pts) == 0:
        print("empty map"); res["empty"] = True
    else:
        d = S.distance_to_surface(pts, prims)
        for name, mask in (("all", np.ones(len(pts), bool)), ("dense", kind == 0), ("landmarks", kind == 1)):
            if mask.sum() == 0:
                continue
            dd = d[mask]
            res[f"acc_{name}"] = dict(
                mean=float(dd.mean()), median=float(np.median(dd)),
                rmse=float(np.sqrt((dd ** 2).mean())), p90=float(np.percentile(dd, 90)),
                inlier5=float((dd < 0.05).mean()), inlier10=float((dd < 0.10).mean()),
                inlier20=float((dd < 0.20).mean()), n=int(mask.sum()))
        samples = S.surface_samples(prims, 400)
        if a.gt:
            traj = np.loadtxt(a.gt)[:, 1:4]
            from scipy.spatial import cKDTree
            near = cKDTree(traj).query(samples, k=1)[0] < a.reach
            samples = samples[near & (samples[:, 2] <= 8.0)]
        from scipy.spatial import cKDTree
        dn = cKDTree(pts).query(samples, k=1)[0]
        res["completeness"] = dict(tau=a.tau, frac=float((dn < a.tau).mean()), n_samples=int(len(samples)))
        acc = res.get("acc_all", {})
        print(f"map {a.map_npz}: {len(pts)} pts | acc median {acc.get('median', float('nan'))*100:.1f} cm, "
              f"rmse {acc.get('rmse', float('nan'))*100:.1f} cm, inlier@10cm {acc.get('inlier10', 0)*100:.0f}% | "
              f"completeness@{a.tau*100:.0f}cm {res['completeness']['frac']*100:.0f}% of {len(samples)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
