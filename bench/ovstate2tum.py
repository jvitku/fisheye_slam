"""Convert an OpenVINS state_estimate.txt (save_total_state output) to TUM.

OpenVINS columns: t qx qy qz qw px py pz vx vy vz bg... (comment lines start
with '#'). TUM: t px py pz qx qy qz qw.

Usage: python -m bench.ovstate2tum state_estimate.txt out_tum.txt
"""

import sys

import numpy as np


def convert(src: str, dst: str) -> int:
    rows = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = line.split()
            if len(v) < 8:
                continue
            t, qx, qy, qz, qw, px, py, pz = (float(x) for x in v[:8])
            rows.append([t, px, py, pz, qx, qy, qz, qw])
    if not rows:
        raise ValueError(f"{src}: no state rows (estimator never initialized?)")
    np.savetxt(dst, np.asarray(rows), fmt="%.9f")
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    n = convert(sys.argv[1], sys.argv[2])
    print(f"{n} poses -> {sys.argv[2]}")
