#!/usr/bin/env python3
"""OpenVINS state_estimate.txt (JPL q_GtoI, p_IinG) -> TUM est.tum (Hamilton q_ItoG)."""
import sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
rows = [l.split() for l in open(src) if not l.startswith("#")]
a = np.array([[float(x) for x in r[:8]] for r in rows])
t, q, p = a[:, 0], a[:, 1:5], a[:, 5:8]
q_h = np.column_stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]])
np.savetxt(dst, np.column_stack([t, p, q_h]), fmt="%.6f %.6f %.6f %.6f %.7f %.7f %.7f %.7f")
print(f"wrote {dst}: {len(t)} poses over {t[-1] - t[0]:.1f} s")
