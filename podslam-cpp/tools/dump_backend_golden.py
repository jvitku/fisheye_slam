#!/usr/bin/env python3
"""Dump the smart backend's exact inputs and outputs for the C++ port's parity tests.

Runs the tracker over the first N frames of a bag and records, per keyframe:
  - the raw IMU samples integrated since the previous keyframe (t, gyro, accel)
    and the bias used for preintegration,
  - every landmark observation added at this keyframe (landmark id, camera id,
    normalized measurement),
  - the solved state after optimize() (T_W_I row-major, velocity, bias).

The C++ Window (port step 2, docs/CPP_PORT_PLAN.md) must reproduce the solved
states from the same inputs to within 1e-6.  Output: JSON lines.

    UV_NO_SYNC=1 PYTHONPATH=. uv run python podslam-cpp/tools/dump_backend_golden.py \
        datasets/data/tumvi/room1_day.bag rigs/tumvi_room1.yaml \
        podslam-cpp/tests/data/backend_golden.jsonl --max-frames 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag"); ap.add_argument("rig"); ap.add_argument("out")
    ap.add_argument("--max-frames", type=int, default=400)
    a = ap.parse_args(argv)

    from podslam import track_bag
    from podslam.backend_smart import SmartBackend
    from podslam.imu import Preintegrator

    rec = {"imu": [], "kf": []}
    orig_integrate = Preintegrator.integrate_until

    def integrate_until(self, buf, t):
        n0 = len(rec["imu"])
        out = orig_integrate(self, buf, t)
        return out

    orig_reg = track_bag.Tracker.register_imu if hasattr(track_bag, "Tracker") else None

    # record raw IMU at the tracker boundary
    from podslam.tracker import Tracker
    orig_register = Tracker.register_imu

    def register(self, t_ns, gyro, accel):
        rec["imu"].append([t_ns * 1e-9, *map(float, gyro), *map(float, accel)])
        return orig_register(self, t_ns, gyro, accel)

    Tracker.register_imu = register

    orig_add_obs = SmartBackend.add_observation
    pending_obs = []

    def add_observation(self, k, cam, j, bearing, t):
        ok = orig_add_obs(self, k, cam, j, bearing, t)
        if ok:
            b = np.asarray(bearing, float)
            pending_obs.append([int(k), int(cam), int(j), b[0] / b[2], b[1] / b[2]])
        return ok

    SmartBackend.add_observation = add_observation

    orig_opt = SmartBackend.optimize

    def optimize(self, k, t):
        out = orig_opt(self, k, t)
        pose, vel, bias = self.kf_state[k]
        rec["kf"].append({
            "k": int(k), "t": float(t),
            "obs": [o for o in pending_obs],
            "T_W_I": [float(x) for x in np.asarray(pose.matrix()).reshape(-1)],
            "vel": [float(x) for x in np.asarray(vel)],
            "bias": [float(x) for x in np.asarray(bias.vector())],
            "n_active": int(self.n_active_lm), "n_valid": int(self.n_valid_lm),
        })
        pending_obs.clear()
        return out

    SmartBackend.optimize = optimize

    track_bag.main([a.bag, "/tmp/claude-1001/-home-jarda-workspace-fisheye-slam/ad5a9868-cf21-42cc-901a-0357558c2607/scratchpad/golden_run",
                    "--rig", a.rig, "--max-frames", str(a.max_frames)])

    with open(a.out, "w") as f:
        f.write(json.dumps({"type": "imu", "rows": rec["imu"]}) + "\n")
        for kf in rec["kf"]:
            f.write(json.dumps({"type": "kf", **kf}) + "\n")
    print(f"wrote {a.out}: {len(rec['imu'])} imu samples, {len(rec['kf'])} keyframes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
