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

    orig_track = Tracker.track

    def track_wrap(self, t_ns, images, masks=None):
        rec.setdefault("frames", []).append(t_ns * 1e-9)
        return orig_track(self, t_ns, images, masks)

    Tracker.track = track_wrap

    orig_add_obs = SmartBackend.add_observation
    pending_obs = []

    def add_observation(self, k, cam, j, bearing, t):
        ok = orig_add_obs(self, k, cam, j, bearing, t)
        if ok:
            b = np.asarray(bearing, float)
            pending_obs.append([int(k), int(cam), int(j), b[0] / b[2], b[1] / b[2]])
        return ok

    SmartBackend.add_observation = add_observation

    orig_init = SmartBackend.initialize

    def initialize(self, k, t, T_W_I, vel, bias, **kw):
        T = np.asarray(T_W_I, float)[:3].reshape(-1)
        rec["init"] = [float(t), *[float(x) for x in T], *[float(x) for x in np.asarray(vel, float)],
                       *[float(x) for x in np.asarray(bias.vector())]]
        return orig_init(self, k, t, T_W_I, vel, bias, **kw)

    SmartBackend.initialize = initialize

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

    # Flat line format (easy to strtod in C++):
    #   R fx0 cx0 ... : one line per rig camera: fx fy cx cy k1..k4 then T_imu_cam row-major (12)
    #   N imu params  : gyro_nd gyro_rw accel_nd accel_rw accel_scale scale_a scale_g scale_aw scale_gw px_sigma
    #   I t gx gy gz ax ay az
    #   O k cam j mx my
    #   K k t T(12 row-major) vx vy vz b(6)
    from podslam.rig import load_rig
    rig2 = load_rig(a.rig)
    t_base = rec["imu"][0][0] if rec["imu"] else 0.0     # %.12g on epoch seconds = 1 ms resolution; write relative times
    with open(a.out, "w") as f:
        for c in rig2.cameras:
            m = c.model
            T = np.asarray(c.T_imu_cam)[:3].reshape(-1)
            f.write("R " + " ".join(f"{x:.12g}" for x in [m.fx, m.fy, m.cx, m.cy, m.k1, m.k2, m.k3, m.k4, *T]) + "\n")
        from podslam.tracker import TrackerConfig
        ns = TrackerConfig().imu_noise_scale
        f.write("N " + " ".join(f"{x:.12g}" for x in [rig2.imu.gyro_noise_density, rig2.imu.gyro_random_walk,
                rig2.imu.accel_noise_density, rig2.imu.accel_random_walk, rig2.imu.accel_scale, *ns,
                float(getattr(rig2, "px_sigma", 1.5))]) + "\n")
        for r in rec["imu"]:
            f.write(f"I {r[0] - t_base:.9f} " + " ".join(f"{x:.12g}" for x in r[1:]) + "\n")
        for t in rec.get("frames", []):
            f.write(f"F {t - t_base:.9f}\n")
        if "init" in rec:
            f.write(f"S {rec['init'][0] - t_base:.9f} " + " ".join(f"{x:.12g}" for x in rec["init"][1:]) + "\n")
        for kf in rec["kf"]:
            for o in kf["obs"]:
                f.write("O " + " ".join(f"{x:.12g}" for x in o) + "\n")
            T = kf["T_W_I"][:12]
            f.write(f"K {kf['k']} {kf['t'] - t_base:.9f} " + " ".join(f"{x:.12g}" for x in [*T, *kf["vel"], *kf["bias"]]) + "\n")
    print(f"wrote {a.out}: {len(rec['imu'])} imu samples, {len(rec['kf'])} keyframes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
