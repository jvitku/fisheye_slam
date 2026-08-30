"""Run podslam on a bag-contract ROS1 bag -> TUM trajectory (IMU frame) + per-frame stats.

Same CLI shape and outputs as experiments/08_oakdpro_slam/cuvslam_track.py so
bench/sweep_cuvslam.sh, bench/compare_rigs.sh and bench/collect_results.py
work unchanged:
    <out>/est.tum        t x y z qx qy qz qw   (pose of the pod IMU in the world)
    <out>/frames.csv     t,tracked,n_obs0,n_obs1,mean0,ms,keyframe,n_landmarks

Usage:
    python3 -m podslam.track_bag input.bag out/ --rig rigs/tumvi_room1.yaml
        [--frontend klt|xfeat] [--preprocess norm|clahe|...] [--masks sat|learned:<pt>]
        [--kf-every 3] [--lag 4] [--no-circle-mask] [--stats out/frames.csv]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from .geometry import R_to_quat_xyzw
from .rig import load_rig
from .tracker import Tracker, TrackerConfig

TS = get_typestore(Stores.ROS1_NOETIC)


def to_gray(msg) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "mono8":
        return data.reshape(msg.height, msg.width).copy()
    if msg.encoding in ("rgb8", "bgr8"):
        rgb = data.reshape(msg.height, msg.width, 3).astype(np.float32)
        r, g, b = (rgb[..., 0], rgb[..., 1], rgb[..., 2]) if msg.encoding == "rgb8" else (rgb[..., 2], rgb[..., 1], rgb[..., 0])
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    raise ValueError(f"unsupported encoding {msg.encoding}")


def stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--frontend", default="klt")
    ap.add_argument("--preprocess", default="norm")
    ap.add_argument("--masks", default="none")
    ap.add_argument("--no-circle-mask", action="store_true")
    ap.add_argument("--kf-every", type=int, default=3)
    ap.add_argument("--lag", type=float, default=4.0)
    ap.add_argument("--px-sigma", type=float, default=1.5)
    ap.add_argument("--imu-noise-scale", default="5.7,1.8,1.2,4.5", help="estimator-side multipliers for accel noise, gyro noise, accel walk, gyro walk")
    ap.add_argument("--backend", default="smart", choices=("smart", "explicit"))
    ap.add_argument("--marg", default="all", choices=("ended", "all", "pin"), help="smart backend marginalisation mode")
    ap.add_argument("--epi", action="store_true", help="smart backend: nonlinear landmark refinement (gtsam throws inside LM on this data)")
    ap.add_argument("--max-features", type=int, default=300)
    ap.add_argument("--frontend-opt", action="append", default=[], help="front-end config override key=value (e.g. default_depth=1.5)")
    ap.add_argument("--init-acc-bias-sigma", type=float, default=0.1, help="static-init prior sigma of the accelerometer bias [m/s^2]")
    ap.add_argument("--cams", default=None, help="comma-separated camera names to use (subset of the rig, first = tracking camera)")
    ap.add_argument("--kf-rot-deg", type=float, default=0.0, help="motion-adaptive keyframes: IMU rotation since last keyframe (0 = off)")
    ap.add_argument("--kf-parallax-px", type=float, default=0.0, help="motion-adaptive keyframes: median parallax since last keyframe (0 = off)")
    ap.add_argument("--noise-gate", type=float, default=4.0, help="KLT detector: min-eigenvalue response >= k x frame median (0 = off)")
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    try:                                   # thermal budget: OpenCV's pool stays small unless asked
        import os
        import cv2
        cv2.setNumThreads(int(os.environ.get("PODSLAM_THREADS", "2")))
    except Exception:
        pass
    rig = load_rig(args.rig)
    if args.cams:
        wanted = [c.strip() for c in args.cams.split(",") if c.strip()]
        by_name = {c.name: c for c in rig.cameras}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise SystemExit(f"--cams: unknown camera(s) {missing}; rig has {[c.name for c in rig.cameras]}")
        rig.cameras = [by_name[w] for w in wanted]
        print(f"using cameras {wanted}")
    noise_scale = tuple(float(x) for x in args.imu_noise_scale.split(","))
    cfg = TrackerConfig(frontend=args.frontend, frontend_cfg={"max_features": args.max_features, "noise_gate": args.noise_gate, **{k: float(v) for k, v in (o.split("=", 1) for o in args.frontend_opt)}},
                        init_acc_bias_sigma=args.init_acc_bias_sigma,
                        preprocess=args.preprocess, masks=args.masks, circle_mask=not args.no_circle_mask,
                        kf_every=args.kf_every, kf_rot_deg=args.kf_rot_deg, kf_parallax_px=args.kf_parallax_px, lag_s=args.lag, px_sigma=args.px_sigma, backend=args.backend, marg_mode=args.marg, smart_epi=args.epi, imu_noise_scale=noise_scale, verbose=args.verbose)
    tracker = Tracker(rig, cfg)
    cam_topics = {c.topic: i for i, c in enumerate(rig.cameras)}
    imu_topic = rig.imu.topic
    tum = (args.out / "est.tum").open("w")
    stats = (args.stats or (args.out / "frames.csv")).open("w")
    stats.write("t,tracked,n_obs0,n_obs1,mean0,ms,keyframe,n_landmarks\n")

    pending: dict[int, dict] = {}
    shift_ns = int(round(rig.cameras[0].time_shift_s * 1e9))
    acc_k = float(rig.imu.accel_scale)
    if acc_k != 1.0:
        print(f"accelerometer scale correction x{acc_k:.4f}")
    if shift_ns:
        print(f"camera->IMU time shift {shift_ns / 1e6:.2f} ms applied to image stamps")
    n_frames = n_ok = n_kf = 0
    LAG_NS = 100_000_000

    def process(t_ns, group):
        nonlocal n_frames, n_ok, n_kf
        imgs = [group[i] for i in range(len(rig.cameras))]
        t0 = time.perf_counter()
        est = tracker.track(t_ns, imgs)
        ms = (time.perf_counter() - t0) * 1e3
        n_frames += 1
        n1 = est.n_obs[1] if len(est.n_obs) > 1 else 0
        stats.write(f"{t_ns / 1e9:.6f},{int(est.ok)},{est.n_obs[0]},{n1},{float(imgs[0].mean()):.2f},{ms:.2f},{int(est.keyframe)},{est.n_landmarks}\n")
        if est.ok and est.T_W_I is not None:
            n_ok += 1; n_kf += int(est.keyframe)
            T = est.T_W_I
            qx, qy, qz, qw = R_to_quat_xyzw(T[:3, :3])
            tum.write(f"{t_ns / 1e9:.6f} {T[0, 3]} {T[1, 3]} {T[2, 3]} {qx} {qy} {qz} {qw}\n")

    def flush(upto_ns):
        for t in sorted(pending):
            if upto_ns is not None and t > upto_ns - LAG_NS:
                break
            g = pending.pop(t)
            if all(i in g for i in range(len(rig.cameras))):
                process(t, g)

    with Reader(args.bag) as reader:
        conns = [c for c in reader.connections if c.topic in cam_topics or c.topic == imu_topic]
        for conn, _, raw in reader.messages(connections=conns):
            msg = TS.deserialize_ros1(raw, conn.msgtype)
            if conn.topic == imu_topic:
                g, a = msg.angular_velocity, msg.linear_acceleration
                tracker.register_imu(stamp_ns(msg), [g.x, g.y, g.z], [a.x * acc_k, a.y * acc_k, a.z * acc_k])
                continue
            # image stamps into the IMU clock (Kalibr timeshift_cam_imu, per rig); the
            # estimate is written with the shifted stamp so it aligns with IMU-frame GT
            t_ns = stamp_ns(msg) + shift_ns
            pending.setdefault(t_ns, {})[cam_topics[conn.topic]] = to_gray(msg)
            flush(t_ns)
            if args.max_frames and n_frames >= args.max_frames:
                break
        flush(None)
    tum.close(); stats.close()
    be = tracker.backend
    resets = be.n_resets if be else 0
    extra = f", {be.n_failed_solves} failed solves" if be is not None and hasattr(be, "n_failed_solves") else ""
    if be is not None and hasattr(be, "n_outliers"):
        extra += f", {be.n_outliers} outlier landmarks, prior absorbed/rejected {be.n_prior_absorbed}/{be.n_prior_rejected}, {be.n_marginalized} marginalised"
    print(f"tracked {n_ok}/{n_frames} frames ({n_kf} keyframes, {resets} soft resets{extra}) -> {args.out / 'est.tum'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
