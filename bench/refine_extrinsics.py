#!/usr/bin/env python3
"""Offline camera-IMU extrinsic refinement from a bag (the OpenVINS-gap experiment).

Runs the podslam tracker over the first N seconds, keeps every keyframe's solved
state and per-camera observations, then solves one batch problem where each
camera's ``body_P_cam`` is a variable (gtsam_unstable ProjectionFactorPPP*):

    keyframe poses      X(k)   priors at the tracker solution (pos 2 cm, rot 0.5 deg)
    extrinsics          T(c)   prior at the rig value (rot 1 deg, trans 1 cm)
    landmarks           L(j)   DLT from the stored bearings at the solved poses

Prints the per-camera correction (deg / mm) and optionally writes a refined rig
yaml.  This mirrors what OpenVINS estimates online; if re-running the benchmark
with the refined rig closes the gap, online calibration is worth building.

    UV_NO_SYNC=1 PYTHONPATH=. uv run python bench/refine_extrinsics.py \
        datasets/data/hilti2022/exp14_basement_2.bag rigs/hilti2022.yaml \
        --cams cam0,cam1 --seconds 40 --out /tmp/hilti_refined.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag"); ap.add_argument("rig")
    ap.add_argument("--cams", default=None)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--kf-every", type=int, default=6)
    ap.add_argument("--min-obs", type=int, default=6)
    ap.add_argument("--pose-sigma", type=float, default=0.02)
    ap.add_argument("--rot-sigma-deg", type=float, default=0.5)
    ap.add_argument("--ext-rot-sigma-deg", type=float, default=1.0)
    ap.add_argument("--ext-trans-sigma", type=float, default=0.01)
    ap.add_argument("--out", default=None, help="write a refined rig yaml here")
    a = ap.parse_args(argv)

    import gtsam
    import gtsam_unstable
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    from podslam.rig import load_rig
    from podslam.track_bag import to_gray
    from podslam.tracker import Tracker, TrackerConfig

    TS = get_typestore(Stores.ROS1_NOETIC)
    rig = load_rig(a.rig)
    if a.cams:
        wanted = [c.strip() for c in a.cams.split(",")]
        by = {c.name: c for c in rig.cameras}
        rig.cameras = [by[w] for w in wanted]
    cfg = TrackerConfig(kf_every=a.kf_every, px_sigma=float(getattr(rig, "px_sigma", 1.5)),
                        imu_noise_scale=(1.0, 1.0, 1.0, 1.0))
    tracker = Tracker(rig, cfg)

    # ---- run the tracker, recording keyframe states and observations
    kf_poses: dict[int, np.ndarray] = {}
    obs: dict[int, list] = {}                      # track id -> [(k, cam, bearing)]
    shift_ns = int(round(rig.cameras[0].time_shift_s * 1e9))
    acc_k = float(rig.imu.accel_scale)
    cam_topics = {c.topic: i for i, c in enumerate(rig.cameras)}
    pending: dict[int, dict] = {}
    t0 = None
    k_now = -1

    def on_frame(t_ns, group):
        nonlocal k_now
        imgs = [group[i] for i in range(len(rig.cameras))]
        est = tracker.track(t_ns, imgs)
        if not (est.ok and est.keyframe and est.T_W_I is not None):
            return
        k_now += 1
        kf_poses[k_now] = est.T_W_I.copy()
        feats = tracker.frontend  # last processed features live on the tracker's frontend? use tracker state instead
        # use the tracker's per-camera observations from this keyframe via track_to_lm-independent ids:
        ff = last_feats[0]
        for c, cam_obs in enumerate(ff.cams):
            for n, tid in enumerate(cam_obs.ids):
                b = cam_obs.bearings[n]
                if np.all(np.isfinite(b)):
                    obs.setdefault(int(tid), []).append((k_now, c, b.copy()))

    last_feats = [None]
    orig_process = tracker.frontend.process
    def wrapped(t_ns, images, masks, dR):
        ff = orig_process(t_ns, images, masks, dR)
        last_feats[0] = ff
        return ff
    tracker.frontend.process = wrapped

    with Reader(a.bag) as reader:
        conns = [c for c in reader.connections if c.topic in cam_topics or c.topic == rig.imu.topic]
        for conn, _, raw in reader.messages(connections=conns):
            msg = TS.deserialize_ros1(raw, conn.msgtype)
            st = msg.header.stamp
            t_ns = st.sec * 1_000_000_000 + st.nanosec
            if t0 is None:
                t0 = t_ns
            if t_ns - t0 > a.seconds * 1e9:
                break
            if conn.topic == rig.imu.topic:
                g, ac = msg.angular_velocity, msg.linear_acceleration
                tracker.register_imu(t_ns, [g.x, g.y, g.z], [ac.x * acc_k, ac.y * acc_k, ac.z * acc_k])
                continue
            t_img = t_ns + shift_ns
            pending.setdefault(t_img, {})[cam_topics[conn.topic]] = to_gray(msg)
            for t in sorted(list(pending)):
                if t > t_img - 100_000_000:
                    break
                grp = pending.pop(t)
                if all(i in grp for i in range(len(rig.cameras))):
                    on_frame(t, grp)
    print(f"tracked: {len(kf_poses)} keyframes, {len(obs)} tracks")

    # ---- batch refinement
    g = gtsam
    X = lambda k: g.symbol('x', k); T = lambda c: g.symbol('t', c); L = lambda j: g.symbol('l', j)
    K = g.Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0)
    graph = g.NonlinearFactorGraph(); values = g.Values()
    rot_s = np.deg2rad(a.rot_sigma_deg)
    pose_prior = g.noiseModel.Diagonal.Sigmas(np.array([rot_s] * 3 + [a.pose_sigma] * 3))
    for k, Twb in kf_poses.items():
        values.insert(X(k), g.Pose3(Twb))
        graph.add(g.PriorFactorPose3(X(k), g.Pose3(Twb), pose_prior))
    ext_prior = g.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(a.ext_rot_sigma_deg)] * 3 + [a.ext_trans_sigma] * 3))
    for c, cam in enumerate(rig.cameras):
        values.insert(T(c), g.Pose3(cam.T_imu_cam))
        graph.add(g.PriorFactorPose3(T(c), g.Pose3(cam.T_imu_cam), ext_prior))
    meas_noise = g.noiseModel.Robust.Create(
        g.noiseModel.mEstimator.Huber.Create(1.345),
        g.noiseModel.Isotropic.Sigma(2, float(getattr(rig, "px_sigma", 1.5)) / rig.cameras[0].model.fx))
    n_lm = n_f = 0
    for tid, lst in obs.items():
        if len(lst) < a.min_obs or len({k for k, _, _ in lst}) < 3:
            continue
        poses = [g.Pose3(kf_poses[k] @ rig.cameras[c].T_imu_cam) for k, c, _ in lst]
        ms = [np.array([b[0] / b[2], b[1] / b[2]]) for _, _, b in lst]
        try:
            p = g.triangulatePoint3(poses, K, ms, 1e-6, False)
        except Exception:
            continue
        p = np.asarray(p, float)
        depths = [(np.linalg.inv(poses[i].matrix()) @ np.append(p, 1.0))[2] for i in range(len(poses))]
        if min(depths) < 0.3 or max(depths) > 60.0:
            continue
        j = n_lm; n_lm += 1
        values.insert(L(j), g.Point3(*p))
        for (k, c, b), m in zip(lst, ms):
            graph.add(gtsam_unstable.ProjectionFactorPPPCal3_S2(m, meas_noise, X(k), T(c), L(j), K))
            n_f += 1
    print(f"batch: {n_lm} landmarks, {n_f} projection factors, {len(kf_poses)} poses")
    params = g.LevenbergMarquardtParams(); params.setMaxIterations(30); params.setVerbosityLM("SILENT")
    result = g.LevenbergMarquardtOptimizer(graph, values, params).optimize()

    refined = {}
    for c, cam in enumerate(rig.cameras):
        T_new = result.atPose3(T(c)).matrix()
        d = np.linalg.inv(cam.T_imu_cam) @ T_new
        ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
        dt = np.linalg.norm(d[:3, 3]) * 1000
        refined[cam.name] = T_new
        print(f"{cam.name}: rotation moved {ang:.3f} deg, translation moved {dt:.1f} mm; new T_imu_cam translation {np.round(T_new[:3, 3], 5)}")

    if a.out:
        import yaml
        raw = yaml.safe_load(open(a.rig))
        for cy in raw["cameras"]:
            if cy["name"] in refined:
                T_ic = refined[cy["name"]]
                cy["T_cam_imu"] = [[float(v) for v in row] for row in np.linalg.inv(T_ic)[:4]]
        with open(a.out, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
