"""Render a VIO debug video from a bag: all cameras with live feature tracks,
estimated position/velocity HUD, and a top-down trajectory inset (est vs GT).

    python -m bench.make_debug_video input.bag out.mp4 --rig rigs/skydio6.yaml \
        [--frontend multiklt] [--gt gt.tum] [--flags "--init-mode auto ..."-style
        tracker options are taken from the same CLI switches as track_bag where
        relevant] [--scale 1.0] [--trail 8]

The tracker runs exactly as in podslam.track_bag (same config surface for the
common switches); the front-end is wrapped to capture per-camera tracked points
for drawing. Tracks render as dot+trail polylines (per-camera id spaces safe).
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from podslam.rig import load_rig
from podslam.track_bag import TS, stamp_ns, to_gray
from podslam.tracker import Tracker, TrackerConfig


def load_tum(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1:4]


def main(argv=None) -> int:
    import cv2
    from rosbags.rosbag1 import Reader

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag"); ap.add_argument("out")
    ap.add_argument("--rig", required=True)
    ap.add_argument("--frontend", default="multiklt")
    ap.add_argument("--init-mode", default="auto")
    ap.add_argument("--kf-dense-init", type=float, default=2.0)
    ap.add_argument("--dyn-weight", action="store_true", default=True)
    ap.add_argument("--gt", default=None, help="gt.tum for the inset + live error")
    ap.add_argument("--scale", type=float, default=1.0, help="per-camera display scale")
    ap.add_argument("--trail", type=int, default=8, help="track trail length [frames]")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--world", action="store_true", help="add live point-cloud + 15 cm occupancy (octomap) panels")
    ap.add_argument("--densify-step", type=int, default=12)
    ap.add_argument("--occ-voxel", type=float, default=0.15)
    a = ap.parse_args(argv)

    rig = load_rig(a.rig)
    n_cams = len(rig.cameras)
    # world mode: cameras in a tall grid on the LEFT, cloud + octomap stacked on the RIGHT
    if getattr(a, "world", False) and n_cams > 2:
        cols, rows = 2, (n_cams + 1) // 2
    else:
        cols = 3 if n_cams > 2 else n_cams
        rows = (n_cams + cols - 1) // cols
    cfg = TrackerConfig(frontend=a.frontend, init_mode=a.init_mode,
                        kf_dense_init_s=a.kf_dense_init, dyn_weight=a.dyn_weight,
                        px_sigma=float(getattr(rig, "px_sigma", 1.5) or 1.5))
    tracker = Tracker(rig, cfg)

    # capture per-camera features by wrapping the front-end
    last_feats = {"ff": None}
    orig = tracker.frontend.process
    def wrapped(t_ns, images, masks, dR):
        ff = orig(t_ns, images, masks, dR)
        last_feats["ff"] = ff
        return ff
    tracker.frontend.process = wrapped

    gt_t = gt_p = None
    if a.gt:
        gt_t, gt_p = load_tum(a.gt)

    cw = int(rig.cameras[0].size[0] * a.scale)
    ch = int(rig.cameras[0].size[1] * a.scale)
    MW = cols * cw                       # camera block width
    PR = int(MW * 1.05) if a.world else 0    # right column width
    W, H = MW + PR, rows * ch
    MH = 0                               # panels start at top of the right column
    densifier = occ = None
    world_pts = {}                       # 5 cm dedup: voxel key -> (x,y,z)
    world_ms = []
    if a.world:
        from podslam.densify import Densifier
        from podslam.occupancy import OccupancyGrid
        densifier = Densifier(rig, grid_step=a.densify_step, max_sigma=0.10, temporal=True)
        occ = OccupancyGrid(voxel=a.occ_voxel)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("VideoWriter failed to open (mp4v)")

    trails = [defaultdict(lambda: deque(maxlen=a.trail)) for _ in range(n_cams)]
    est_path: list[np.ndarray] = []
    cam_topics = {c.topic: i for i, c in enumerate(rig.cameras)}
    imu_topic = rig.imu.topic
    shift_ns = int(round(rig.cameras[0].time_shift_s * 1e9))
    acc_k = float(rig.imu.accel_scale)
    pending: dict[int, dict] = {}
    n_frames = 0
    t0_ns = None

    def hud(canvas, t_s, est):
        be = tracker.backend
        lines = [f"t {t_s:7.2f} s   {'KF' if est.keyframe else '  '}   {est.status}"]
        if est.T_W_I is not None:
            p = est.T_W_I[:3, 3]
            v = np.asarray(tracker.kf_navstate.velocity()) if tracker.kf_navstate is not None else np.zeros(3)
            lines.append(f"pos [{p[0]:+6.2f} {p[1]:+6.2f} {p[2]:+6.2f}] m")
            lines.append(f"vel [{v[0]:+5.2f} {v[1]:+5.2f} {v[2]:+5.2f}] |v| {np.linalg.norm(v):4.2f} m/s")
            if gt_t is not None:
                i = int(np.searchsorted(gt_t, t_s))
                if 0 < i < len(gt_t):
                    err = np.linalg.norm(p - gt_p[i])
                    lines.append(f"|err vs GT| {err*100:5.1f} cm")
        n_lm = est.n_landmarks
        dw = getattr(be, "n_downweighted", 0) if be is not None else 0
        lines.append(f"landmarks {n_lm}   downweighted {dw}   obs " + "/".join(str(x) for x in est.n_obs))
        # translucent dark panel behind the HUD, then the green text on top
        pw = max(cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0] for ln in lines) + 20
        ph = 26 * len(lines) + 12
        roi = canvas[4:4 + ph, 4:4 + pw]
        cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, dst=roi)
        y = 26
        for ln in lines:
            cv2.putText(canvas, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 255, 120), 1, cv2.LINE_AA)
            y += 26

    def inset(canvas, cur):
        size, m = 260, 12
        x0, y0 = W - size - m, H - size - m
        cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (140, 140, 140), 1)
        pts = [gt_p[:, :2]] if gt_p is not None else []
        if est_path:
            pts.append(np.asarray(est_path)[:, :2])
        if not pts:
            return
        allp = np.vstack(pts)
        lo, hi = allp.min(0) - 0.3, allp.max(0) + 0.3
        span = max((hi - lo).max(), 1e-3)
        def to_px(p):
            q = (p - lo) / span
            return (x0 + 8 + int(q[0] * (size - 16)), y0 + size - 8 - int(q[1] * (size - 16)))
        if gt_p is not None:
            for i in range(1, len(gt_p), 4):
                cv2.line(canvas, to_px(gt_p[i - 4, :2] if i >= 4 else gt_p[0, :2]), to_px(gt_p[i, :2]), (110, 110, 110), 1)
        if len(est_path) > 1:
            ep = np.asarray(est_path)
            for i in range(1, len(ep)):
                cv2.line(canvas, to_px(ep[i - 1, :2]), to_px(ep[i, :2]), (80, 255, 120), 2)
        if cur is not None:
            cv2.circle(canvas, to_px(cur[:2]), 4, (60, 60, 255), -1)
        cv2.putText(canvas, "top-down: est vs GT", (x0 + 6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def draw_world(canvas, t_s):
        import time as _t
        # shared orbit camera around the scene
        shown = [q for c_, q in world_pts.values() if c_ >= 2]
        allp = np.asarray(shown) if shown else np.zeros((0, 3))
        if len(allp) > 20:                      # robust framing: ignore stray far points
            lo = np.percentile(allp, 2, axis=0)
            hi = np.percentile(allp, 98, axis=0)
            ctr = (lo + hi) / 2
            ext = float(np.clip((hi - lo).max(), 3.0, 12.0))
        else:
            ctr, ext = np.zeros(3), 4.0
        az = 0.5 + t_s * 0.12
        el = 0.62
        ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
        R = np.array([[ca, sa, 0], [-sa * se, ca * se, ce], [-sa * ce, ca * ce, -se]])
        camd = 1.7 * ext
        f = (H // 2) * 0.95
        def proj(pw):
            q = R @ (np.asarray(pw) - ctr)
            z = q[2] + camd
            if z < 0.2:
                return None
            return int(f * q[0] / z), int(f * q[1] / z), z
        # right column, two stacked panels
        panels = [(MW, W, 0, H // 2, "live points (2-hit filtered)"),
                  (MW, W, H // 2, H, f"occupancy {a.occ_voxel*100:.0f} cm")]
        for x0, x1, y0_, y1_, label in panels:
            cv2.rectangle(canvas, (x0, y0_), (x1, y1_), (18, 14, 11), -1)
            cv2.line(canvas, (x0, y0_), (x0, y1_), (60, 60, 60), 1)
            cv2.line(canvas, (x0, y1_ - 1), (x1, y1_ - 1), (60, 60, 60), 1)
        def hcol(z):
            t = min(max((z - (ctr[2] - ext / 3)) / max(ext * 0.66, 1e-3), 0), 1)
            return (int(140 + 60 * (1 - t)), int(90 + 150 * t), int(40 + 40 * t))
        # top-right panel: points (same >=2-hit voxel filter the product map applies)
        cx0, cy0 = MW + PR // 2, H // 4
        for pw in shown:
            pr = proj(pw)
            if pr is None: continue
            x, y, _ = pr
            xx, yy = cx0 + x, cy0 + y
            if MW + 1 <= xx < W - 1 and 1 <= yy < H // 2 - 1:
                canvas[yy - 1:yy + 1, xx - 1:xx + 1] = hcol(pw[2])
        # bottom-right panel: occupied voxels as z-sorted squares
        cx1, cy1 = MW + PR // 2, 3 * H // 4
        cents, _odds = occ.occupied()
        if len(cents):
            prs = []
            for c in cents:
                pr = proj(c)
                if pr is None: continue
                prs.append((pr[2], cx1 + pr[0], cy1 + pr[1], c[2]))
            prs.sort(key=lambda r: -r[0])
            for z, x, y, h in prs:
                s_ = max(1, int(f * a.occ_voxel / z * 0.75))
                if MW <= x - s_ and x + s_ < W and H // 2 <= y - s_ and y + s_ < H:
                    col = hcol(h)
                    cv2.rectangle(canvas, (x - s_, y - s_), (x + s_, y + s_), col, -1)
                    cv2.rectangle(canvas, (x - s_, y - s_), (x + s_, y + s_), (25, 20, 16), 1)
        # trajectory + drone marker on both panels
        for cx, cy in ((cx0, cy0), (cx1, cy1)):
            if len(est_path) > 1:
                ep = est_path[::3]
                last = None
                for q in ep:
                    pr = proj(q)
                    if pr is None: continue
                    pt = (cx + pr[0], cy + pr[1])
                    if last is not None:
                        cv2.line(canvas, last, pt, (80, 255, 120), 1, cv2.LINE_AA)
                    last = pt
                if last is not None:
                    cv2.circle(canvas, last, 4, (60, 60, 255), -1)
        for (x0, x1, y0_, y1_, label) in panels:
            cv2.putText(canvas, label, (x0 + 10, y0_ + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        st = occ.stats()
        ms = np.mean(world_ms[-20:]) if world_ms else 0
        cv2.putText(canvas, f"pts {len(shown)}/{len(world_pts)}  vox occ {st['occupied']} free {st['free']}  {ms:.0f} ms/kf",
                    (MW + 10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 160, 170), 1, cv2.LINE_AA)

    def render(t_ns, group):
        nonlocal n_frames, t0_ns
        if t0_ns is None:
            t0_ns = t_ns
        imgs = [group[i] for i in range(n_cams)]
        est = tracker.track(t_ns, imgs)
        ff = last_feats["ff"]
        canvas = np.zeros((H, W, 3), np.uint8)
        for i in range(n_cams):
            tile = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR)
            if a.scale != 1.0:
                tile = cv2.resize(tile, (cw, ch))
            if ff is not None and i < len(ff.cams):
                cam = ff.cams[i]
                tr = trails[i]
                live = set()
                for tid, px in zip(cam.ids, cam.px):
                    tid = int(tid)
                    live.add(tid)
                    tr[tid].append((int(px[0] * a.scale), int(px[1] * a.scale)))
                for tid in [t for t in tr if t not in live]:
                    del tr[tid]
                for tid, q in tr.items():
                    pts = list(q)
                    for k in range(1, len(pts)):
                        cv2.line(tile, pts[k - 1], pts[k], (0, 170, 255), 1, cv2.LINE_AA)
                    cv2.circle(tile, pts[-1], 2, (80, 255, 120), -1, cv2.LINE_AA)
            cv2.putText(tile, rig.cameras[i].name, (8, ch - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            r, c = divmod(i, cols)
            canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = tile
        cur = None
        if est.T_W_I is not None:
            cur = est.T_W_I[:3, 3].copy()
            est_path.append(cur)
        if a.world and est.ok and est.keyframe and est.T_W_I is not None and densifier is not None:
            import time as _t
            w0 = _t.perf_counter()
            T = est.T_W_I
            pw = densifier.world_points(T, [tracker.condition(im) for im in imgs], tracker.static_masks)
            be = tracker.backend
            lm = []
            if be is not None:
                for v_ in be.lm_point.values():
                    v_ = np.asarray(v_, float).reshape(-1)
                    if v_.shape == (3,) and np.isfinite(v_).all():
                        lm.append(v_)
            lm = np.asarray(lm) if lm else np.zeros((0, 3))
            allw = np.vstack([pw, lm]) if len(pw) or len(lm) else np.zeros((0, 3))
            # occupancy gets only near-field points: range sigma grows ~z^2/baseline,
            # beyond ~4 m one fisheye-pair range is no longer voxel-scale evidence
            if len(allw):
                rng = np.linalg.norm(allw - T[:3, 3], axis=1)
                occ.integrate(T[:3, 3], allw[rng < 4.0])
            for q in allw:
                k = tuple(np.floor(q / 0.05).astype(int))
                e = world_pts.get(k)
                world_pts[k] = (e[0] + 1, q) if e else (1, q)
            world_ms.append((_t.perf_counter() - w0) * 1e3)
        if a.world:
            draw_world(canvas, (t_ns - t0_ns) * 1e-9)
        hud(canvas, (t_ns - t0_ns) * 1e-9, est)
        if not a.world:
            inset(canvas, cur)
        vw.write(canvas)
        n_frames += 1

    def flush(upto_ns):
        for t in sorted(pending):
            if upto_ns is not None and t > upto_ns - 100_000_000:
                break
            g = pending.pop(t)
            if all(i in g for i in range(n_cams)):
                render(t, g)

    with Reader(a.bag) as reader:
        conns = [c for c in reader.connections if c.topic in cam_topics or c.topic == imu_topic]
        for conn, _, raw in reader.messages(connections=conns):
            msg = TS.deserialize_ros1(raw, conn.msgtype)
            if conn.topic == imu_topic:
                g, ac = msg.angular_velocity, msg.linear_acceleration
                tracker.register_imu(stamp_ns(msg), [g.x, g.y, g.z], [ac.x * acc_k, ac.y * acc_k, ac.z * acc_k])
                continue
            t_ns = stamp_ns(msg) + shift_ns
            pending.setdefault(t_ns, {})[cam_topics[conn.topic]] = to_gray(msg)
            flush(t_ns)
            if a.max_frames and n_frames >= a.max_frames:
                break
        flush(None)
    vw.release()
    extra = ""
    if world_ms:
        extra = f" | world-model avg {np.mean(world_ms):.0f} ms/kf ({len(world_ms)} keyframes), occ {occ.stats()}"
    print(f"{a.out}: {n_frames} frames, {Path(a.out).stat().st_size / 1e6:.1f} MB{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
