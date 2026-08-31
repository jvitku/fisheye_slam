"""Minisim fast lane: analytic GPU/CPU raycaster over scene.py primitives.

Renders every camera of a rig along a trajectory (analytic from scene.py, or a
TUM file from a PX4 SITL flight), under a lighting/weather condition from
scene.CONDITIONS: day/night ambient+sun, rig-attached IR floods (inverse-square,
cone falloff, Lambert), Koschmieder fog with night flood backscatter, and
propwash dust particles near the body (bright flood backscatter blobs at night).

Output matches bench/frames2bag.py's contract:
    out/<cam>/frames.csv, <cam>/%06d.png, <cam>/%06d.depth.npy (depth cams)
    out/imu.csv      t_ns,gx,gy,gz,ax,ay,az   (noisy, rig IMU model)
    out/gt.tum       body pose at 100 Hz
    out/meta.yaml

Usage (CUDA torch lives in the 3dfe/cuvslam-ml container; CPU numpy-torch works
for short runs):
    python -m bench.minisim.render rigs/skydio3.yaml indoor day out/ [--traj f.tum]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.minisim import scene as S            # noqa: E402
from podslam.geometry import euler_to_R         # noqa: E402
from podslam.rig import T_MOUNT_OPTICAL, load_rig  # noqa: E402

NS = 1_000_000_000
T0_NS = 1_700_000_000 * NS                      # fixed epoch for all minisim bags


# ----------------------------------------------------------- torch textures

def hash3(q: torch.Tensor) -> torch.Tensor:
    h = (q[..., 0] * 374761393 + q[..., 1] * 668265263 + q[..., 2] * 2147483647) & 0x7FFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0x7FFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF).to(torch.float32) / float(0xFFFFFF)


def value_noise_t(p: torch.Tensor, scale: float) -> torch.Tensor:
    q = p / scale
    q0 = torch.floor(q).to(torch.int64)
    f = q - q0
    f = f * f * (3.0 - 2.0 * f)
    out = torch.zeros(p.shape[:-1], device=p.device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner = q0 + torch.tensor([dx, dy, dz], device=p.device)
                w = (f[..., 0] if dx else 1 - f[..., 0]) * \
                    (f[..., 1] if dy else 1 - f[..., 1]) * \
                    (f[..., 2] if dz else 1 - f[..., 2])
                out += w * hash3(corner)
    return out


def albedo_t(p: torch.Tensor, tex: dict) -> torch.Tensor:
    out = torch.full(p.shape[:-1], float(tex.get("base", 0.5)), device=p.device)
    for scale, amp in tex.get("octaves", ((0.4, 0.25), (0.08, 0.15))):
        out += amp * (value_noise_t(p, scale) - 0.5) * 2.0
    return out.clamp(0.02, 1.0)


# ------------------------------------------------------------ intersection

class Tracer:
    """Vectorised nearest-hit raycaster over the scene primitives."""

    def __init__(self, prims: list[dict], device):
        self.prims = prims
        self.dev = device
        self.boxes, self.rooms, self.cyls, self.grounds = [], [], [], []
        for i, pr in enumerate(prims):
            if pr["kind"] == "box":
                self.boxes.append((i, pr))
            elif pr["kind"] == "room":
                self.rooms.append((i, pr))
            elif pr["kind"] == "cylinder":
                self.cyls.append((i, pr))
            elif pr["kind"] == "ground":
                self.grounds.append((i, pr))
        if self.boxes:
            self.bc = torch.tensor(np.array([b["center"] for _, b in self.boxes]), dtype=torch.float32, device=device)
            self.bh = torch.tensor(np.array([b["half"] for _, b in self.boxes]), dtype=torch.float32, device=device)
        if self.cyls:
            self.cc = torch.tensor(np.array([c["center_xy"] for _, c in self.cyls]), dtype=torch.float32, device=device)
            self.cr = torch.tensor([c["radius"] for _, c in self.cyls], dtype=torch.float32, device=device)
            self.cz = torch.tensor(np.array([[c["z0"], c["z1"]] for _, c in self.cyls]), dtype=torch.float32, device=device)

    def trace(self, o: torch.Tensor, d: torch.Tensor):
        """o,d (N,3) -> t (N), normal (N,3), prim id (N; -1 = miss/sky)."""
        n = len(o)
        INF = 1e9
        t_best = torch.full((n,), INF, device=self.dev)
        nrm = torch.zeros((n, 3), device=self.dev)
        pid = torch.full((n,), -1, dtype=torch.int64, device=self.dev)
        dsafe = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)

        if self.boxes:                                   # slab test, (N,P)
            t1 = (self.bc - self.bh - o[:, None]) / dsafe[:, None]
            t2 = (self.bc + self.bh - o[:, None]) / dsafe[:, None]
            tmin, tmax = torch.minimum(t1, t2), torch.maximum(t1, t2)
            tn, ax = tmin.max(-1)
            tf = tmax.min(-1).values
            hit = (tn > 1e-3) & (tn < tf)
            tn = torch.where(hit, tn, torch.full_like(tn, INF))
            tb, pb = tn.min(-1)
            upd = tb < t_best
            if upd.any():
                axis = ax[torch.arange(n, device=self.dev), pb]
                sgn = -torch.sign(torch.gather(d, 1, axis[:, None]))[:, 0]
                nb = torch.zeros((n, 3), device=self.dev)
                nb.scatter_(1, axis[:, None], sgn[:, None])
                t_best = torch.where(upd, tb, t_best)
                nrm = torch.where(upd[:, None], nb, nrm)
                ids = torch.tensor([i for i, _ in self.boxes], device=self.dev)
                pid = torch.where(upd, ids[pb], pid)

        for i, pr in self.rooms:                         # interior of AABB
            c = torch.tensor(pr["center"], dtype=torch.float32, device=self.dev)
            h = torch.tensor(pr["half"], dtype=torch.float32, device=self.dev)
            tex = (c + torch.sign(dsafe) * h - o) / dsafe
            te, ax = tex.min(-1)
            ok = (te > 1e-3) & (te < t_best)
            if ok.any():
                sgn = -torch.sign(torch.gather(d, 1, ax[:, None]))[:, 0]
                nb = torch.zeros((n, 3), device=self.dev)
                nb.scatter_(1, ax[:, None], sgn[:, None])
                t_best = torch.where(ok, te, t_best)
                nrm = torch.where(ok[:, None], nb, nrm)
                pid = torch.where(ok, torch.full_like(pid, i), pid)

        if self.cyls:                                    # vertical side surface
            oxy = o[:, None, :2] - self.cc[None]
            dxy = d[:, None, :2]
            a = (dxy * dxy).sum(-1).clamp_min(1e-12)
            b = 2 * (oxy * dxy).sum(-1)
            cq = (oxy * oxy).sum(-1) - self.cr[None] ** 2
            disc = b * b - 4 * a * cq
            ok = disc > 0
            tt = torch.where(ok, (-b - torch.sqrt(disc.clamp_min(0))) / (2 * a), torch.full_like(b, INF))
            z = o[:, None, 2] + tt * d[:, None, 2]
            ok = ok & (tt > 1e-3) & (z >= self.cz[None, :, 0]) & (z <= self.cz[None, :, 1])
            tt = torch.where(ok, tt, torch.full_like(tt, INF))
            tc, pc = tt.min(-1)
            upd = tc < t_best
            if upd.any():
                hitp = o + tc[:, None] * d
                cxy = self.cc[pc]
                nb = torch.zeros((n, 3), device=self.dev)
                nb[:, :2] = hitp[:, :2] - cxy
                nb = nb / nb.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                ids = torch.tensor([i for i, _ in self.cyls], device=self.dev)
                t_best = torch.where(upd, tc, t_best)
                nrm = torch.where(upd[:, None], nb, nrm)
                pid = torch.where(upd, ids[pc], pid)

        for i, pr in self.grounds:
            tg = (pr["z"] - o[:, 2]) / dsafe[:, 2]
            ok = (tg > 1e-3) & (d[:, 2] < 0) & (tg < t_best)
            if ok.any():
                t_best = torch.where(ok, tg, t_best)
                nrm = torch.where(ok[:, None], torch.tensor([0.0, 0.0, 1.0], device=self.dev).expand(n, 3), nrm)
                pid = torch.where(ok, torch.full_like(pid, i), pid)

        return t_best, nrm, pid


# ----------------------------------------------------------------- shading

SUN_DIR = np.array([0.35, 0.25, -0.9]) / np.linalg.norm([0.35, 0.25, -0.9])


def shade(tracer, prims, o, d, cond, floods_w, device, chunk=250_000):
    """Radiance + depth for rays (N,3). floods_w: [(pos_w, dir_w, cone_cos, I)]."""
    N = len(o)
    rad = torch.zeros(N, device=device)
    depth = torch.zeros(N, device=device)
    sun = torch.tensor(-SUN_DIR, dtype=torch.float32, device=device)
    beta = cond["fog_beta"]
    for s in range(0, N, chunk):
        oc, dc = o[s:s + chunk], d[s:s + chunk]
        t, nrm, pid = tracer.trace(oc, dc)
        hit = pid >= 0
        p = oc + t[:, None] * dc
        alb = torch.zeros(len(oc), device=device)
        for i, pr in enumerate(prims):
            m = pid == i
            if m.any():
                alb[m] = albedo_t(p[m], pr["tex"])
        L = torch.full((len(oc),), float(cond["ambient"]), device=device)
        if cond["sun"] > 0:
            L = L + cond["sun"] * (nrm * sun).sum(-1).clamp_min(0)
        for fp, fd, ccos, inten in floods_w:
            v = p - fp                                    # flood -> surface
            dist2 = (v * v).sum(-1).clamp_min(0.05)
            vn = v / dist2.sqrt()[:, None]
            ang = (vn * fd).sum(-1)
            cone = ((ang - ccos) / max(1e-3, 1 - ccos)).clamp(0, 1)
            lam = (-(nrm * vn).sum(-1)).clamp_min(0)
            L = L + inten * cone * lam / dist2
        col = alb * L
        sky = 0.75 * cond["ambient"] + 0.02
        col = torch.where(hit, col, torch.full_like(col, sky))
        tt = torch.where(hit, t, torch.full_like(t, 60.0))
        if beta > 0:                                      # Koschmieder + night backscatter
            tr = torch.exp(-beta * tt)
            if cond["flood"] and floods_w:
                A = torch.zeros(len(oc), device=device)
                M = 8
                for k in range(M):
                    ts = tt * (k + 0.5) / M
                    ps = oc + ts[:, None] * dc
                    for fp, fd, ccos, inten in floods_w:
                        v = ps - fp
                        dist2 = (v * v).sum(-1).clamp_min(0.02)
                        vn = v / dist2.sqrt()[:, None]
                        cone = (((vn * fd).sum(-1) - ccos) / max(1e-3, 1 - ccos)).clamp(0, 1)
                        A = A + 0.10 * beta * inten * cone / dist2 * torch.exp(-beta * ts) * (tt / M)
                col = col * tr + A
            else:
                col = col * tr + 0.65 * (1 - tr)
        rad[s:s + chunk] = col
        depth[s:s + chunk] = torch.where(hit, t, torch.zeros_like(t))
    return rad, depth


def splat_dust(img, dust, T_cam_pod, model, cond, flood_pods, device):
    """Alpha/additive gaussian splats of body-frame dust particles."""
    R, tr = T_cam_pod[:3, :3], T_cam_pod[:3, 3]
    pc = (R @ dust.p.T).T + tr                            # particles in cam optical
    infront = pc[:, 2] > 0.02
    if not infront.any():
        return img
    px, ok = model.project(pc[infront])
    r = dust.r[infront]
    dcam = np.linalg.norm(pc[infront], axis=1)
    H, W = img.shape
    if cond["flood"]:                                     # backscatter brightness
        bright = np.zeros(len(r))
        for fp, fd, ccos, inten in flood_pods:
            v = dust.p[infront] - fp
            d2 = (v * v).sum(1).clip(0.01, None)
            vn = v / np.sqrt(d2)[:, None]
            cone = np.clip(((vn * fd).sum(1) - ccos) / max(1e-3, 1 - ccos), 0, 1)
            bright += 2.2 * inten * cone / d2 * (r / 0.002) ** 2 / np.clip(dcam, 0.05, None) ** 2 * 1e-3
        alpha = np.clip(bright * 4, 0, 0.95)
        val = np.clip(bright, 0, 1.0)
    else:
        alpha = np.clip(0.35 * (r / 0.002) / np.clip(dcam, 0.08, None), 0.05, 0.6)
        val = np.full(len(r), 0.45)
    sig = np.clip(getattr(model, "fx", 150.0) * r * 2.5 / np.clip(dcam, 0.05, None), 0.6, 12.0)
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    for k in np.argsort(-dcam):
        if not ok[k]:
            continue
        u, v = px[k]
        if u < -30 or u > W + 30 or v < -30 or v > H + 30:
            continue
        s = float(sig[k])
        r0, r1 = max(0, int(v - 3 * s)), min(H, int(v + 3 * s) + 1)
        c0, c1 = max(0, int(u - 3 * s)), min(W, int(u + 3 * s) + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        g = torch.exp(-(((xx[r0:r1, c0:c1] - u) ** 2 + (yy[r0:r1, c0:c1] - v) ** 2) / (2 * s * s)))
        a = float(alpha[k]) * g
        img[r0:r1, c0:c1] = img[r0:r1, c0:c1] * (1 - a) + float(val[k]) * a
    return img


# ------------------------------------------------------------- trajectories

class TumTrajectory:
    """Interpolating trajectory from a TUM file (PX4 SITL flights plug in here)."""

    def __init__(self, path):
        d = np.loadtxt(path)
        self.t = d[:, 0] - d[0, 0]
        self.p = d[:, 1:4]
        self.q = d[:, 4:8]
        self.dur = float(self.t[-1])

    def __call__(self, t):
        from scipy.spatial.transform import Rotation, Slerp
        t = np.clip(t, 0, self.dur)
        i = np.searchsorted(self.t, t) - 1
        i = int(np.clip(i, 0, len(self.t) - 2))
        a = (t - self.t[i]) / max(1e-9, self.t[i + 1] - self.t[i])
        p = (1 - a) * self.p[i] + a * self.p[i + 1]
        R = Slerp([0, 1], Rotation.from_quat(self.q[i:i + 2]))(a).as_matrix()
        return R, p


def synth_imu(traj, dur, rig, seed, out_csv):
    """True IMU from the trajectory + the rig's noise model, at the rig rate."""
    rate = rig.imu.rate_hz
    dt = 1.0 / rate
    rng = np.random.default_rng(seed)
    bg = np.zeros(3)
    ba = np.zeros(3)
    rows = []
    for k in range(int(dur * rate)):
        t = k * dt
        gyro, accel = S.imu_from_trajectory(traj, max(t, 2 * 1e-4))
        bg += rng.normal(0, rig.imu.gyro_random_walk * np.sqrt(dt), 3)
        ba += rng.normal(0, rig.imu.accel_random_walk * np.sqrt(dt), 3)
        g = gyro + bg + rng.normal(0, rig.imu.gyro_noise_density * np.sqrt(rate), 3)
        a = accel + ba + rng.normal(0, rig.imu.accel_noise_density * np.sqrt(rate), 3)
        rows.append([T0_NS + int(t * NS), *g, *a])
    with open(out_csv, "w") as f:
        f.write("t_ns,gx,gy,gz,ax,ay,az\n")
        for r in rows:
            f.write(f"{r[0]}," + ",".join(f"{x:.9g}" for x in r[1:]) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rig"); ap.add_argument("scene", choices=list(S.SCENES)); ap.add_argument("condition", choices=list(S.CONDITIONS)); ap.add_argument("out")
    ap.add_argument("--traj", default=None, help="TUM trajectory (e.g. PX4 SITL flight); default analytic")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = scene/traj default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--depth-every", type=int, default=2, help="depth frame decimation for depth: true cams")
    ap.add_argument("--exposure", type=float, default=1.0)
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    rig = load_rig(a.rig)
    raw = yaml.safe_load(open(a.rig))
    cond = S.CONDITIONS[a.condition]
    scene_fn, traj_fn, default_dur = S.SCENES[a.scene]
    prims = scene_fn()
    traj = TumTrajectory(a.traj) if a.traj else traj_fn
    dur = a.duration or (traj.dur if a.traj else default_dur)
    tracer = Tracer(prims, device)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # illuminators in pod frame
    floods_pod = []
    if cond["flood"]:
        for il in raw.get("illuminators", []) or ([raw["illuminator"]] if "illuminator" in raw else []):
            m = il["mount"]
            fd = euler_to_R(m["rpy_deg"]) @ np.array([1.0, 0, 0])
            floods_pod.append((np.array(m["position"], float), fd,
                               float(np.cos(np.deg2rad(il.get("cone_half_angle_deg", 60)))),
                               float(il.get("intensity", 30000)) * 1e-4))
    # camera ray grids (optical frame), pod-frame extrinsics
    cams = []
    for c in rig.cameras:
        w, h = c.size
        xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays, ok = c.model.unproject(np.stack([xs.ravel(), ys.ravel()], 1))
        mask = c.circle_mask()
        if mask is not None:
            ok = ok & (mask.ravel() > 0)
        T_pod_cam = rig.imu.T_body_imu @ c.T_imu_cam
        cams.append(dict(cam=c, rays=torch.tensor(rays, dtype=torch.float32, device=device),
                         ok=torch.tensor(ok.copy(), device=device), T_pod_cam=T_pod_cam,
                         depth=c.depth_topic is not None))
        (out / c.name).mkdir(exist_ok=True)

    n_frames = int(dur * a.fps)
    dust = S.DustField(seed=a.seed) if cond["dust"] else None
    t_render0 = time.time()
    fcsv = {c["cam"].name: open(out / c["cam"].name / "frames.csv", "w") for c in cams}
    for f in fcsv.values():
        f.write("index,t_ns\n")
    with open(out / "gt.tum", "w") as fgt:                # 100 Hz body GT
        from scipy.spatial.transform import Rotation
        for k in range(int(dur * 100)):
            t = k / 100.0
            R, p = traj(t)
            q = Rotation.from_matrix(R).as_quat()
            fgt.write(f"{(T0_NS + int(t * NS)) / NS:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                      f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
    synth_imu(traj, dur, rig, a.seed + 77, out / "imu.csv")

    for k in range(n_frames):
        t = k / a.fps
        t_ns = T0_NS + int(t * NS)
        if dust is not None:
            dust.step(1.0 / a.fps)
        done = all((out / c["cam"].name / f"{k:06d}.png").exists()
                   and (not c["depth"] or k % a.depth_every or (out / c["cam"].name / f"{k:06d}.depth.npy").exists())
                   for c in cams)
        R_wp, p_wp = traj(t)
        R_wp_t = torch.tensor(R_wp, dtype=torch.float32, device=device)
        p_wp_t = torch.tensor(p_wp, dtype=torch.float32, device=device)
        floods_w = []
        for fp, fd, ccos, inten in floods_pod:
            floods_w.append((R_wp_t @ torch.tensor(fp, dtype=torch.float32, device=device) + p_wp_t,
                             R_wp_t @ torch.tensor(fd, dtype=torch.float32, device=device), ccos, inten))
        for ci, c in enumerate(cams):
            fcsv[c["cam"].name].write(f"{k},{t_ns}\n")
            if done:
                continue
            w, h = c["cam"].size
            Rpc = torch.tensor(c["T_pod_cam"][:3, :3], dtype=torch.float32, device=device)
            ppc = torch.tensor(c["T_pod_cam"][:3, 3], dtype=torch.float32, device=device)
            d_w = (R_wp_t @ (Rpc @ c["rays"].T)).T
            o_w = (R_wp_t @ ppc + p_wp_t).expand(len(d_w), 3)
            rad, dep = shade(tracer, prims, o_w, d_w, cond, floods_w, device, a.chunk)
            img = rad.reshape(h, w)
            if dust is not None:
                T_cam_pod = np.linalg.inv(c["T_pod_cam"])
                img = splat_dust(img, dust, T_cam_pod, c["cam"].model, cond, floods_pod, device)
            img = torch.where(c["ok"].reshape(h, w), img, torch.zeros_like(img))
            # exposure + gamma + sensor noise
            g = (a.exposure * img).clamp_min(0) ** (1 / 2.2)
            noise = torch.randn(g.shape, device=device) * (0.004 + 0.012 * torch.sqrt(g.clamp_min(0)))
            u8 = ((g + noise).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            Image.fromarray(u8, "L").save(out / c["cam"].name / f"{k:06d}.png", compress_level=1)
            if c["depth"] and k % a.depth_every == 0:
                np.save(out / c["cam"].name / f"{k:06d}.depth.npy",
                        dep.reshape(h, w).cpu().numpy().astype(np.float32))
        if k % 50 == 0:
            el = time.time() - t_render0
            print(f"frame {k}/{n_frames}  {el:.0f}s  ({el / max(k, 1):.2f} s/frame)", flush=True)
    for f in fcsv.values():
        f.close()
    yaml.safe_dump(dict(rig=a.rig, scene=a.scene, condition=a.condition, fps=a.fps,
                        duration=dur, seed=a.seed, traj=a.traj or "analytic"),
                   open(out / "meta.yaml", "w"))
    print(f"rendered {n_frames} frames x {len(cams)} cams -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
