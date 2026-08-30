"""End-to-end estimator test on a synthetic world: a KB4 stereo rig flies a known
trajectory past random 3D points; a 'perfect' front-end supplies the true
correspondences (0.3 px noise); the IMU is derived from the trajectory. The
recovered IMU trajectory must match ground truth to centimetres. This checks
the whole chain — extrinsic conventions, preintegration, projection factors,
triangulation, fixed-lag smoothing — without any image processing."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gtsam")

from podslam.frontend.base import CamObs, FrameFeatures, Frontend
from podslam.geometry import R_to_quat_xyzw, T_from_Rt, euler_to_R, exp_so3, inv_T
from podslam.rig import Camera, Imu, Rig, R_MOUNT_OPTICAL
from podslam.tracker import Tracker, TrackerConfig
from tools.fisheye.models import KannalaBrandt4

RATE_CAM, RATE_IMU, G = 20.0, 200.0, 9.81


def make_rig():
    """TUM-VI-like stereo: two KB4 cameras 10 cm apart, both looking along body +x."""
    cams = []
    for name, y in (("cam0", 0.05), ("cam1", -0.05)):
        T_imu_cam = T_from_Rt(R_MOUNT_OPTICAL, [0.0, y, 0.0])
        cams.append(Camera(name=name, model=KannalaBrandt4(fx=190.0, fy=190.0, cx=256.0, cy=256.0), size=(512, 512),
                           T_imu_cam=T_imu_cam, topic=f"/{name}", fov_deg=190.0, fx=190.0))
    imu = Imu(topic="/imu0", rate_hz=RATE_IMU, gyro_noise_density=1.6e-4, gyro_random_walk=2.2e-5,
              accel_noise_density=2.8e-3, accel_random_walk=8.6e-4)
    return Rig(name="synthetic", cameras=cams, imu=imu)


def trajectory(t):
    """Body pose: still for 1.5 s, then a circle (r = 1.5 m, 0.5 rad/s) with yaw following
    the tangent, plus a gentle bob in z."""
    tau = max(0.0, t - 1.5)
    s = tau * tau / (2 * 2.0) if tau < 2.0 else tau - 1.0          # smooth start
    w = 0.5
    p = np.array([1.5 * np.cos(w * s) - 1.5, 1.5 * np.sin(w * s), 1.0 + 0.1 * np.sin(0.8 * s)])
    yaw = w * s + np.pi / 2
    R = euler_to_R([5.0 * np.sin(0.7 * s), 3.0 * np.sin(0.9 * s), np.degrees(yaw)])
    return R, p


def imu_from_trajectory(t, dt=1e-3):
    R, p = trajectory(t)
    Rp, pp = trajectory(t - dt); Rn, pn = trajectory(t + dt)
    acc_w = (pn - 2 * p + pp) / (dt * dt)
    # angular velocity in body frame from R^T dR/dt
    dR = R.T @ (Rn - Rp) / (2 * dt)
    w = np.array([dR[2, 1], dR[0, 2], dR[1, 0]])
    a_body = R.T @ (acc_w + np.array([0, 0, G]))
    return w, a_body


class PerfectFrontend(Frontend):
    def __init__(self, rig, points, noise_px=0.3, seed=0):
        super().__init__(rig)
        self.points = points; self.noise = noise_px; self.rng = np.random.default_rng(seed)
        self.pose = None

    def set_pose(self, R, p):
        self.pose = (R, p)

    def process(self, t_ns, images, masks, dR_imu):
        R_wb, p_wb = self.pose
        cams = []
        for cam in self.rig.cameras:
            T_w_c = T_from_Rt(R_wb, p_wb) @ cam.T_imu_cam
            T_c_w = inv_T(T_w_c)
            pc = (T_c_w[:3, :3] @ self.points.T).T + T_c_w[:3, 3]
            px, valid = cam.model.project(pc)
            theta = np.degrees(np.arctan2(np.hypot(pc[:, 0], pc[:, 1]), pc[:, 2]))
            valid &= (theta < 75) & (pc[:, 2] > 0.3) & (np.linalg.norm(pc, axis=1) < 12)
            px = px + self.rng.normal(0, self.noise, px.shape)
            b, bvalid = cam.model.unproject(px)
            valid &= bvalid & np.all(np.isfinite(b), axis=1)
            ids = np.nonzero(valid)[0].astype(np.int64)
            cams.append(CamObs(ids=ids, px=px[valid].astype(np.float32), bearings=b[valid]))
        return FrameFeatures(t_ns=t_ns, cams=cams)


def align(est, gt):
    mu_e, mu_g = est.mean(0), gt.mean(0)
    H = (est - mu_e).T @ (gt - mu_g)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return (R @ est.T).T + (mu_g - R @ mu_e)


def test_synthetic_stereo_vio_recovers_trajectory():
    rig = make_rig()
    rng = np.random.default_rng(1)
    # points on the walls/floor/ceiling of a 8 x 8 x 3 m room around the circle
    pts = np.concatenate([
        np.stack([rng.uniform(-5, 3, 300), rng.uniform(-4, 4, 300), rng.uniform(0, 3, 300)], 1),
        np.stack([np.full(150, 3.5), rng.uniform(-4, 4, 150), rng.uniform(0, 3, 150)], 1),
        np.stack([rng.uniform(-5, 3, 150), np.full(150, 4.5), rng.uniform(0, 3, 150)], 1),
    ])
    fe = PerfectFrontend(rig, pts)
    tracker = Tracker(rig, TrackerConfig(preprocess="none", circle_mask=False, kf_every=2, lag_s=3.0), frontend=fe)
    duration = 14.0
    gt, est = [], []
    imu_t = np.arange(0.0, duration, 1 / RATE_IMU)
    cam_t = np.arange(0.0, duration, 1 / RATE_CAM)
    ii = 0
    for tc in cam_t:
        while ii < len(imu_t) and imu_t[ii] <= tc:
            w, a = imu_from_trajectory(imu_t[ii])
            w = w + rng.normal(0, 1e-3, 3) + np.array([0.002, -0.001, 0.0015])   # noise + small bias
            a = a + rng.normal(0, 5e-3, 3)
            tracker.register_imu(int(round(imu_t[ii] * 1e9)), w, a)
            ii += 1
        R, p = trajectory(tc)
        fe.set_pose(R, p)
        e = tracker.track(int(round(tc * 1e9)), [np.zeros((512, 512), np.uint8)] * 2)
        if e.ok and e.T_W_I is not None:
            gt.append(p); est.append(e.T_W_I[:3, 3])
    gt, est = np.array(gt), np.array(est)
    assert len(est) > 200
    ate = np.sqrt(np.mean(np.sum((align(est, gt) - gt) ** 2, axis=1)))
    print(f"synthetic ATE {ate * 100:.2f} cm over {len(est)} frames, resets {tracker.backend.n_resets}")
    assert ate < 0.03                       # 3 cm on a 14 s flight with 0.3 px noise
    assert tracker.backend.n_resets == 0
