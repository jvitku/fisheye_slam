"""PyCuVSLAM stereo(+IMU) tracking on a bag-contract ROS1 bag -> TUM + depth dump.

Feeds the synced stereo pair (and the IMU, Inertial mode) to cuVSLAM and writes:
    <out>/est.tum            time x y z qx qy qz qw  — pose of the POD IMU frame
                             (rig frame = cam0 optical; converted with
                             rig_from_imu so it compares 1:1 with the ground
                             truth from bench/split_bag.py and with OpenVINS)
    <out>/depth/NNNNNN.npy   (optional --dump-depth) depth [m] + poses.txt —
                             the nvblox fuser input (see run_cuvslam.sh)

Written against the cuvslam 17.0.0 wheel (introspected, not the upstream
examples, which are ahead of the wheel): Tracker.OdometryConfig /
Tracker.OdometryMode.Inertial, Distortion.Model.Pinhole, Rig with cam0 as the
rig origin, ImuCalibration.rig_from_imu, track() -> (PoseEstimate, slam_pose)
with PoseEstimate.world_from_rig (PoseWithCovariance or None on failure).

Usage (inside 3dfe/cuvslam, --gpus all):
    python3 cuvslam_track.py input.bag out/ --rig /rigs/oakdpro.yaml
        [--dump-depth] [--no-imu] [--unrectified]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

import cuvslam

TS = get_typestore(Stores.ROS1_NOETIC)

# Bag contract defaults (docs/oak_d_pro_slam.md §4); a rig yaml may override
# them per camera (`topic`, `depth_topic`) and for the IMU (`topic`), e.g.
# rigs/tumvi_room1.yaml for the real TUM-VI bags.
DEFAULT_LEFT, DEFAULT_RIGHT = "/uav1/cam0/color/image_raw", "/uav1/cam1/color/image_raw"
DEFAULT_DEPTH = "/uav1/cam0/depth/image_raw"
DEFAULT_IMU = "/uav1/sensor_pod/imu"

# Camera FLU mount frame (rig yaml: rpy 0 = optical axis along +x) -> optical
# frame (x right, y down, z forward); same constant as rig_math.R_MOUNT_OPTICAL.
T_FLU_OPT = np.eye(4)
T_FLU_OPT[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]


def euler_to_R(rpy_deg) -> np.ndarray:
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mount_to_T(mount: dict | None) -> np.ndarray:
    T = np.eye(4)
    if mount:
        T[:3, :3] = euler_to_R(mount["rpy_deg"])
        T[:3, 3] = mount["position"]
    return T


def R_to_quat_xyzw(m: np.ndarray) -> list[float]:
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return (q / np.linalg.norm(q)).tolist()


def quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def T_to_pose(T: np.ndarray) -> "cuvslam.Pose":
    return cuvslam.Pose(rotation=R_to_quat_xyzw(T[:3, :3]), translation=T[:3, 3].tolist())


def make_rig(rig_yaml: str):
    """rig yaml -> (cuvslam.Rig, T_rig_imu, topics).

    Rig frame = cam0 optical frame (PyCuVSLAM convention). Two rig flavours:
      sim rigs (rigs/oakdpro.yaml): FLU pod-relative `mount`s — the pod mount
        cancels out; camera optical = mount . T_FLU_OPT; IMU frame = its mount
      real rigs (rigs/tumvi_room1.yaml): Kalibr `T_cam_imu` per camera
        (IMU -> camera optical); IMU frame = the calibration's IMU frame
    """
    rig = yaml.safe_load(open(rig_yaml))
    cams_yaml = rig["cameras"]
    if all("T_cam_imu" in c for c in cams_yaml):
        # reference frame = the IMU: T_ref_cam = inv(T_cam_imu)
        T_ref_cam = [np.linalg.inv(np.asarray(c["T_cam_imu"], dtype=np.float64)) for c in cams_yaml]
        T_ref_imu = np.eye(4)
    else:
        T_ref_cam = [mount_to_T(c["mount"]) @ T_FLU_OPT for c in cams_yaml]
        T_ref_imu = mount_to_T(rig["imu"].get("mount"))
    T_rig_ref = np.linalg.inv(T_ref_cam[0])
    T_pod_cam = T_ref_cam
    T_rig_pod = T_rig_ref

    cams = []
    for c, T_pc in zip(cams_yaml, T_pod_cam):
        intr = c["intrinsics"]
        cam = cuvslam.Camera()
        cam.size = tuple(int(v) for v in c["resolution"])
        cam.focal = (float(intr["fx"]), float(intr["fy"]))
        cam.principal = (float(intr["cx"]), float(intr["cy"]))
        if c["model"] == "kb4":
            cam.distortion = cuvslam.Distortion(
                cuvslam.Distortion.Model.Fisheye,
                [float(intr.get(k, 0.0)) for k in ("k1", "k2", "k3", "k4")])
        elif any(intr.get(k) for k in ("k1", "k2", "p1", "p2")):
            cam.distortion = cuvslam.Distortion(
                cuvslam.Distortion.Model.Brown,
                [float(intr.get(k, 0.0)) for k in ("k1", "k2", "k3", "p1", "p2")])
        else:   # ideal pinhole (sim rig / rectified device output)
            cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole, [])
        cam.rig_from_camera = T_to_pose(T_rig_pod @ T_pc)
        cams.append(cam)

    imu_yaml = rig["imu"]
    T_rig_imu = T_rig_pod @ T_ref_imu
    imu = cuvslam.ImuCalibration()
    imu.rig_from_imu = T_to_pose(T_rig_imu)
    imu.gyroscope_noise_density = float(imu_yaml.get("gyro_noise_density", 1e-4))
    imu.gyroscope_random_walk = float(imu_yaml.get("gyro_random_walk", 1e-5))
    imu.accelerometer_noise_density = float(imu_yaml.get("accel_noise_density", 1e-3))
    imu.accelerometer_random_walk = float(imu_yaml.get("accel_random_walk", 1e-4))
    imu.frequency = float(imu_yaml.get("rate_hz", 200))

    r = cuvslam.Rig()
    r.cameras = cams
    r.imus = [imu]
    topics = {
        "left": cams_yaml[0].get("topic", DEFAULT_LEFT),
        "right": cams_yaml[1].get("topic", DEFAULT_RIGHT),
        "depth": cams_yaml[0].get("depth_topic", DEFAULT_DEPTH),
        "imu": imu_yaml.get("topic", DEFAULT_IMU),
    }
    return r, T_rig_imu, topics


def to_gray(msg) -> np.ndarray:
    """sensor_msgs/Image (mono8 | rgb8 | bgr8) -> uint8 grayscale array."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "mono8":
        return data.reshape(msg.height, msg.width).copy()
    if msg.encoding in ("rgb8", "bgr8"):
        rgb = data.reshape(msg.height, msg.width, 3).astype(np.float32)
        r, g, b = (rgb[..., 0], rgb[..., 1], rgb[..., 2]) if msg.encoding == "rgb8" \
            else (rgb[..., 2], rgb[..., 1], rgb[..., 0])
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


def stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--dump-depth", action="store_true",
                    help="also dump depth frames + tracked poses for nvblox")
    ap.add_argument("--no-imu", action="store_true",
                    help="visual-only stereo odometry (OdometryMode.Multicamera)")
    ap.add_argument("--unrectified", action="store_true",
                    help="the pair is NOT rectified (sim ideal pinhole and "
                         "record_oak.py rectified output both are)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rig, T_rig_imu, topics_map = make_rig(args.rig)
    LEFT, RIGHT, DEPTH, IMU = (topics_map[k] for k in ("left", "right", "depth", "imu"))
    Mode = cuvslam.Tracker.OdometryMode
    mode = Mode.Multicamera if args.no_imu else Mode.Inertial
    cfg = cuvslam.Tracker.OdometryConfig()
    cfg.async_sba = False                   # deterministic offline replay
    cfg.enable_observations_export = False
    cfg.enable_final_landmarks_export = False
    cfg.rectified_stereo_camera = not args.unrectified
    cfg.odometry_mode = mode
    tracker = cuvslam.Tracker(rig, cfg)
    print(f"cuVSLAM {getattr(cuvslam, '__version__', '?')} tracker: mode={mode}, "
          f"rectified={not args.unrectified}")

    tum = (args.out / "est.tum").open("w")
    depth_dir = args.out / "depth"
    poses_txt = None
    if args.dump_depth:
        depth_dir.mkdir(exist_ok=True)
        poses_txt = (depth_dir / "poses.txt").open("w")

    pending: dict[int, dict[str, np.ndarray]] = {}
    n_frames = n_valid = n_imu = 0
    with Reader(args.bag) as reader:
        topics = {c.topic for c in reader.connections}
    want_depth = args.dump_depth and DEPTH in topics
    LAG_NS = 100_000_000   # a group is complete when 100 ms newer data has arrived

    def process(t_ns: int, group: dict) -> None:
        nonlocal n_frames, n_valid
        estimate, _ = tracker.track(t_ns, [group["l"], group["r"]])
        n_frames += 1
        if estimate.world_from_rig is None:
            return
        n_valid += 1
        p = estimate.world_from_rig.pose
        T_w_rig = np.eye(4)
        T_w_rig[:3, :3] = quat_xyzw_to_R(list(p.rotation))
        T_w_rig[:3, 3] = list(p.translation)
        T_w_imu = T_w_rig @ T_rig_imu                # report the pod IMU frame
        t = t_ns / 1e9
        x, y, z = T_w_imu[:3, 3]
        qx, qy, qz, qw = R_to_quat_xyzw(T_w_imu[:3, :3])
        tum.write(f"{t:.6f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
        if want_depth and "depth" in group:
            np.save(depth_dir / f"{n_frames:06d}.npy", group["depth"])
            poses_txt.write(f"{n_frames:06d} {t:.6f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

    def flush(upto_ns: int | None) -> None:
        """Track every complete group older than `upto_ns` (all, if None), in time order."""
        for t in sorted(pending):
            if upto_ns is not None and t > upto_ns:
                break
            group = pending[t]
            complete = "l" in group and "r" in group and (not want_depth or "depth" in group)
            if complete or upto_ns is None or t < upto_ns - LAG_NS:
                if "l" in group and "r" in group:
                    process(t, group)
                del pending[t]

    with Reader(args.bag) as reader:
        conns = [c for c in reader.connections if c.topic in (LEFT, RIGHT, DEPTH, IMU)]
        for conn, _, raw in reader.messages(connections=conns):
            msg = TS.deserialize_ros1(raw, conn.msgtype)
            if conn.topic == IMU:
                if not args.no_imu:
                    m = cuvslam.ImuMeasurement()
                    m.timestamp_ns = stamp_ns(msg)
                    g, a = msg.angular_velocity, msg.linear_acceleration
                    m.angular_velocities = np.asarray([g.x, g.y, g.z], dtype=np.float64)
                    m.linear_accelerations = np.asarray([a.x, a.y, a.z], dtype=np.float64)
                    tracker.register_imu_measurement(0, m)
                    n_imu += 1
                continue

            t_ns = stamp_ns(msg)
            if conn.topic == DEPTH:
                if want_depth:
                    d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                    pending.setdefault(t_ns, {})["depth"] = d.copy()
            else:
                pending.setdefault(t_ns, {})["l" if conn.topic == LEFT else "r"] = to_gray(msg)
            # IMU samples must precede the frames that use them, so only frames
            # older than the newest data (by the lag) are tracked here.
            flush(t_ns)
        flush(None)

    tum.close()
    if poses_txt:
        poses_txt.close()
    print(f"tracked {n_valid}/{n_frames} stereo frames ({n_imu} IMU samples) "
          f"-> {args.out / 'est.tum'}")


if __name__ == "__main__":
    main()
