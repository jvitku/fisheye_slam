"""Record a real OAK-D Pro into the bag contract (docs/oak_d_pro_slam.md §4).

Produces the SAME ROS1 bag layout as the Isaac sim rig (rigs/oakdpro.yaml),
so every runner in experiments/08_oakdpro_slam works unchanged on HW data:

    /uav1/cam0/color/image_raw   rectified left  mono8
    /uav1/cam1/color/image_raw   rectified right mono8
    /uav1/cam0/depth/image_raw   on-device stereo depth, 32FC1 [m], cam0-aligned
    /uav1/sensor_pod/imu         IMU with DEVICE timestamps (not host arrival —
                                 see depthai-ros #461 for why that matters)

Also dumps the factory calibration to calib_<serial>.yaml next to the bag
(rectified-pinhole intrinsics + IMU extrinsics) — pass it to the runners via
--calib; never use the rig yaml's spec-sheet numbers on real data.

Pure python: depthai + rosbags, no ROS install needed (euroc2bag.py pattern).
VERIFY-ON-HW: written against depthai v2 docs; validate on a physical device.

Active illumination:
    --dot 0..1     IR laser dot projector (helps DEPTH, poisons VIO feature
                   tracking — keep 0 for odometry runs; per-frame dot/flood
                   interleaving is the v2 item, docs/oak_d_pro_slam.md §6)
    --flood 0..1   IR flood LED (night feature tracking — the pod concept)

Usage: python3 record_oak.py out.bag [--fps 20] [--imu-hz 200]
                                     [--dot 0] [--flood 0] [--duration 120]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import depthai as dai
import numpy as np
import yaml
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS1_NOETIC)
NS = 1_000_000_000

TOPICS = {
    "left": ("/uav1/cam0/color/image_raw", "sensor_msgs/msg/Image"),
    "right": ("/uav1/cam1/color/image_raw", "sensor_msgs/msg/Image"),
    "depth": ("/uav1/cam0/depth/image_raw", "sensor_msgs/msg/Image"),
    "imu": ("/uav1/sensor_pod/imu", "sensor_msgs/msg/Imu"),
}


def build_pipeline(fps: int, imu_hz: int) -> dai.Pipeline:
    p = dai.Pipeline()

    mono_l = p.create(dai.node.MonoCamera)
    mono_r = p.create(dai.node.MonoCamera)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    for m in (mono_l, mono_r):
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        m.setFps(fps)

    stereo = p.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)  # depth in cam0/left frame
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor(
        [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], imu_hz)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    for name, out in (("left", stereo.rectifiedLeft),
                      ("right", stereo.rectifiedRight),
                      ("depth", stereo.depth),
                      ("imu", imu.out)):
        x = p.create(dai.node.XLinkOut)
        x.setStreamName(name)
        out.link(x.input)
    return p


def dump_calibration(device: dai.Device, out_yaml: Path, width=1280, height=800) -> None:
    """Factory calibration -> calib_<serial>.yaml (rectified-pinhole contract).

    Rectified streams share the LEFT camera's intrinsics after rectification;
    imu_to_camera composes IMU extrinsics with the camera extrinsics.
    VERIFY-ON-HW: frame conventions of getImuToCameraExtrinsics vs SAI's
    imuToCamera (docs use camera<-imu, row-major 4x4, meters).
    """
    calib = device.readCalibration()
    cameras = []
    for socket in (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C):
        K = np.array(calib.getCameraIntrinsics(socket, width, height))
        try:
            T = np.array(calib.getImuToCameraExtrinsics(socket))
            T[:3, 3] /= 100.0  # depthai extrinsics are in centimeters
        except Exception:
            T = np.eye(4)  # older EEPROMs lack IMU extrinsics — flag it
            print(f"WARN: no IMU extrinsics for {socket}; wrote identity")
        cameras.append({
            "width": width, "height": height,
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "imu_to_camera": T.tolist(),
        })
    out_yaml.write_text(yaml.safe_dump({
        "device": device.getMxId(),
        "baseline_m": abs(calib.getBaselineDistance()) / 100.0,
        "cameras": cameras,
    }))
    print(f"calibration -> {out_yaml}")


def make_header(stamp_ns: int, frame_id: str):
    Header = TS.types["std_msgs/msg/Header"]
    Time = TS.types["builtin_interfaces/msg/Time"]
    kw = {"stamp": Time(sec=int(stamp_ns // NS), nanosec=int(stamp_ns % NS)),
          "frame_id": frame_id}
    if "seq" in Header.__dataclass_fields__:
        kw["seq"] = 0
    return Header(**kw)


def image_msg(arr: np.ndarray, stamp_ns: int, frame_id: str, encoding: str):
    Image = TS.types["sensor_msgs/msg/Image"]
    step = arr.shape[1] * arr.dtype.itemsize
    return Image(header=make_header(stamp_ns, frame_id),
                 height=arr.shape[0], width=arr.shape[1],
                 encoding=encoding, is_bigendian=0, step=step,
                 data=np.ascontiguousarray(arr).view(np.uint8).reshape(-1))


def imu_msg(stamp_ns: int, gyro, accel):
    Imu = TS.types["sensor_msgs/msg/Imu"]
    Quat = TS.types["geometry_msgs/msg/Quaternion"]
    Vec3 = TS.types["geometry_msgs/msg/Vector3"]
    return Imu(header=make_header(stamp_ns, "oak_imu"),
               orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
               orientation_covariance=np.full(9, -1.0),
               angular_velocity=Vec3(x=gyro.x, y=gyro.y, z=gyro.z),
               angular_velocity_covariance=np.zeros(9),
               linear_acceleration=Vec3(x=accel.x, y=accel.y, z=accel.z),
               linear_acceleration_covariance=np.zeros(9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--imu-hz", type=int, default=200)
    ap.add_argument("--dot", type=float, default=0.0, help="dot projector 0..1")
    ap.add_argument("--flood", type=float, default=0.0, help="flood LED 0..1")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds")
    args = ap.parse_args()

    with dai.Device(build_pipeline(args.fps, args.imu_hz)) as device:
        device.setIrLaserDotProjectorIntensity(args.dot)
        device.setIrFloodLightIntensity(args.flood)
        dump_calibration(
            device, args.out.with_name(f"calib_{device.getMxId()}.yaml"))

        queues = {n: device.getOutputQueue(n, maxSize=8, blocking=False)
                  for n in ("left", "right", "depth", "imu")}

        # Device timestamps are monotonic-since-boot; anchor once to wall time
        # so all messages share one consistent clock.
        anchor_wall = time.time()
        anchor_dev = None

        def to_ns(td) -> int:
            nonlocal anchor_dev
            dev_s = td.total_seconds()
            if anchor_dev is None:
                anchor_dev = dev_s
            return int((anchor_wall + dev_s - anchor_dev) * NS)

        with Writer(args.out) as writer:
            conns = {k: writer.add_connection(t, m, typestore=TS)
                     for k, (t, m) in TOPICS.items()}

            def put(key, msg, stamp_ns):
                _, msgtype = TOPICS[key]
                writer.write(conns[key], stamp_ns,
                             TS.serialize_ros1(msg, msgtype))

            t_end = time.time() + args.duration
            n = {"left": 0, "imu": 0}
            while time.time() < t_end:
                for side in ("left", "right"):
                    f = queues[side].tryGet()
                    if f is not None:
                        ts = to_ns(f.getTimestampDevice())
                        cam = "cam0" if side == "left" else "cam1"
                        put(side, image_msg(f.getFrame(), ts, cam, "mono8"), ts)
                        n["left"] += side == "left"
                f = queues["depth"].tryGet()
                if f is not None:
                    ts = to_ns(f.getTimestampDevice())
                    depth_m = f.getFrame().astype(np.float32) / 1000.0  # mm->m
                    put("depth", image_msg(depth_m, ts, "cam0", "32FC1"), ts)
                pkt = queues["imu"].tryGet()
                if pkt is not None:
                    for s in pkt.packets:
                        ts = to_ns(s.gyroscope.getTimestampDevice())
                        put("imu", imu_msg(ts, s.gyroscope, s.acceleroMeter), ts)
                        n["imu"] += 1
                time.sleep(0.001)

        print(f"{args.out}: {n['left']} frames, {n['imu']} imu samples, "
              f"dot={args.dot} flood={args.flood}")


if __name__ == "__main__":
    main()
