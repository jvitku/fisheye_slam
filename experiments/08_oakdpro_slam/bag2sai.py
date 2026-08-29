"""Convert a bag-contract ROS1 bag into a Spectacular AI recording folder.

Output layout (SAI "recording" format, consumed by spectacularAI.Replay):
    out/
      data.jsonl          IMU samples + frame-group index
      data.mkv            cam0 (left) lossless FFV1 video
      data2.mkv           cam1 (right)
      calibration.json    pinhole intrinsics + imuToCamera per camera
      vio_config.yaml     engine knobs (stereo, no feature dots)

Format reference: https://spectacularai.github.io/docs/sdk/recording.html
VERIFY-ON-FIRST-RUN: field names below were written from the docs + recorded
samples of sdk-examples; validate by replaying one converted bag and fix here.

Usage (inside 3dfe/spectacularai, see run_spectacularai.sh):
    python3 bag2sai.py input.bag out_dir --rig /rigs/oakdpro.yaml
    python3 bag2sai.py input.bag out_dir --calib calib_<serial>.yaml   # HW bag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS1_NOETIC)

CAM_TOPICS = ["/uav1/cam0/color/image_raw", "/uav1/cam1/color/image_raw"]
IMU_TOPIC = "/uav1/sensor_pod/imu"

# FLU body/pod frame -> optical frame (z forward, x right, y down).
R_FLU_TO_OPT = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def _mount_T(mount) -> np.ndarray:
    """rig mount {position, rpy_deg} -> 4x4 pod_T_frame (FLU)."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", mount["rpy_deg"], degrees=True).as_matrix()
    T[:3, 3] = mount["position"]
    return T


def calibration_from_rig(rig_path: str) -> dict:
    """rigs/oakdpro.yaml -> SAI calibration.json dict (imuToCamera, pinhole)."""
    rig = yaml.safe_load(open(rig_path))
    T_pod_imu = _mount_T(rig["imu"].get("mount", {"position": [0, 0, 0], "rpy_deg": [0, 0, 0]}))
    cams = []
    for cam in rig["cameras"]:
        T_pod_cam_flu = _mount_T(cam["mount"])
        # optical camera frame: rotate FLU axes to optical convention
        T_pod_cam = T_pod_cam_flu.copy()
        T_pod_cam[:3, :3] = T_pod_cam_flu[:3, :3] @ R_FLU_TO_OPT.T
        T_cam_imu = np.linalg.inv(T_pod_cam) @ T_pod_imu
        intr = cam["intrinsics"]
        cams.append({
            "model": "pinhole",
            "imageWidth": cam["resolution"][0],
            "imageHeight": cam["resolution"][1],
            "focalLengthX": float(intr["fx"]),
            "focalLengthY": float(intr["fy"]),
            "principalPointX": float(intr["cx"]),
            "principalPointY": float(intr["cy"]),
            "imuToCamera": T_cam_imu.tolist(),
        })
    return {"cameras": cams}


def calibration_from_yaml(calib_path: str) -> dict:
    """calib_<serial>.yaml written by hw/record_oak.py -> SAI calibration."""
    c = yaml.safe_load(open(calib_path))
    return {"cameras": [
        {
            "model": "pinhole",
            "imageWidth": cc["width"], "imageHeight": cc["height"],
            "focalLengthX": cc["fx"], "focalLengthY": cc["fy"],
            "principalPointX": cc["cx"], "principalPointY": cc["cy"],
            "imuToCamera": cc["imu_to_camera"],
        }
        for cc in c["cameras"]
    ]}


def convert(bag: Path, out: Path, calibration: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    writers = {}   # cam index -> cv2.VideoWriter
    jsonl = (out / "data.jsonl").open("w")
    pending = {}   # stamp_ns -> {cam_ind: written}
    frame_number = 0

    def video_writer(ind, img):
        name = "data.mkv" if ind == 0 else f"data{ind + 1}.mkv"
        return cv2.VideoWriter(
            str(out / name), cv2.VideoWriter_fourcc(*"FFV1"),
            20.0, (img.shape[1], img.shape[0]), isColor=False,
        )

    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in CAM_TOPICS + [IMU_TOPIC]]
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = TS.deserialize_ros1(raw, conn.msgtype)
            t = t_ns / 1e9
            if conn.topic == IMU_TOPIC:
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                g, a = msg.angular_velocity, msg.linear_acceleration
                jsonl.write(json.dumps({"time": stamp, "sensor": {
                    "type": "gyroscope", "values": [g.x, g.y, g.z]}}) + "\n")
                jsonl.write(json.dumps({"time": stamp, "sensor": {
                    "type": "accelerometer", "values": [a.x, a.y, a.z]}}) + "\n")
                continue

            ind = CAM_TOPICS.index(conn.topic)
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            if ind not in writers:
                writers[ind] = video_writer(ind, img)
            writers[ind].write(img)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            key = round(stamp, 4)  # sim cams are perfectly synced
            pending.setdefault(key, set()).add(ind)
            if pending[key] == {0, 1}:
                jsonl.write(json.dumps({
                    "time": stamp, "number": frame_number,
                    "frames": [{"cameraInd": 0, "time": stamp},
                               {"cameraInd": 1, "time": stamp}],
                }) + "\n")
                frame_number += 1
                del pending[key]

    for w in writers.values():
        w.release()
    jsonl.close()
    json.dump(calibration, (out / "calibration.json").open("w"), indent=2)
    yaml.safe_dump(
        {"useStereo": True, "useSlam": True},
        (out / "vio_config.yaml").open("w"),
    )
    print(f"wrote {out} ({frame_number} stereo frames)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rig", help="rig yaml (sim bags: exact GT calibration)")
    src.add_argument("--calib", help="calib_<serial>.yaml from hw/record_oak.py")
    args = ap.parse_args()
    calib = calibration_from_rig(args.rig) if args.rig else calibration_from_yaml(args.calib)
    convert(args.bag, args.out, calib)


if __name__ == "__main__":
    main()
