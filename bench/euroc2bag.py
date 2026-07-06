"""Convert a EuRoC-format dataset directory (TUM-VI export) to a ROS1 bag.

The TUM mirror stopped serving .bag exports (2026-07), so we build the bag
ourselves from the euroc tar: mav0/cam*/data/*.png -> sensor_msgs/Image
(mono8), mav0/imu0/data.csv -> sensor_msgs/Imu. Pure python (rosbags + PIL),
no ROS needed.

Usage:
    python -m bench.euroc2bag <euroc_root_with_mav0> out.bag \
        [--cam-topic-fmt /cam{i}/image_raw] [--imu-topic /imu0]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS1_NOETIC)
NS = 1_000_000_000


def make_header(stamp_ns: int, frame_id: str):
    Header = TS.types["std_msgs/msg/Header"]
    Time = TS.types["builtin_interfaces/msg/Time"]
    kwargs = {"stamp": Time(sec=int(stamp_ns // NS), nanosec=int(stamp_ns % NS)),
              "frame_id": frame_id}
    if "seq" in Header.__dataclass_fields__:
        kwargs["seq"] = 0
    return Header(**kwargs)


def image_msg(path: Path, stamp_ns: int, frame_id: str):
    raw = np.asarray(PILImage.open(path))
    if raw.dtype == np.uint16:
        # TUM-VI *_16 exports are 16-bit PNGs; linear >>8 matches cv_bridge's
        # mono16 -> mono8 conversion that consumers of the original bags saw.
        # (PIL .convert("L") CLIPS >255 instead -> saturated white frames.)
        img = (raw >> 8).astype(np.uint8)
    else:
        img = np.asarray(PILImage.open(path).convert("L"))
    Image = TS.types["sensor_msgs/msg/Image"]
    return Image(
        header=make_header(stamp_ns, frame_id),
        height=img.shape[0], width=img.shape[1],
        encoding="mono8", is_bigendian=0, step=img.shape[1],
        data=np.ascontiguousarray(img).reshape(-1),
    )


def imu_msg(stamp_ns: int, wx, wy, wz, ax, ay, az):
    Imu = TS.types["sensor_msgs/msg/Imu"]
    Quat = TS.types["geometry_msgs/msg/Quaternion"]
    Vec3 = TS.types["geometry_msgs/msg/Vector3"]
    return Imu(
        header=make_header(stamp_ns, "imu0"),
        orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=np.full(9, -1.0),
        angular_velocity=Vec3(x=wx, y=wy, z=wz),
        angular_velocity_covariance=np.zeros(9),
        linear_acceleration=Vec3(x=ax, y=ay, z=az),
        linear_acceleration_covariance=np.zeros(9),
    )


def convert(root: Path, out: str, cam_topic_fmt: str, imu_topic: str) -> dict:
    mav0 = root / "mav0"
    if not mav0.is_dir():
        raise SystemExit(f"{root}: no mav0/ directory")

    # collect (stamp_ns, kind, payload) events, then write chronologically
    events: list[tuple[int, str, object]] = []

    cams = sorted(d for d in mav0.glob("cam*") if d.is_dir())
    for i, cam in enumerate(cams):
        for f in sorted((cam / "data").glob("*.png")):
            events.append((int(f.stem), f"cam{i}", f))

    with open(mav0 / "imu0" / "data.csv") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            events.append((int(row[0]), "imu", [float(v) for v in row[1:7]]))

    events.sort(key=lambda e: e[0])
    counts: dict[str, int] = {}

    with Writer(out) as w:
        conns = {"imu": w.add_connection(imu_topic, "sensor_msgs/msg/Imu", typestore=TS)}
        for i in range(len(cams)):
            conns[f"cam{i}"] = w.add_connection(
                cam_topic_fmt.format(i=i), "sensor_msgs/msg/Image", typestore=TS)

        for stamp, kind, payload in events:
            if kind == "imu":
                msg = imu_msg(stamp, *payload)
                raw = TS.serialize_ros1(msg, "sensor_msgs/msg/Imu")
            else:
                msg = image_msg(payload, stamp, kind)
                raw = TS.serialize_ros1(msg, "sensor_msgs/msg/Image")
            w.write(conns[kind], stamp, raw)
            counts[kind] = counts.get(kind, 0) + 1

    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="directory containing mav0/")
    ap.add_argument("out", help="output .bag path")
    ap.add_argument("--cam-topic-fmt", default="/cam{i}/image_raw")
    ap.add_argument("--imu-topic", default="/imu0")
    args = ap.parse_args(argv)
    counts = convert(Path(args.root), args.out, args.cam_topic_fmt, args.imu_topic)
    for k, n in sorted(counts.items()):
        print(f"{n:8d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
