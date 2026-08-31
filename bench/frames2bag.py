"""Assemble a benchmark bag from offline-rendered frames + the flight bag.

Second half of "physics once, render offline" (bench/README.md). Inputs:

    frames/<cam>/frames.csv          index,t_ns         (render_from_poses.py)
    frames/<cam>/<index>.png         color frame (gray or RGB)
    frames/<cam>/<index>.depth.npy   float32 depth [m]  (cameras with depth: true)
    flight.bag                       IMU stream(s) + /uav1/ground_truth from the
                                     flight pass (bench/record.sh, RECORD_IMAGES=0)
    rig yaml                         camera/IMU topic names (composite-aware:
                                     /uav1/pod_cam0/..., /uav1/oakd/imu)

Output: one bag following the same contract bench/record.sh would have
produced live — images on /uav1/<cam>/color/image_raw (mono8 | rgb8), depth
on /uav1/<cam>/depth/image_raw (32FC1 [m]), IMU + ground truth copied
byte-for-byte — in time order, streamed (never more than one frame per camera
in memory). A composite rig's bag then goes through bench/split_bag.py as usual.

Usage:
    python -m bench.frames2bag frames/ flight.bag rigs/pod3_oakdpro.yaml out.bag
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

from bench.gt_extract import DEFAULT_TOPIC as GT_TOPIC
from bench.rigdef import camera_topic, load_rig

TS = get_typestore(Stores.ROS1_NOETIC)
IMAGE_TYPE = "sensor_msgs/msg/Image"
NS = 1_000_000_000


def _header(t_ns: int, frame_id: str):
    Header = TS.types["std_msgs/msg/Header"]
    Time = TS.types["builtin_interfaces/msg/Time"]
    kw = {"stamp": Time(sec=t_ns // NS, nanosec=t_ns % NS), "frame_id": frame_id}
    if "seq" in Header.__dataclass_fields__:
        kw["seq"] = 0
    return Header(**kw)


def image_msg(t_ns: int, frame_id: str, arr: np.ndarray, encoding: str):
    """numpy (H,W) uint8 | (H,W,3) uint8 | (H,W) float32 -> serialized sensor_msgs/Image."""
    h, w = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    data = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
    msg = TS.types[IMAGE_TYPE](
        header=_header(t_ns, frame_id), height=h, width=w, encoding=encoding,
        is_bigendian=0, step=w * channels * arr.dtype.itemsize, data=data)
    return TS.serialize_ros1(msg, IMAGE_TYPE)


def read_frames_csv(path: Path) -> list[tuple[int, int]]:
    with open(path) as f:
        rows = [(int(r["index"]), int(r["t_ns"])) for r in csv.DictReader(f)]
    rows.sort(key=lambda r: r[1])
    return rows


def camera_stream(frames_dir: Path, cam: dict, color_topic: str, depth_topic: str | None):
    """Yield (t_ns, topic, raw) for one camera, in time order, one frame at a time."""
    cam_dir = frames_dir / cam["name"]
    for index, t_ns in read_frames_csv(cam_dir / "frames.csv"):
        img = np.asarray(Image.open(cam_dir / f"{index:06d}.png"))
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        encoding = "mono8" if img.ndim == 2 else "rgb8"
        yield t_ns, color_topic, image_msg(t_ns, cam["name"], img.astype(np.uint8), encoding)
        if depth_topic is not None and (cam_dir / f"{index:06d}.depth.npy").exists():
            depth = np.load(cam_dir / f"{index:06d}.depth.npy").astype(np.float32)
            yield t_ns, depth_topic, image_msg(t_ns, cam["name"], depth, "32FC1")


def flight_stream(reader: Reader, topics: set[str]):
    conns = [c for c in reader.connections if c.topic in topics]
    for conn, t_ns, raw in reader.messages(connections=conns):
        yield t_ns, conn.topic, raw


def frames2bag(frames_dir: str, flight_bag: str, rig_path: str, out_bag: str) -> dict[str, int]:
    rig = load_rig(rig_path)
    frames = Path(frames_dir)
    counts: dict[str, int] = {}
    out = Path(out_bag)
    out.unlink(missing_ok=True)

    with Reader(flight_bag) as reader, Writer(out) as writer:
        flight_topics = {imu["topic"] for imu in rig["imus"]} | {GT_TOPIC}
        present = {c.topic for c in reader.connections}
        missing = flight_topics - present
        if missing:
            raise ValueError(f"{flight_bag}: missing flight topics {sorted(missing)}")
        wconn = {}
        for c in reader.connections:
            if c.topic in flight_topics:
                wconn[c.topic] = writer.add_connection(
                    c.topic, c.msgtype, typestore=TS, callerid=c.ext.callerid, latching=c.ext.latching)

        streams = [flight_stream(reader, flight_topics)]
        for cam in rig["cameras"]:
            color = camera_topic(rig, cam)
            depth = camera_topic(rig, cam, "depth") if cam["depth"] else None
            wconn[color] = writer.add_connection(color, IMAGE_TYPE, typestore=TS)
            if depth:
                wconn[depth] = writer.add_connection(depth, IMAGE_TYPE, typestore=TS)
            streams.append(camera_stream(frames, cam, color, depth))

        # k-way merge on (t_ns, stream order) — every stream is time-sorted
        merged = heapq.merge(*(((t, i, topic, raw) for t, topic, raw in s)
                               for i, s in enumerate(streams)))
        for t_ns, _, topic, raw in merged:
            writer.write(wconn[topic], t_ns, raw)
            counts[topic] = counts.get(topic, 0) + 1
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir")
    ap.add_argument("flight_bag")
    ap.add_argument("rig")
    ap.add_argument("out_bag")
    args = ap.parse_args(argv)
    counts = frames2bag(args.frames_dir, args.flight_bag, args.rig, args.out_bag)
    for topic, n in sorted(counts.items()):
        print(f"{n:8d}  {topic}")
    print(f"-> {args.out_bag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
