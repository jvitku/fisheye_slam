#!/usr/bin/env python3
"""Run a TorchScript image enhancer (experiments/09_cuvslam_robustness/train_enhancer.py)
over every camera image of a rosbag and write a new bag, so the tracker can be
evaluated on enhanced frames at normal speed (the U-Net is 3.7 ms/frame on a GPU,
200 ms/frame on one CPU core).

    docker run --gpus all ... 3dfe/cuvslam-ml python3 bench/enhance_bag.py \
        datasets/data/tumvi/room1_night.bag datasets/data/tumvi/room1_nightenh.bag models/enhancer_night.pt
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS1_NOETIC)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("model")
    ap.add_argument("--cam-topic", action="append", default=None)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(a.model, map_location=dev).eval().to(dev)
    n = 0; t0 = time.time()
    with Reader(a.src) as reader, Writer(a.dst) as writer:
        topics = a.cam_topic or [c.topic for c in reader.connections if c.msgtype == "sensor_msgs/msg/Image"]
        wconn = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=TS, callerid=c.ext.callerid, latching=c.ext.latching)
                 for c in reader.connections}
        for conn, timestamp, raw in reader.messages():
            if conn.topic in topics:
                msg = TS.deserialize_ros1(raw, conn.msgtype)
                img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
                with torch.no_grad():
                    x = torch.from_numpy(img.copy()).float().to(dev)[None, None] / 255.0
                    out = (model(x).clamp(0, 1)[0, 0] * 255.0).round().byte().cpu().numpy()
                msg.data = np.ascontiguousarray(out).reshape(-1)
                raw = TS.serialize_ros1(msg, conn.msgtype)
                n += 1
            writer.write(wconn[conn.id], timestamp, raw)
    print(f"enhanced {n} images from {topics} in {time.time() - t0:.0f} s on {dev} -> {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
