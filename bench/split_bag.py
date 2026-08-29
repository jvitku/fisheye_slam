"""Split a composite-rig recording into one bag per member pod.

A composite rig (rigs/pod3_oakdpro.yaml) records both devices of one flight
into one bag with namespaced topics (/uav1/oakd_cam0/..., /uav1/oakd/imu).
This tool writes, per member, a bag that follows that member's OWN single-rig
contract again (/uav1/cam0/color/image_raw, /uav1/sensor_pod/imu, ...) so
every existing runner consumes it unchanged, plus the simulator ground truth
expressed at that member's IMU frame (bench/gt_extract.py) — the frame a VIO
estimate on that device lives in.

    combo.bag  --rig rigs/pod3_oakdpro.yaml  ->  combo_pod.bag  + combo_pod.gt.txt
                                                 combo_oakd.bag + combo_oakd.gt.txt

Messages are copied byte-for-byte (only the topic name changes); the shared
ground truth goes into every output.

Usage:
    python -m bench.split_bag combo.bag --rig rigs/pod3_oakdpro.yaml
        [--out-dir DIR] [--no-gt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

from bench.gt_extract import DEFAULT_TOPIC as GT_TOPIC
from bench.gt_extract import body_to_sensor_T, extract
from bench.rigdef import camera_topic, load_rig, vehicle_prefix


def topic_map(rig: dict, ns: str) -> dict[str, str]:
    """Composite topic -> member's single-rig topic, for pod `ns`."""
    if not rig.get("composite"):
        raise ValueError("split_bag needs a composite rig (one with a `pods:` list)")
    prefix = vehicle_prefix(rig)
    mapping = {}
    for cam in rig["cameras"]:
        if cam["pod_ns"] != ns:
            continue
        streams = ["color", "depth"] if cam["depth"] else ["color"]
        for stream in streams:
            mapping[camera_topic(rig, cam, stream)] = \
                f"{prefix}/{cam['source_name']}/{stream}/image_raw"
    for imu in rig["imus"]:
        if imu["pod_ns"] == ns:
            mapping[imu["topic"]] = imu["source_topic"]
    mapping[GT_TOPIC] = GT_TOPIC
    return mapping


def split_bag(src: str, rig_path: str, out_dir: str | None = None,
              with_gt: bool = True) -> dict[str, dict]:
    """Write one bag (+ gt.txt) per member pod. Returns {ns: {bag, gt, counts}}."""
    rig = load_rig(rig_path)
    src_path = Path(src)
    out_base = Path(out_dir) if out_dir else src_path.parent
    out_base.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    result = {}

    for pod in rig["pods"]:
        ns = pod["ns"]
        mapping = topic_map(rig, ns)
        dst = out_base / f"{src_path.stem}_{ns}.bag"
        counts: dict[str, int] = {}
        with Reader(src) as reader, Writer(dst) as writer:
            conns = [c for c in reader.connections if c.topic in mapping]
            wconn = {
                c.id: writer.add_connection(
                    mapping[c.topic], c.msgtype, typestore=typestore,
                    callerid=c.ext.callerid, latching=c.ext.latching,
                )
                for c in conns
            }
            for conn, timestamp, raw in reader.messages(connections=conns):
                writer.write(wconn[conn.id], timestamp, raw)
                counts[mapping[conn.topic]] = counts.get(mapping[conn.topic], 0) + 1
        entry = {"bag": str(dst), "counts": counts, "rig": pod["rig"], "name": pod["name"]}
        if with_gt and GT_TOPIC in counts:
            gt = out_base / f"{src_path.stem}_{ns}.gt.txt"
            extract(src, str(gt), GT_TOPIC, body_to_sensor_T(rig, ns))
            entry["gt"] = str(gt)
        result[ns] = entry
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--rig", required=True, help="composite rig yaml the bag was recorded with")
    ap.add_argument("--out-dir", default=None, help="default: next to the input bag")
    ap.add_argument("--no-gt", action="store_true", help="skip per-pod ground-truth extraction")
    args = ap.parse_args(argv)

    result = split_bag(args.bag, args.rig, args.out_dir, with_gt=not args.no_gt)
    for ns, entry in result.items():
        print(f"[{ns}] {entry['bag']}" + (f"  gt: {entry['gt']}" if "gt" in entry else ""))
        for topic, n in sorted(entry["counts"].items()):
            print(f"  {n:8d}  {topic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
