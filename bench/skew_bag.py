"""Generate unsynchronized-camera variants of a benchmark bag.

The simulator records all cameras perfectly synchronized. Real cheap rigs are
NOT synchronized — so instead of hacking asynchrony into the sim, we apply
deterministic per-topic timestamp offsets (and optional gaussian jitter) to a
recorded ROS1 bag. One clean recording → any number of skew profiles, and every
candidate sees the *identical* image data with only the timing changed.

Both the bag receive-time and the message header stamp are shifted.

Usage:
    python -m bench.skew_bag in.bag out.bag \
        --offset /uav1/cam1/color/image_raw=0.015 \
        --offset /uav1/cam2/color/image_raw=-0.008 \
        --jitter /uav1/cam1/color/image_raw=0.002 \
        --seed 0

Requires the `rosbags` package (pure python, no ROS installation).
"""

from __future__ import annotations

import argparse
import heapq
import sys

import numpy as np

from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore


def parse_topic_value(pairs: list[str], what: str) -> dict[str, float]:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"bad --{what} '{p}', expected TOPIC=SECONDS")
        topic, val = p.rsplit("=", 1)
        out[topic] = float(val)
    return out


def shift_header(msg, shift_ns: int) -> None:
    header = getattr(msg, "header", None)
    if header is None:
        return
    total = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec + shift_ns
    header.stamp.sec = int(total // 1_000_000_000)
    header.stamp.nanosec = int(total % 1_000_000_000)


def skew_bag(src: str, dst: str, offsets: dict[str, float],
             jitters: dict[str, float], seed: int = 0) -> dict[str, int]:
    """Rewrite `src` bag to `dst` with per-topic time shifts. Returns per-topic counts."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}

    # Writer requires chronological order; shifted messages can leapfrog
    # unshifted ones, so drain through a min-heap with a safety margin.
    max_shift_s = max(
        [abs(v) for v in offsets.values()] + [5 * v for v in jitters.values()] + [0.0]
    )
    margin_ns = int((2 * max_shift_s + 0.1) * 1e9)

    with Reader(src) as reader, Writer(dst) as writer:
        wconn = {}
        for conn in reader.connections:
            wconn[conn.id] = writer.add_connection(
                conn.topic, conn.msgtype, typestore=typestore,
                callerid=conn.ext.callerid, latching=conn.ext.latching,
            )

        heap: list = []
        seq = 0

        def flush(upto_ns: int | None) -> None:
            while heap and (upto_ns is None or heap[0][0] <= upto_ns):
                ts, _, cid, raw = heapq.heappop(heap)
                writer.write(wconn[cid], ts, raw)

        for conn, timestamp, rawdata in reader.messages():
            shift_ns = 0
            if conn.topic in offsets or conn.topic in jitters:
                shift_s = offsets.get(conn.topic, 0.0)
                if conn.topic in jitters:
                    shift_s += float(rng.normal(0.0, jitters[conn.topic]))
                shift_ns = int(round(shift_s * 1e9))
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                shift_header(msg, shift_ns)
                rawdata = typestore.serialize_ros1(msg, conn.msgtype)
            counts[conn.topic] = counts.get(conn.topic, 0) + 1
            seq += 1
            heapq.heappush(heap, (timestamp + shift_ns, seq, conn.id, rawdata))
            flush(timestamp - margin_ns)
        flush(None)

    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--offset", action="append", default=[], metavar="TOPIC=SECONDS")
    ap.add_argument("--jitter", action="append", default=[], metavar="TOPIC=STDDEV_S")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    offsets = parse_topic_value(args.offset, "offset")
    jitters = parse_topic_value(args.jitter, "jitter")
    if not offsets and not jitters:
        raise SystemExit("nothing to do: no --offset/--jitter given")

    counts = skew_bag(args.src, args.dst, offsets, jitters, args.seed)
    for topic, n in sorted(counts.items()):
        marker = " <- skewed" if topic in offsets or topic in jitters else ""
        print(f"{n:8d}  {topic}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
