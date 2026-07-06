"""Photometric night / day->night-transition degradation for benchmark bags.

Proxy for the night+IR condition while the Isaac lighting axis awaits GPU
bring-up: model a camera-colocated IR flood LED in darkness by treating the
day image as a reflectance proxy and re-lighting it:

    L_night(x) = L_day(x) * (ambient + beam * vignette(r(x))) + sensor noise

- vignette: gaussian beam profile centered on the principal point (the flood
  LED is rigidly colocated with the cameras -> the pattern is static in the
  image, matching the real pod). Peripheral fisheye FOV falls outside the
  beam and goes near-black.
- ambient: residual scene light (moonlight class), default 4%.
- noise: high-gain sensor noise (gaussian read noise + poisson-ish shot
  noise), applied after re-lighting.

What this proxy CANNOT reproduce: moving cast shadows from the drone-mounted
light (geometry-dependent) — that needs the Isaac lighting axis. Results here
are therefore a LOWER bound on night difficulty.

Modes:
    night       whole sequence re-lit
    transition  day -> night linear ramp over the middle 20% of the sequence
                (first 40% day, last 40% night)

Deterministic: noise is seeded per-frame from the message timestamp.

Usage:
    python -m bench.degrade_bag in.bag out.bag --mode night \
        [--cam-topic /cam0/image_raw --cam-topic /cam1/image_raw] \
        [--ambient 0.04] [--beam 0.7] [--beam-sigma 0.45] [--noise 6]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS1_NOETIC)


def beam_vignette(h: int, w: int, sigma_frac: float) -> np.ndarray:
    """Gaussian IR-flood beam profile, peak 1.0 at image center."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2
    sigma = sigma_frac * np.hypot(h, w) / 2.0
    return np.exp(-r2 / (2.0 * sigma * sigma))


def relight(img: np.ndarray, vignette: np.ndarray, ambient: float, beam: float,
            noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float64)
    lit = f * (ambient + beam * vignette)
    # shot-ish noise grows with signal, plus constant read noise
    noise = rng.normal(0.0, 1.0, f.shape) * np.sqrt(
        noise_sigma**2 + 0.5 * np.maximum(lit, 0.0)
    )
    return np.clip(lit + noise, 0, 255).astype(np.uint8)


def night_factor(mode: str, progress: float) -> float:
    """0.0 = full day, 1.0 = full night, by sequence progress in [0, 1]."""
    if mode == "night":
        return 1.0
    if mode == "transition":
        return float(np.clip((progress - 0.4) / 0.2, 0.0, 1.0))
    raise ValueError(f"unknown mode {mode}")


def degrade_bag(src: str, dst: str, cam_topics: list[str] | None, mode: str,
                ambient: float = 0.04, beam: float = 0.7,
                beam_sigma: float = 0.45, noise_sigma: float = 6.0) -> dict:
    with Reader(src) as reader:
        t0, t1 = reader.start_time, reader.end_time
        span = max(t1 - t0, 1)
        if cam_topics is None:
            cam_topics = [c.topic for c in reader.connections
                          if c.msgtype == "sensor_msgs/msg/Image"]

        counts: dict[str, int] = {}
        vignette_cache: dict[tuple, np.ndarray] = {}

        with Writer(dst) as writer:
            wconn = {c.id: writer.add_connection(
                c.topic, c.msgtype, typestore=TS,
                callerid=c.ext.callerid, latching=c.ext.latching)
                for c in reader.connections}

            for conn, timestamp, raw in reader.messages():
                if conn.topic in cam_topics:
                    msg = TS.deserialize_ros1(raw, conn.msgtype)
                    if msg.encoding != "mono8":
                        raise SystemExit(f"only mono8 supported, got {msg.encoding}")
                    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)

                    key = (msg.height, msg.width)
                    if key not in vignette_cache:
                        vignette_cache[key] = beam_vignette(*key, beam_sigma)

                    k = night_factor(mode, (timestamp - t0) / span)
                    if k > 0.0:
                        rng = np.random.default_rng(timestamp & 0xFFFFFFFF)
                        night = relight(img, vignette_cache[key], ambient, beam,
                                        noise_sigma, rng)
                        out = night if k >= 1.0 else np.clip(
                            (1.0 - k) * img.astype(np.float64) + k * night, 0, 255
                        ).astype(np.uint8)
                        msg.data = np.ascontiguousarray(out).reshape(-1)
                        raw = TS.serialize_ros1(msg, conn.msgtype)
                    counts[conn.topic] = counts.get(conn.topic, 0) + 1
                writer.write(wconn[conn.id], timestamp, raw)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", choices=["night", "transition"], required=True)
    ap.add_argument("--cam-topic", action="append", default=None,
                    help="image topics to degrade (default: all Image topics)")
    ap.add_argument("--ambient", type=float, default=0.04)
    ap.add_argument("--beam", type=float, default=0.7)
    ap.add_argument("--beam-sigma", type=float, default=0.45)
    ap.add_argument("--noise", type=float, default=6.0)
    args = ap.parse_args(argv)
    counts = degrade_bag(args.src, args.dst, args.cam_topic, args.mode,
                         args.ambient, args.beam, args.beam_sigma, args.noise)
    for t, n in sorted(counts.items()):
        print(f"{n:8d}  {t} degraded ({args.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
