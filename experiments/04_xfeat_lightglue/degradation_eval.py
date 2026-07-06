"""Hybrid front-end study: XFeat vs ORB matching on day vs night bags.

Samples frame pairs (temporal gap = motion baseline) at the SAME timestamps
from the day bag and its night-degraded variant, runs both matchers on both
conditions, and reports match counts and RANSAC inlier statistics. This
quantifies what a learned front-end buys under the pod's night+IR conditions
BEFORE committing to VIO integration.

Usage (inside the xfeat venv, from repo root):
    python experiments/04_xfeat_lightglue/degradation_eval.py \
        day.bag night.bag --topic /cam0/image_raw --pairs 40 --gap 10 \
        --xfeat-repo candidates/accelerated_features --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS1_NOETIC)


def load_frames(bag: str, topic: str) -> list[tuple[int, np.ndarray]]:
    frames = []
    with Reader(bag) as r:
        for conn, ts_, raw in r.messages():
            if conn.topic == topic:
                m = TS.deserialize_ros1(raw, conn.msgtype)
                frames.append((ts_, np.frombuffer(m.data, np.uint8).reshape(m.height, m.width).copy()))
    return frames


def sample_pairs(n_frames: int, n_pairs: int, gap: int) -> list[tuple[int, int]]:
    idx = np.linspace(0, n_frames - gap - 1, n_pairs).astype(int)
    return [(int(i), int(i + gap)) for i in idx]


def ransac_inliers(p0: np.ndarray, p1: np.ndarray) -> tuple[int, float]:
    if len(p0) < 8:
        return 0, 0.0
    _, mask = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 1.5, 0.999)
    if mask is None:
        return 0, 0.0
    inl = int(mask.sum())
    return inl, inl / len(p0)


class OrbMatcher:
    name = "orb"

    def __init__(self, n=2048):
        self.orb = cv2.ORB_create(nfeatures=n)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __call__(self, a, b):
        k0, d0 = self.orb.detectAndCompute(a, None)
        k1, d1 = self.orb.detectAndCompute(b, None)
        if d0 is None or d1 is None or len(d0) < 8 or len(d1) < 8:
            return np.zeros((0, 2)), np.zeros((0, 2))
        m = self.bf.match(d0, d1)
        return (np.array([k0[x.queryIdx].pt for x in m]),
                np.array([k1[x.trainIdx].pt for x in m]))


class XFeatMatcher:
    name = "xfeat"

    def __init__(self, repo: str, top_k=2048):
        import torch
        self.xfeat = torch.hub.load(repo, "XFeat", source="local",
                                    pretrained=True, top_k=top_k)
        self.top_k = top_k

    def __call__(self, a, b):
        # XFeat wants HxWxC uint8 or tensors; grayscale -> 3-channel
        a3 = np.stack([a] * 3, axis=-1)
        b3 = np.stack([b] * 3, axis=-1)
        p0, p1 = self.xfeat.match_xfeat(a3, b3, top_k=self.top_k)
        return np.asarray(p0), np.asarray(p1)


def evaluate(frames, pairs, matcher) -> dict:
    matches, inliers, ratios = [], [], []
    for i, j in pairs:
        p0, p1 = matcher(frames[i][1], frames[j][1])
        inl, ratio = ransac_inliers(p0, p1)
        matches.append(len(p0))
        inliers.append(inl)
        ratios.append(ratio)
    return {
        "matches_mean": float(np.mean(matches)),
        "inliers_mean": float(np.mean(inliers)),
        "inliers_median": float(np.median(inliers)),
        "inlier_ratio_mean": float(np.mean(ratios)),
        "pairs_below_30_inliers": int(np.sum(np.array(inliers) < 30)),
        "n_pairs": len(pairs),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("day_bag")
    ap.add_argument("night_bag")
    ap.add_argument("--topic", default="/cam0/image_raw")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--xfeat-repo", default="candidates/accelerated_features")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    results = {}
    matchers = [OrbMatcher(), XFeatMatcher(args.xfeat_repo)]
    for cond, bag in (("day", args.day_bag), ("night", args.night_bag)):
        frames = load_frames(bag, args.topic)
        pairs = sample_pairs(len(frames), args.pairs, args.gap)
        for m in matchers:
            key = f"{m.name}/{cond}"
            results[key] = evaluate(frames, pairs, m)
            r = results[key]
            print(f"{key:12s}: {r['matches_mean']:7.1f} matches, "
                  f"{r['inliers_mean']:7.1f} inliers ({r['inlier_ratio_mean']:5.1%}), "
                  f"{r['pairs_below_30_inliers']}/{r['n_pairs']} pairs <30 inl")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
