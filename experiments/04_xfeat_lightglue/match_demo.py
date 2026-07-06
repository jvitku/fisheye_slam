#!/usr/bin/env python3
"""XFeat matching demo on a fisheye image pair.

First exploratory script for Track D1: run XFeat (accelerated_features) on two
fisheye frames, draw matches, and print basic stats. Compares against ORB as the
classical reference. Weights are fetched via torch.hub on first run.

Usage:
    python match_demo.py <img0> <img1> [out.png]
"""

import sys
import time

import cv2
import numpy as np
import torch


def run_xfeat(img0, img1, top_k=2048):
    xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k)
    t0 = time.perf_counter()
    mkpts0, mkpts1 = xfeat.match_xfeat(img0, img1, top_k=top_k)
    dt = time.perf_counter() - t0
    return np.asarray(mkpts0), np.asarray(mkpts1), dt


def run_orb(img0, img1, n=2048):
    orb = cv2.ORB_create(nfeatures=n)
    t0 = time.perf_counter()
    k0, d0 = orb.detectAndCompute(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), None)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    if d0 is None or d1 is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), time.perf_counter() - t0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d0, d1)
    dt = time.perf_counter() - t0
    p0 = np.array([k0[m.queryIdx].pt for m in matches])
    p1 = np.array([k1[m.trainIdx].pt for m in matches])
    return p0, p1, dt


def inlier_ratio(p0, p1):
    """Fraction of matches consistent with a fundamental matrix (RANSAC)."""
    if len(p0) < 8:
        return 0.0, np.zeros(len(p0), dtype=bool)
    _, mask = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 1.5, 0.999)
    mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(p0), dtype=bool)
    return mask.mean(), mask


def draw(img0, img1, p0, p1, mask, path):
    h = max(img0.shape[0], img1.shape[0])
    canvas = np.zeros((h, img0.shape[1] + img1.shape[1], 3), dtype=np.uint8)
    canvas[: img0.shape[0], : img0.shape[1]] = img0
    canvas[: img1.shape[0], img0.shape[1] :] = img1
    off = np.array([img0.shape[1], 0])
    for a, b, ok in zip(p0.astype(int), (p1 + off).astype(int), mask):
        color = (0, 200, 0) if ok else (0, 0, 200)
        cv2.line(canvas, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    img0, img1 = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])
    assert img0 is not None and img1 is not None, "failed to read input images"
    out = sys.argv[3] if len(sys.argv) > 3 else "matches.png"

    for name, fn in [("xfeat", run_xfeat), ("orb", run_orb)]:
        p0, p1, dt = fn(img0, img1)
        ratio, mask = inlier_ratio(p0, p1)
        print(f"{name:6s}: {len(p0):5d} matches, {ratio:5.1%} RANSAC inliers, {dt * 1e3:7.1f} ms")
        if name == "xfeat":
            draw(img0, img1, p0, p1, mask, out)
            print(f"        match visualization -> {out}")


if __name__ == "__main__":
    main()
