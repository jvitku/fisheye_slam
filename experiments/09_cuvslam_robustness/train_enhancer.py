"""Learned night->day image conditioning for the closed cuVSLAM tracker.

A small U-Net (grayscale in/out, ~0.2 M params, real-time on a GPU) trained to
invert the pod's night degradation (bench/degrade_bag.py: reflectance x
(ambient + IR beam vignette) + high-gain noise) so the tracker sees day-like
frames. Training pairs are generated ON THE FLY from a DIFFERENT TUM-VI
sequence than the one we evaluate on (room2 -> train, room1 -> test), with
randomized degradation parameters (ambient, beam width, noise) so the model
does not just memorize one relighting.

Outputs a TorchScript model consumed by cuvslam_track.py --preprocess enhance:<pt>.

Usage (inside 3dfe/cuvslam-ml, --gpus all):
    python3 train_enhancer.py --frames /data/room2/mav0/cam0/data --out /models/enhancer.pt
        [--epochs 8] [--steps 300] [--batch 8] [--crop 256]
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/scripts_bench")          # bench/degrade_bag.py mounted here
from degrade_bag import beam_vignette, relight  # noqa: E402


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """3-level U-Net, residual output (predicts the correction, not the image)."""

    def __init__(self, ch=(16, 32, 64)):
        super().__init__()
        self.e1, self.e2, self.e3 = Block(1, ch[0]), Block(ch[0], ch[1]), Block(ch[1], ch[2])
        self.d2, self.d1 = Block(ch[2] + ch[1], ch[1]), Block(ch[1] + ch[0], ch[0])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        e1 = self.e1(xp)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        y = xp + self.out(d1)
        return y[..., :h, :w]


def load_frames(pattern: str, limit: int) -> list[np.ndarray]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no frames match {pattern}")
    step = max(1, len(files) // limit)
    out = []
    for f in files[::step][:limit]:
        img = np.asarray(Image.open(f))
        if img.dtype == np.uint16:                    # TUM-VI 16-bit PNGs -> mono8 (as euroc2bag)
            img = (img.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
        if img.ndim == 3:
            img = img[..., 0]
        out.append(img)
    return out


def make_pair(day: np.ndarray, rng: np.random.Generator, vig_cache: dict, crop: int):
    """Random degradation of a random crop; returns (night, day) float32 in [0,1]."""
    h, w = day.shape
    y, x = rng.integers(0, h - crop + 1), rng.integers(0, w - crop + 1)
    ambient = float(rng.uniform(0.02, 0.08)); beam = float(rng.uniform(0.4, 0.9))
    sigma = float(rng.choice([0.35, 0.45, 0.6])); noise = float(rng.uniform(3.0, 9.0))
    key = (h, w, sigma)
    if key not in vig_cache:
        vig_cache[key] = beam_vignette(h, w, sigma)
    night = relight(day, vig_cache[key], ambient, beam, noise, rng)
    return (night[y:y + crop, x:x + crop].astype(np.float32) / 255.0,
            day[y:y + crop, x:x + crop].astype(np.float32) / 255.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", required=True, help="glob dir of day PNGs (train sequence)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--frames-limit", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_frames(str(Path(args.frames) / "*.png"), args.frames_limit)
    print(f"{len(frames)} training frames, device {dev}")
    model = TinyUNet().to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.steps)
    vig_cache: dict = {}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for it in range(args.steps):
            pairs = [make_pair(frames[rng.integers(len(frames))], rng, vig_cache, args.crop) for _ in range(args.batch)]
            x = torch.from_numpy(np.stack([p[0] for p in pairs]))[:, None].to(dev)
            y = torch.from_numpy(np.stack([p[1] for p in pairs]))[:, None].to(dev)
            pred = model(x)
            # L1 + gradient (edge) loss: the tracker cares about gradients, not absolute tone
            gx = lambda t: t[..., :, 1:] - t[..., :, :-1]
            gy = lambda t: t[..., 1:, :] - t[..., :-1, :]
            loss = F.l1_loss(pred, y) + 2.0 * (F.l1_loss(gx(pred), gx(y)) + F.l1_loss(gy(pred), gy(y)))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        print(f"epoch {ep + 1}/{args.epochs}  loss {tot / args.steps:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 1, 512, 512, device=dev)
        scripted = torch.jit.trace(model, x)
        # timing at the pod's resolution
        torch.cuda.synchronize() if dev == "cuda" else None
        t1 = time.time()
        for _ in range(50):
            scripted(x)
        torch.cuda.synchronize() if dev == "cuda" else None
        print(f"inference 512x512: {(time.time() - t1) / 50 * 1e3:.2f} ms/frame on {dev}")
    scripted.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
