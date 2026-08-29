"""Fuse dumped depth frames + tracked poses into a TSDF mesh via nvblox.

Bridges cuvslam_track.py --dump-depth output (depth/*.npy [m] + poses.txt)
into the 3DMatch layout consumed by nvblox's `fuse_3dmatch` example binary
(depth as 16-bit PNG in mm, per-frame camera-to-world 4x4 .txt,
camera-intrinsics.txt), then invokes it. VERIFY-ON-FIRST-RUN: fuse_3dmatch
argument list against the built /opt/nvblox binary.

Poses in poses.txt are cam0-odometry (optical frame, world = first-frame);
depth is aligned to cam0 by contract, so no extra extrinsic is needed.

Usage (inside 3dfe/nvblox): python3 nvblox_fuse.py <depth_dir> <out_mesh.ply> --rig rig.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

FUSE_BIN = "/opt/nvblox/nvblox/build/executables/fuse_3dmatch"


def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("depth_dir", type=Path)
    ap.add_argument("out_mesh", type=Path)
    ap.add_argument("--rig", required=True)
    args = ap.parse_args()

    rig = yaml.safe_load(open(args.rig))
    intr = rig["cameras"][0]["intrinsics"]

    with tempfile.TemporaryDirectory() as td:
        seq = Path(td) / "seq-01"
        seq.mkdir(parents=True)
        K = ("{fx} 0 {cx}\n0 {fy} {cy}\n0 0 1\n").format(**intr)
        (Path(td) / "camera-intrinsics.txt").write_text(K)

        for line in (args.depth_dir / "poses.txt").read_text().splitlines():
            f = line.split()
            idx, (x, y, z, qx, qy, qz, qw) = f[0], map(float, f[2:9])
            depth_m = np.load(args.depth_dir / f"{idx}.npy")
            depth_mm = np.nan_to_num(depth_m * 1000.0, posinf=0).astype(np.uint16)
            cv2.imwrite(str(seq / f"frame-{idx}.depth.png"), depth_mm)
            # fuse_3dmatch also expects a color frame; feed depth as gray.
            cv2.imwrite(str(seq / f"frame-{idx}.color.png"),
                        (np.clip(depth_m / 10.0, 0, 1) * 255).astype(np.uint8))
            T = np.eye(4)
            T[:3, :3] = quat_to_R(qx, qy, qz, qw)
            T[:3, 3] = [x, y, z]
            np.savetxt(seq / f"frame-{idx}.pose.txt", T, fmt="%.9f")

        subprocess.run(
            [FUSE_BIN, td, "--mesh_output_path", str(args.out_mesh)],
            check=True,
        )
    print(f"mesh: {args.out_mesh}")


if __name__ == "__main__":
    main()
