"""Replay a Spectacular AI recording folder -> TUM trajectory + keyframe map.

Runs the SAI engine offline on a folder produced by bag2sai.py (or recorded
natively on the device). Outputs:
    <out>/est.tum       time x y z qx qy qz qw   (bench/evaluate.py input)
    <out>/map.ply       merged keyframe point cloud (Mapping API), if enabled

API reference: https://spectacularai.github.io/docs/sdk/ (Replay + Mapping).
Written against the sdk-examples replay/mapping patterns — VERIFY-ON-FIRST-RUN.

Usage (inside 3dfe/spectacularai):
    python3 sai_replay.py <recording_dir> <out_dir> [--no-map]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import spectacularAI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-map", action="store_true", help="VIO only, skip Mapping API")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tum = (args.out / "est.tum").open("w")
    points: list[np.ndarray] = []

    def on_vio(out) -> None:
        # out.pose: .time [s], .position (x,y,z), .orientation (x,y,z,w)
        p, q = out.pose.position, out.pose.orientation
        tum.write(f"{out.pose.time:.6f} {p.x} {p.y} {p.z} {q.x} {q.y} {q.z} {q.w}\n")

    def on_mapping(mapper_out) -> None:
        # Collect the FINAL versions of keyframe clouds (world frame).
        if not mapper_out.finalMap:
            return
        for kf_id in mapper_out.updatedKeyFrames:
            kf = mapper_out.map.keyFrames.get(kf_id)
            if kf is None or kf.pointCloud is None or kf.pointCloud.empty():
                continue
            pts = kf.pointCloud.getPositionData()          # Nx3, camera frame
            T = kf.frameSet.primaryFrame.cameraPose.getCameraToWorldMatrix()
            pts_w = pts @ T[:3, :3].T + T[:3, 3]
            points.append(pts_w.astype(np.float32))

    if args.no_map:
        replay = spectacularAI.Replay(str(args.recording))
    else:
        replay = spectacularAI.Replay(str(args.recording), on_mapping)
    replay.setOutputCallback(on_vio)
    replay.runReplay()
    tum.close()

    if points:
        cloud = np.concatenate(points)
        _write_ply(args.out / "map.ply", cloud)
        print(f"map.ply: {len(cloud)} points")
    print(f"trajectory: {args.out / 'est.tum'}")


def _write_ply(path: Path, xyz: np.ndarray) -> None:
    with path.open("wb") as f:
        f.write(
            b"ply\nformat binary_little_endian 1.0\n"
            + f"element vertex {len(xyz)}\n".encode()
            + b"property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        xyz.astype("<f4").tofile(f)


if __name__ == "__main__":
    main()
