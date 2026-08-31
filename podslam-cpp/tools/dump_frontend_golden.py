#!/usr/bin/env python3
"""Dump the KLT front-end's exact inputs and outputs for the C++ port's parity tests.

Records, per processed frame, exactly what the tracker hands to
``frontend.process`` and what comes back:

  frames.bin   binary: for each frame
                 int64 t_ns, int32 n_cams, int32 h, int32 w,
                 then n_cams raw uint8 images (h*w each),
                 float64[9] dR_imu row-major (identity if None)
  tracks.txt   text: per frame, per camera one line
                 T <t_ns> <cam> <n>  followed by n lines  "<id> <u> <v>"
               plus "G <t_ns> <n_new>" after each frame's cameras.

The C++ front-end (port step 3) must reproduce ids and pixel positions exactly
(same OpenCV calls, same gates).  Frames are conditioned (post-preprocess), so
the conditioning layer is excluded from this parity and tested separately.

    UV_NO_SYNC=1 PYTHONPATH=. uv run python podslam-cpp/tools/dump_frontend_golden.py \
        datasets/data/tumvi/room1_day.bag rigs/tumvi_room1.yaml \
        /tmp/frontend_golden --max-frames 200
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag"); ap.add_argument("rig"); ap.add_argument("outdir")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--frontend", default="klt", choices=("klt", "multiklt"),
                    help="multiklt: patch MultiKltFrontend.process (one record per real frame, merged per-cam tracks)")
    ap.add_argument("--extra", default="", help="extra track_bag args, comma-separated (e.g. --cams,cam0:cam1,--kf-every,6)")
    a = ap.parse_args(argv)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    from podslam import track_bag
    from podslam.frontend.klt import KltFrontend
    from podslam.frontend.multiklt import MultiKltFrontend
    from podslam.rig import load_rig

    # complete rig-lines file for the C++ side: R (intrinsics + T_imu_cam),
    # N (imu noise, accel scale, estimator noise scale, px_sigma), F (fov per cam)
    rig_obj = load_rig(a.rig)
    with open(out / "rig.txt", "w") as fr:
        for c in rig_obj.cameras:
            m = c.model
            vals = [m.fx, m.fy, m.cx, m.cy, m.k1, m.k2, m.k3, m.k4]
            vals += [c.T_imu_cam[r][col] for r in range(3) for col in range(4)]
            fr.write("R " + " ".join(f"{v:.17g}" for v in vals) + "\n")
        i = rig_obj.imu
        px_sigma = float(getattr(rig_obj, "px_sigma", 1.5) or 1.5)
        fr.write("N " + " ".join(f"{v:.17g}" for v in
                 [i.gyro_noise_density, i.gyro_random_walk, i.accel_noise_density, i.accel_random_walk,
                  float(getattr(i, "accel_scale", 1.0)), 5.7, 1.8, 1.2, 4.5, px_sigma]) + "\n")
        fr.write("F " + " ".join(f"{float(c.fov_deg or 0):.17g}" for c in rig_obj.cameras) + "\n")

    fb = open(out / "frames.bin", "wb")
    ft = open(out / "tracks.txt", "w")
    fi = open(out / "imu.txt", "w")

    from podslam.tracker import Tracker
    orig_reg = Tracker.register_imu

    def register(self, t_ns, gyro, accel):
        fi.write(f"{t_ns} " + " ".join(f"{float(x):.12g}" for x in [*gyro, *accel]) + "\n")
        return orig_reg(self, t_ns, gyro, accel)

    Tracker.register_imu = register

    orig = KltFrontend.process

    def process(self, t_ns, images, masks, dR_imu):
        h, w = images[0].shape
        fb.write(struct.pack("<qiii", int(t_ns), len(images), h, w))
        for im in images:
            fb.write(np.ascontiguousarray(im, dtype=np.uint8).tobytes())
        dR = np.eye(3) if dR_imu is None else np.asarray(dR_imu, float)
        fb.write(dR.astype("<f8").tobytes())
        ff = orig(self, t_ns, images, masks, dR_imu)
        for c, cam in enumerate(ff.cams):
            ft.write(f"T {int(t_ns)} {c} {len(cam)}\n")
            for i in range(len(cam)):
                ft.write(f"{int(cam.ids[i])} {cam.px[i][0]:.6f} {cam.px[i][1]:.6f}\n")
        ft.write(f"G {int(t_ns)} {ff.n_new}\n")
        return ff

    if a.frontend == "multiklt":
        orig_multi = MultiKltFrontend.process

        def process_multi(self, t_ns, images, masks, dR_imu):
            h, w = images[0].shape
            fb.write(struct.pack("<qiii", int(t_ns), len(images), h, w))
            for im in images:
                fb.write(np.ascontiguousarray(im, dtype=np.uint8).tobytes())
            dR = np.eye(3) if dR_imu is None else np.asarray(dR_imu, float)
            fb.write(dR.astype("<f8").tobytes())
            ff = orig_multi(self, t_ns, images, masks, dR_imu)
            for c, cam in enumerate(ff.cams):
                ft.write(f"T {int(t_ns)} {c} {len(cam)}\n")
                for i in range(len(cam)):
                    ft.write(f"{int(cam.ids[i])} {cam.px[i][0]:.6f} {cam.px[i][1]:.6f}\n")
            ft.write(f"G {int(t_ns)} {ff.n_new}\n")
            return ff

        MultiKltFrontend.process = process_multi
    else:
        KltFrontend.process = process
    extra = [x.replace(":", ",") for x in a.extra.split(",") if x] if a.extra else []
    track_bag.main([a.bag, str(out / "run"), "--rig", a.rig, "--max-frames", str(a.max_frames),
                    "--frontend", a.frontend] + extra)
    fb.close(); ft.close(); fi.close()
    import shutil
    shutil.copy(Path(a.outdir) / "run" / "est.tum", out / "python_est.tum")
    print(f"wrote {out}/frames.bin ({(out / 'frames.bin').stat().st_size / 1e6:.0f} MB) and tracks.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
