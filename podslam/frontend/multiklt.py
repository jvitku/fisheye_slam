"""Per-camera KLT front-end for divergent multi-camera rigs (Skydio-style).

The classic KltFrontend is stereo-centric: it tracks only in cam0 and LK-matches
those features into the other cameras — which yields nothing when the cameras
barely overlap (200-deg fisheyes 120 deg apart).  This wrapper runs one
independent single-camera KLT tracker per camera (disjoint id spaces, shared
IMU rotation prediction); every camera contributes its own tracks and the smart
rig backend triangulates them from parallax over time (mono_landmarks).
"""
from __future__ import annotations

from .base import FrameFeatures, Frontend
from .klt import KltFrontend

ID_STRIDE = 100_000_000


class _MonoRig:
    """Single-camera view of the rig (KltFrontend only touches rig.cameras[0]
    outside its stereo path, which a 1-camera rig never enters)."""

    def __init__(self, cam):
        self.cameras = [cam]


class MultiKltFrontend(Frontend):
    per_cam = True

    def __init__(self, rig, config=None):
        super().__init__(rig, config)
        self.subs = []
        for i, cam in enumerate(rig.cameras):
            sub = KltFrontend(_MonoRig(cam), dict(config or {}))
            sub.next_id = i * ID_STRIDE
            self.subs.append(sub)

    def reset(self):
        for i, sub in enumerate(self.subs):
            sub.reset()
            sub.next_id = i * ID_STRIDE

    def process(self, t_ns, images, masks, dR_imu):
        cams, n_new = [], 0
        for i, sub in enumerate(self.subs):
            ff = sub.process(t_ns, [images[i]], [masks[i]] if masks is not None else None, dR_imu)
            cams.append(ff.cams[0])
            n_new += ff.n_new
        return FrameFeatures(t_ns=t_ns, cams=cams, n_new=n_new)

    def set_depths(self, depths):
        for tid, d in depths.items():
            self.subs[int(tid) // ID_STRIDE].set_depths({int(tid): d})
