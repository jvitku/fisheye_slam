"""Front-ends: turn images into tracked bearing observations. Pick by name."""
from .base import CamObs, FrameFeatures, Frontend  # noqa: F401


def build_frontend(name: str, rig, config=None):
    if name == "klt":
        from .klt import KltFrontend
        return KltFrontend(rig, config)
    if name == "multiklt":
        from .multiklt import MultiKltFrontend
        return MultiKltFrontend(rig, config)
    if name == "xfeat":
        from .xfeat import XFeatFrontend
        return XFeatFrontend(rig, config)
    raise ValueError(f"unknown frontend {name!r} (klt | multiklt | xfeat)")
