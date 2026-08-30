"""Front-end contract.

A front-end consumes conditioned images (+ masks, + the IMU rotation since the
previous frame) and returns, per camera, the tracked features as *bearings*
(unit vectors in that camera's optical frame) with persistent track ids.
Ids are shared across cameras for stereo correspondences (the cam1 entry with
the same id as a cam0 entry is its stereo match). Everything downstream is
lens-model agnostic — a learned front-end only has to speak this dataclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CamObs:
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    bearings: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))

    def __len__(self) -> int:
        return int(len(self.ids))


@dataclass
class FrameFeatures:
    t_ns: int
    cams: list                      # CamObs per camera index
    n_new: int = 0                  # freshly detected tracks this frame (diagnostics)


class Frontend(ABC):
    def __init__(self, rig, config=None):
        self.rig = rig
        self.config = config or {}

    def set_depths(self, depths: dict) -> None:
        """Estimator feedback: track id -> depth in the tracking camera (m); used as the
        stereo initial guess instead of the front-end's own (self-reinforcing) estimate."""
        return None

    @abstractmethod
    def process(self, t_ns: int, images: list, masks: list | None,
                dR_imu: np.ndarray | None) -> FrameFeatures:
        """images: uint8 grayscale per camera; masks: 255 = valid (or None);
        dR_imu: R_{I_prev <- I_cur} from the gyro (None on the first frame)."""

    def reset(self) -> None:
        pass
