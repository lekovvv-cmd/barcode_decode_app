from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class FrameQuality:
    sharpness: float
    contrast: float
    brightness: float

    @property
    def quality(self) -> float:
        # Same formula as frontend for consistency.
        return self.sharpness * 0.7 + self.contrast * 0.3


def compute_sharpness(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    return float(var)


def compute_contrast(gray: np.ndarray) -> float:
    return float(gray.var())


def compute_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def compute_frame_quality(gray: np.ndarray) -> FrameQuality:
    sharp = compute_sharpness(gray)
    contrast = compute_contrast(gray)
    brightness = compute_brightness(gray)
    return FrameQuality(sharpness=sharp, contrast=contrast, brightness=brightness)


def score_and_sort_frames(
    frames: Tuple[np.ndarray, ...]
) -> Tuple[Tuple[np.ndarray, FrameQuality], ...]:
    """Return frames paired with quality, sorted best-first."""
    scored = []
    for img in frames:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        q = compute_frame_quality(gray)
        scored.append((img, q))
    scored.sort(key=lambda p: p[1].quality, reverse=True)
    return tuple(scored)

