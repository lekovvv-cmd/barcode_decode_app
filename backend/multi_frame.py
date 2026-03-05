from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def median_merge(frames: Sequence[np.ndarray]) -> np.ndarray:
  """Median merge a list of equal-sized grayscale or BGR frames."""
  if not frames:
      raise ValueError("No frames to merge")

  stack = np.stack(frames, axis=0).astype(np.float32)
  med = np.median(stack, axis=0)
  med = np.clip(med, 0, 255).astype(np.uint8)
  return med

