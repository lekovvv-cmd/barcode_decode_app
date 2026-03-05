from __future__ import annotations

import cv2
import numpy as np


def estimate_code_type(gray: np.ndarray) -> str:
    """
    Rough heuristic to distinguish 1D vs 2D:
    - Stronger horizontal gradients -> likely 1D (vertical bars).
    - Similar gradients -> treat as 2D.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    mag_x = np.mean(np.abs(gx))
    mag_y = np.mean(np.abs(gy))

    if mag_x > mag_y * 1.3:
        return "1D"
    return "2D"

