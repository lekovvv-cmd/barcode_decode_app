from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def normalize_frame(img: np.ndarray, target_height: int = 320) -> np.ndarray:
    """Resize keeping aspect ratio and convert to grayscale."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    scale = target_height / float(h)
    new_w = int(w * scale)
    resized = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized


def estimate_orientation(gray: np.ndarray) -> float:
    """Use Hough lines to estimate dominant orientation (degrees)."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 80)
    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180.0 / np.pi) - 90.0
        angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    return median_angle


def rotate_image(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def deskew(gray: np.ndarray) -> np.ndarray:
    """Small additional deskew using image moments."""
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 1.0:
        return gray
    return rotate_image(gray, angle)


def generate_variants(gray: np.ndarray) -> List[np.ndarray]:
    """Generate multiple preprocessed variants for ensemble decoding."""
    variants: List[np.ndarray] = []

    variants.append(gray)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(clahe_img)

    # Adaptive threshold
    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10
    )
    variants.append(adap)

    # Otsu
    _, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(otsu)

    # Inverted
    variants.append(cv2.bitwise_not(gray))

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, kernel)
    variants.append(sharp)

    # Bilateral
    bilat = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    variants.append(bilat)

    return variants


def upscale(gray: np.ndarray, scale: float = 2.5) -> np.ndarray:
    h, w = gray.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)


def anti_aliasing_prep(gray: np.ndarray) -> np.ndarray:
    """Gaussian blur then sharpen to fight aliasing."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(blurred, -1, kernel)

