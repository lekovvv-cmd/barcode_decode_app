from __future__ import annotations

from typing import List, Sequence

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


def align_frames_ecc(
    frames: Sequence[np.ndarray],
    reference: np.ndarray | None = None,
    motion_model: int = cv2.MOTION_AFFINE,
    number_of_iterations: int = 80,
    termination_eps: float = 1e-5,
) -> List[np.ndarray]:
    """
    Align frames to a reference frame using ECC.

    If alignment fails for a frame, the original frame is kept.
    """
    if not frames:
        return []

    ref = reference if reference is not None else frames[0]
    if ref.ndim == 3:
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref

    ref_h, ref_w = ref_gray.shape[:2]
    aligned: List[np.ndarray] = [ref_gray]

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    for idx, frame in enumerate(frames):
        if idx == 0 and reference is None:
            continue

        if frame.ndim == 3:
            moving_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            moving_gray = frame

        if moving_gray.shape[:2] != (ref_h, ref_w):
            moving_gray = cv2.resize(
                moving_gray, (ref_w, ref_h), interpolation=cv2.INTER_AREA
            )

        if motion_model == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            cv2.findTransformECC(
                ref_gray.astype(np.float32) / 255.0,
                moving_gray.astype(np.float32) / 255.0,
                warp_matrix,
                motion_model,
                criteria,
                None,
                1,
            )
            if motion_model == cv2.MOTION_HOMOGRAPHY:
                warped = cv2.warpPerspective(
                    moving_gray,
                    warp_matrix,
                    (ref_w, ref_h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                warped = cv2.warpAffine(
                    moving_gray,
                    warp_matrix,
                    (ref_w, ref_h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            aligned.append(warped)
        except cv2.error:
            aligned.append(moving_gray)

    return aligned
