from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_WEIGHTS_DIR = _ROOT / "weights"
_DETECTOR_PATH = _WEIGHTS_DIR / "barcode_stage1_detector_v8s.pt"

# Prompt and repository names differ; prefer exact prompt path, fallback to existing file.
_VALIDATOR_CANDIDATES = [
    _WEIGHTS_DIR / "yolo2_finetune.pt",
    _WEIGHTS_DIR / "yolo2v8n.pt",
]

_detector: YOLO | None = None
_validator: YOLO | None = None
_model_lock = threading.Lock()


def _resolve_validator_path() -> Path:
    for path in _VALIDATOR_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Validator weights not found. Tried: {[str(p) for p in _VALIDATOR_CANDIDATES]}"
    )


def get_models() -> Tuple[YOLO, YOLO]:
    global _detector, _validator
    if _detector is not None and _validator is not None:
        return _detector, _validator

    with _model_lock:
        if _detector is None:
            _detector = YOLO(str(_DETECTOR_PATH))
        if _validator is None:
            _validator = YOLO(str(_resolve_validator_path()))

    return _detector, _validator


def warmup_models() -> None:
    detector, validator = get_models()
    detector(
        np.zeros((1024, 1024, 3), dtype=np.uint8),
        conf=0.03,
        iou=0.85,
        imgsz=1024,
        max_det=50,
        device="cpu",
        verbose=False,
    )
    validator(
        np.zeros((256, 256, 3), dtype=np.uint8),
        conf=0.25,
        device="cpu",
        verbose=False,
    )


def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_shape: Tuple[int, int] | Tuple[int, int, int],
    scale: float = 1.35,
) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    nw = bw * scale
    nh = bh * scale

    nx1 = int(max(0, round(cx - nw / 2.0)))
    ny1 = int(max(0, round(cy - nh / 2.0)))
    nx2 = int(min(w, round(cx + nw / 2.0)))
    ny2 = int(min(h, round(cy + nh / 2.0)))

    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)

    return nx1, ny1, nx2, ny2


def _resize_long_side(image: np.ndarray, target_long_side: int = 1024) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        return image, 1.0

    scale = target_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_barcode_crops(
    image: np.ndarray,
    max_validate_candidates: int = 8,
) -> List[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    if image is None or image.size == 0:
        return []

    detector, validator = get_models()
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized, scale = _resize_long_side(color, target_long_side=1024)

    det_result = detector(
        resized,
        conf=0.03,
        iou=0.85,
        imgsz=1024,
        max_det=50,
        device="cpu",
        verbose=False,
    )[0]

    raw_boxes = det_result.boxes.xyxy if det_result.boxes is not None else None
    if raw_boxes is None or raw_boxes.shape[0] == 0:
        logger.info("[YOLO] proposals=0 validated=0")
        return []

    boxes = raw_boxes.detach().cpu().numpy()
    confs = (
        det_result.boxes.conf.detach().cpu().numpy()
        if det_result.boxes.conf is not None
        else np.ones((boxes.shape[0],), dtype=np.float32)
    )
    order = np.argsort(-confs)
    validated_crops: List[Tuple[np.ndarray, Tuple[float, float, float, float]]] = []

    inv_scale = 1.0 / scale
    candidate_crops: List[np.ndarray] = []
    candidate_boxes: List[Tuple[float, float, float, float]] = []
    limit = min(max_validate_candidates, len(order))
    for idx in order[:limit]:
        box = boxes[idx]
        x1, y1, x2, y2 = [float(v) * inv_scale for v in box[:4]]
        ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, color.shape, scale=1.35)
        crop = color[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue
        candidate_crops.append(crop)
        h, w = color.shape[:2]
        candidate_boxes.append(
            (
                float(ex1 / max(1, w)),
                float(ey1 / max(1, h)),
                float(ex2 / max(1, w)),
                float(ey2 / max(1, h)),
            )
        )

    if not candidate_crops:
        logger.info("[YOLO] proposals=%d validated=0", len(boxes))
        return []

    val_results = validator(
        candidate_crops,
        conf=0.25,
        device="cpu",
        verbose=False,
    )
    for crop, box_norm, val_result in zip(candidate_crops, candidate_boxes, val_results):
        val_boxes = val_result.boxes.xyxy if val_result.boxes is not None else None
        if val_boxes is None or val_boxes.shape[0] == 0:
            continue
        validated_crops.append((crop, box_norm))

    logger.info("[YOLO] proposals=%d validated=%d", len(boxes), len(validated_crops))
    return validated_crops
