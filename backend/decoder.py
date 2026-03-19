from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from fastapi import UploadFile
from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol

try:
    from .frame_quality import compute_frame_quality
    from .multi_frame import align_frames_ecc, median_merge
    from .preprocess import (
        anti_aliasing_prep,
        deskew,
        estimate_orientation,
        generate_variants,
        normalize_frame,
        rotate_image,
        upscale,
    )
    from .type_detection import estimate_code_type
except ImportError:
    from frame_quality import compute_frame_quality
    from multi_frame import align_frames_ecc, median_merge
    from preprocess import (
        anti_aliasing_prep,
        deskew,
        estimate_orientation,
        generate_variants,
        normalize_frame,
        rotate_image,
        upscale,
    )
    from type_detection import estimate_code_type

try:
    from .yolo_detector import detect_barcode_crops as _detect_barcode_crops
except Exception:  # pragma: no cover - optional fallback stage dependency
    try:
        from yolo_detector import detect_barcode_crops as _detect_barcode_crops
    except Exception:  # pragma: no cover
        _detect_barcode_crops = None

logger = logging.getLogger(__name__)

try:
    import zxingcpp  # type: ignore

    ZXING_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    ZXING_AVAILABLE = False


SUPPORTED_1D = {
    "EAN13",
    "EAN8",
    "UPCA",
    "UPCE",
    "CODE128",
    "CODE39",
    "ITF",
    "CODABAR",
}

SUPPORTED_2D = {"QRCODE", "DATAMATRIX", "PDF417", "AZTEC"}
REQUEST_BUDGET_SEC = 2.8
REQUEST_BUDGET_MAX_SEC = 3.8
YOLO_DECODE_MAX_CROPS = 4
_QR_DETECTOR = cv2.QRCodeDetector()
_GS1_AI01_RE = re.compile(r"\(01\)\s*([0-9]{14})")
_GS1_AI21_RE = re.compile(r"\(21\)\s*([^\(\)\s]+)")


def decode_confidence(
    value: str,
    btype: str,
    frame_quality_norm: float,
    decoder_agreement: float,
) -> float:
    checksum_score = checksum_validity(value, btype)
    conf = (
        0.4 * checksum_score
        + 0.3 * np.clip(frame_quality_norm, 0.0, 1.0)
        + 0.3 * np.clip(decoder_agreement, 0.0, 1.0)
    )
    return float(np.clip(conf, 0.0, 1.0))


def quality_to_unit_interval(quality: float) -> float:
    # Smooth saturation so extreme values do not dominate confidence.
    return float(1.0 - np.exp(-quality / 180.0))


def checksum_validity(value: str, btype: str) -> float:
    t = (btype or "").upper()
    if t in {"EAN13", "EAN-13"}:
        return 1.0 if _is_valid_ean13(value) else 0.0
    if t in {"EAN8", "EAN-8"}:
        return 1.0 if _is_valid_ean8(value) else 0.0
    if t in {"UPCA", "UPC-A"}:
        return 1.0 if _is_valid_upca(value) else 0.0
    if t in {"UPCE", "UPC-E"}:
        # Keep partial credit if we cannot expand UPC-E robustly.
        return 0.7 if value.isdigit() and len(value) in {7, 8} else 0.0
    # Formats without standard public checksum rules in this module.
    return 0.6


def _is_valid_ean13(value: str) -> bool:
    if not value.isdigit() or len(value) != 13:
        return False
    digits = [int(c) for c in value]
    s = sum(digits[i] for i in range(0, 12, 2)) + 3 * sum(
        digits[i] for i in range(1, 12, 2)
    )
    check = (10 - (s % 10)) % 10
    return check == digits[12]


def _is_valid_ean8(value: str) -> bool:
    if not value.isdigit() or len(value) != 8:
        return False
    digits = [int(c) for c in value]
    s = 3 * sum(digits[i] for i in range(0, 7, 2)) + sum(
        digits[i] for i in range(1, 7, 2)
    )
    check = (10 - (s % 10)) % 10
    return check == digits[7]


def _is_valid_upca(value: str) -> bool:
    if not value.isdigit() or len(value) != 12:
        return False
    digits = [int(c) for c in value]
    s = 3 * sum(digits[i] for i in range(0, 11, 2)) + sum(
        digits[i] for i in range(1, 11, 2)
    )
    check = (10 - (s % 10)) % 10
    return check == digits[11]


def _expand_roi(gray: np.ndarray, expansion: float = 0.2) -> np.ndarray:
    """
    Expand detected content ROI by ~20% and clamp to bounds.
    Falls back to the original frame when content box is unavailable.
    """
    if gray.size == 0:
        return gray

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Prefer dark foreground for bars/modules.
    if np.mean(thr) > 127:
        thr = 255 - thr

    thr = cv2.morphologyEx(
        thr,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    pts = cv2.findNonZero(thr)
    if pts is None:
        return gray

    x, y, w, h = cv2.boundingRect(pts)
    if w <= 2 or h <= 2:
        return gray

    pad_x = int(w * expansion)
    pad_y = int(h * expansion)

    h_img, w_img = gray.shape[:2]
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w_img, x + w + pad_x)
    y1 = min(h_img, y + h + pad_y)

    if x1 <= x0 or y1 <= y0:
        return gray

    return gray[y0:y1, x0:x1]


def _frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))
    similarity = 1.0 - (diff / 255.0)
    return float(np.clip(similarity, 0.0, 1.0))


def _deduplicate_frames(
    frames: Sequence[np.ndarray],
    similarity_threshold: float = 0.95,
) -> List[np.ndarray]:
    unique: List[np.ndarray] = []
    for frame in frames:
        is_dup = False
        for kept in unique:
            if _frame_similarity(frame, kept) >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(frame)
    return unique


def _build_dedup_id(decoded: str, btype: str) -> str:
    text = (decoded or "").strip()
    if not text:
        return "raw:"

    ai01 = _GS1_AI01_RE.search(text)
    ai21 = _GS1_AI21_RE.search(text)
    if ai01 and ai21:
        return f"dm:{ai01.group(1)}:{ai21.group(1)}".lower()
    if ai01:
        return f"gtin:{ai01.group(1)}".lower()

    digits = "".join(ch for ch in text if ch.isdigit())
    t = (btype or "").upper()
    if t in {"EAN13", "EAN-13"} and len(digits) == 13:
        return f"gtin:0{digits}".lower()
    if t in {"UPCA", "UPC-A"} and len(digits) == 12:
        return f"gtin:0{digits}".lower()
    if t in {"EAN8", "EAN-8"} and len(digits) == 8:
        return f"ean8:{digits}".lower()
    if t in {"UPCE", "UPC-E"} and len(digits) in {7, 8}:
        return f"upce:{digits}".lower()

    return f"raw:{text.lower()}"


def _normalize_excluded_ids(exclude_ids: Optional[Sequence[str]]) -> set[str]:
    normalized: set[str] = set()
    for item in exclude_ids or []:
        value = str(item).strip().lower()
        if value:
            normalized.add(value)
    return normalized


def _should_skip_excluded(result: Dict, excluded_ids: set[str]) -> bool:
    decoded = str(result.get("decoded") or "").strip()
    btype = str(result.get("type") or "")
    dedup_id = _build_dedup_id(decoded, btype)
    result["dedup_id"] = dedup_id

    if not excluded_ids:
        return False

    keys = {
        dedup_id.lower(),
        f"raw:{decoded.lower()}",
        decoded.lower(),
    }
    if any(key in excluded_ids for key in keys):
        logger.info("decode_skip excluded dedup_id=%s type=%s", dedup_id, btype)
        return True
    return False


def _compute_decode_budget_sec(frame_count: int, yolo_validated: int) -> float:
    budget = REQUEST_BUDGET_SEC
    if frame_count <= 1:
        budget += 0.35
    if yolo_validated >= 2:
        budget += 0.45
    if yolo_validated >= 4:
        budget += 0.25
    return float(min(REQUEST_BUDGET_MAX_SEC, budget))


async def decode_frames(
    files: List[UploadFile],
    exclude_ids: Optional[Sequence[str]] = None,
) -> Dict:
    started_at = time.perf_counter()
    excluded_set = _normalize_excluded_ids(exclude_ids)

    # 1. Load frames to OpenCV, limit to 10.
    images: List[np.ndarray] = []
    for f in files[:10]:
        content = await f.read()
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        images.append(img)

    if not images:
        return {"decoded": None}

    # 2. Normalize and deduplicate near-identical frames while preserving original resolution.
    dedup_original: List[np.ndarray] = []
    dedup_normalized: List[np.ndarray] = []
    for image in images:
        normalized = normalize_frame(image)
        is_dup = False
        for kept in dedup_normalized:
            if _frame_similarity(normalized, kept) >= 0.95:
                is_dup = True
                break
        if is_dup:
            continue
        dedup_original.append(image)
        dedup_normalized.append(normalized)

    if not dedup_normalized:
        return {"decoded": None}

    # 3. Score quality and keep best up to 10.
    scored = sorted(
        [
            (idx, compute_frame_quality(gray))
            for idx, gray in enumerate(dedup_normalized)
        ],
        key=lambda item: item[1].quality,
        reverse=True,
    )
    best_indices = [idx for idx, _ in scored[:10]]
    best_frames = [dedup_normalized[idx] for idx in best_indices]
    best_original = dedup_original[best_indices[0]]

    # 4. YOLO-first path for fixed overhead camera setups with small/distant codes.
    # Early exit on first successful crop decode.
    yolo_crops: List[Tuple[np.ndarray, Tuple[float, float, float, float]]] = []
    if _detect_barcode_crops is not None:
        try:
            yolo_crops = _detect_barcode_crops(best_original)
        except Exception as exc:
            logger.warning("[YOLO] fallback_failed error=%s", exc)
            yolo_crops = []

    budget_sec = _compute_decode_budget_sec(
        frame_count=len(best_frames),
        yolo_validated=len(yolo_crops),
    )

    for crop, bbox_norm in yolo_crops[:YOLO_DECODE_MAX_CROPS]:
        if time.perf_counter() - started_at > budget_sec:
            logger.info("decode_budget_exhausted stage=yolo_crops")
            return {"decoded": None}

        result = _decode_yolo_crop(crop)
        if result:
            if _should_skip_excluded(result, excluded_set):
                continue
            result["strategy"] = "yolo_crop"
            result["frames_used"] = 1
            result["bbox"] = list(bbox_norm)
            logger.info("[YOLO-DECODE] success strategy=yolo_crop")
            return result

    if time.perf_counter() - started_at > budget_sec:
        logger.info("decode_budget_exhausted stage=post_yolo")
        return {"decoded": None}

    # 5. Orientation normalization from best-quality frame.
    best_gray = best_frames[0]
    orientation_angle = estimate_orientation(best_gray)
    if abs(orientation_angle) > 0.1:
        rotated = [rotate_image(im, -orientation_angle) for im in best_frames]
    else:
        rotated = best_frames

    # 6. Small ROI expansion to avoid cutting finder/guard patterns.
    expanded = [_expand_roi(im, expansion=0.2) for im in rotated]

    # Keep best frame first and get type hint.
    best_gray = expanded[0]
    best_q = compute_frame_quality(best_gray)
    best_q_norm = quality_to_unit_interval(best_q.quality)
    code_type_hint = estimate_code_type(best_gray)

    # 7. Additional deskew pass after orientation normalization.
    oriented: List[np.ndarray] = [deskew(img) for img in expanded]

    if time.perf_counter() - started_at > budget_sec:
        logger.info("decode_budget_exhausted stage=post_orientation")
        return {"decoded": None}

    # 2D special path: prefer single best frame strategies first.
    if code_type_hint == "2D":
        result = _decode_single_strategy(best_gray, frame_quality_norm=best_q_norm)
        if result:
            if not _should_skip_excluded(result, excluded_set):
                result["strategy"] = "best_frame"
                result["frames_used"] = 1
                logger.info(
                    "decode_success strategy=best_frame type=%s", result.get("type")
                )
                return result

        up_best = upscale(best_gray, scale=2.0)
        result = _decode_single_strategy(up_best, frame_quality_norm=best_q_norm)
        if result:
            if not _should_skip_excluded(result, excluded_set):
                result["strategy"] = "super_resolution"
                result["frames_used"] = 1
                logger.info(
                    "decode_success strategy=super_resolution type=%s",
                    result.get("type"),
                )
                return result

    if time.perf_counter() - started_at > budget_sec:
        logger.info("decode_budget_exhausted stage=post_2d_single")
        return {"decoded": None}

    # 1D or remaining 2D path: try single-frame ensemble.
    for gray in oriented:
        if time.perf_counter() - started_at > budget_sec:
            logger.info("decode_budget_exhausted stage=single_frame_loop")
            return {"decoded": None}

        q_norm = quality_to_unit_interval(compute_frame_quality(gray).quality)
        result = _decode_single_strategy(gray, frame_quality_norm=q_norm)
        if result:
            if _should_skip_excluded(result, excluded_set):
                continue
            result["strategy"] = "single_frame"
            result["frames_used"] = 1
            logger.info(
                "decode_success strategy=single_frame type=%s", result.get("type")
            )
            return result

    if time.perf_counter() - started_at > budget_sec:
        logger.info("decode_budget_exhausted stage=before_multiframe")
        return {"decoded": None}

    # Multi-frame alignment + median merge.
    try:
        aligned = align_frames_ecc(oriented, reference=best_gray)
        med = median_merge(aligned)
    except ValueError:
        med = best_gray

    q_med_norm = quality_to_unit_interval(compute_frame_quality(med).quality)
    result = _decode_single_strategy(med, frame_quality_norm=q_med_norm)
    if result:
        if not _should_skip_excluded(result, excluded_set):
            result["strategy"] = "median_multiframe"
            result["frames_used"] = len(oriented)
            logger.info(
                "decode_success strategy=median_multiframe type=%s", result.get("type")
            )
            return result

    if time.perf_counter() - started_at > budget_sec:
        logger.info("decode_budget_exhausted stage=before_super_resolution")
        return {"decoded": None}

    # Super resolution fallback.
    up = upscale(med, scale=2.5)
    q_up_norm = quality_to_unit_interval(compute_frame_quality(up).quality)
    result = _decode_single_strategy(up, frame_quality_norm=q_up_norm)
    if result:
        if not _should_skip_excluded(result, excluded_set):
            result["strategy"] = "super_resolution"
            result["frames_used"] = len(oriented)
            logger.info(
                "decode_success strategy=super_resolution type=%s", result.get("type")
            )
            return result

    if time.perf_counter() - started_at > budget_sec:
        logger.info("decode_budget_exhausted stage=before_aliasing")
        return {"decoded": None}

    # Anti-aliasing pass.
    aa = anti_aliasing_prep(up)
    q_aa_norm = quality_to_unit_interval(compute_frame_quality(aa).quality)
    result = _decode_single_strategy(aa, frame_quality_norm=q_aa_norm)
    if result:
        if not _should_skip_excluded(result, excluded_set):
            result["strategy"] = "aliasing_prep"
            result["frames_used"] = len(oriented)
            logger.info(
                "decode_success strategy=aliasing_prep type=%s", result.get("type")
            )
            return result

    if time.perf_counter() - started_at > budget_sec:
        logger.info("decode_budget_exhausted stage=before_signal")
        return {"decoded": None}

    # Final 1D signal extraction fallback.
    if code_type_hint == "1D":
        result = _signal_decode_1d(aa)
        if result:
            if not _should_skip_excluded(result, excluded_set):
                result["strategy"] = "signal_decode"
                result["frames_used"] = len(oriented)
                logger.info(
                    "decode_success strategy=signal_decode type=%s", result.get("type")
                )
                return result

    return {"decoded": None}


def _decode_single_strategy(
    gray: np.ndarray,
    frame_quality_norm: float,
) -> Optional[Dict]:
    """Run ensemble of decoders on one grayscale image and compute confidence."""
    variants = generate_variants(gray)
    for img in variants:
        candidates: List[Dict] = []

        res = _decode_with_pyzbar(img)
        if res:
            res["decoder"] = "pyzbar"
            candidates.append(res)

        if ZXING_AVAILABLE:
            res = _decode_with_zxing(img)
            if res:
                res["decoder"] = "zxing"
                candidates.append(res)

        res = _decode_with_opencv(img)
        if res:
            res["decoder"] = "opencv"
            candidates.append(res)

        if not candidates:
            continue

        selected = _select_best_candidate(
            candidates,
            frame_quality_norm=frame_quality_norm,
        )
        if selected:
            return selected

    return None


def _select_best_candidate(
    candidates: List[Dict],
    frame_quality_norm: float,
) -> Optional[Dict]:
    value_counter = Counter(c["decoded"] for c in candidates if c.get("decoded"))
    if not value_counter:
        return None

    best: Optional[Dict] = None
    best_score = -1.0

    for value, count in value_counter.items():
        same_value = [c for c in candidates if c.get("decoded") == value]
        if not same_value:
            continue

        type_counter = Counter(c.get("type", "UNKNOWN") for c in same_value)
        winner_type = type_counter.most_common(1)[0][0]
        agreement = float(count / max(1, len(candidates)))
        confidence = decode_confidence(
            value,
            winner_type,
            frame_quality_norm=frame_quality_norm,
            decoder_agreement=agreement,
        )

        # Tie-break: valid check-summed 1D codes (EAN/UPC) get a small boost
        # so they are not shadowed by "first candidate wins" behavior.
        checksum = checksum_validity(value, winner_type)
        score = confidence
        if winner_type in {"EAN13", "EAN8", "UPCA", "UPCE"} and checksum >= 0.99:
            score += 0.08

        if score > best_score:
            best_score = score
            best = {
                "decoded": value,
                "type": winner_type,
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "decoder_agreement": agreement,
            }

    return best


def _decode_yolo_crop(crop: np.ndarray) -> Optional[Dict]:
    """
    High-recall industrial decode path for YOLO crops:
    - upscale x2.5 (critical for tiny labels)
    - equalize / adaptive-threshold / inverted-threshold / sharpen variants
    - try 0/90/180/270 rotations
    """
    versions = _preprocess_versions_for_crop(crop)
    for version in versions:
        q_norm = quality_to_unit_interval(compute_frame_quality(version).quality)
        result = _decode_direct_ensemble(version, frame_quality_norm=q_norm)
        if result:
            return result

        for k in (1, 2, 3):
            rotated = np.rot90(version, k=k).copy()
            q_rot_norm = quality_to_unit_interval(compute_frame_quality(rotated).quality)
            result = _decode_direct_ensemble(rotated, frame_quality_norm=q_rot_norm)
            if result:
                return result
    return None


def _preprocess_versions_for_crop(crop: np.ndarray) -> List[np.ndarray]:
    versions: List[np.ndarray] = []
    big = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if big.ndim == 3 else big

    versions.append(gray)
    versions.append(cv2.equalizeHist(gray))

    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )
    versions.append(thr)
    versions.append(255 - thr)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    versions.append(cv2.filter2D(gray, -1, kernel))
    return versions


def _decode_direct_ensemble(gray: np.ndarray, frame_quality_norm: float) -> Optional[Dict]:
    candidates: List[Dict] = []

    if ZXING_AVAILABLE:
        zxing_res = _decode_with_zxing(gray)
        if zxing_res:
            zxing_res["decoder"] = "zxing"
            candidates.append(zxing_res)

    pyzbar_res = _decode_with_pyzbar(gray)
    if pyzbar_res:
        pyzbar_res["decoder"] = "pyzbar"
        candidates.append(pyzbar_res)

    opencv_res = _decode_with_opencv(gray)
    if opencv_res:
        opencv_res["decoder"] = "opencv"
        candidates.append(opencv_res)

    if not candidates:
        return None

    return _select_best_candidate(
        candidates,
        frame_quality_norm=frame_quality_norm,
    )


def _decode_with_pyzbar(gray: np.ndarray) -> Optional[Dict]:
    symbol_names = [
        "EAN13",
        "EAN8",
        "UPCA",
        "UPCE",
        "CODE128",
        "CODE39",
        "I25",
        "CODABAR",
        "QRCODE",
        "DATAMATRIX",
        "PDF417",
        "AZTEC",
    ]
    symbols = [getattr(ZBarSymbol, name) for name in symbol_names if hasattr(ZBarSymbol, name)]
    decoded = pyzbar_decode(gray, symbols=symbols) if symbols else pyzbar_decode(gray)
    if not decoded:
        return None

    d = decoded[0]
    value = d.data.decode("utf-8", errors="ignore")
    btype = d.type or "UNKNOWN"
    return {
        "decoded": value,
        "type": btype,
    }


def _decode_with_zxing(gray: np.ndarray) -> Optional[Dict]:
    if not ZXING_AVAILABLE:
        return None
    result = zxingcpp.read_barcode(gray)
    if not result or not result.text:
        return None
    btype = result.format.name if result.format else "UNKNOWN"
    return {
        "decoded": result.text,
        "type": btype,
    }


def _decode_with_opencv(gray: np.ndarray) -> Optional[Dict]:
    data, points, _ = _QR_DETECTOR.detectAndDecode(gray)
    if not data:
        return None
    return {
        "decoded": data,
        "type": "QRCODE",
    }


def _run_lengths(bits: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    if bits.size == 0:
        return runs
    current = int(bits[0])
    count = 1
    for b in bits[1:]:
        bi = int(b)
        if bi == current:
            count += 1
        else:
            runs.append((current, count))
            current = bi
            count = 1
    runs.append((current, count))
    return runs


def _resample_to_modules(runs: List[Tuple[int, int]], expected_modules: int) -> Optional[List[int]]:
    if not runs:
        return None

    lengths = np.array([length for _, length in runs], dtype=np.float32)
    small = np.sort(lengths)[: max(4, len(lengths) // 5)]
    module = float(np.median(small)) if small.size > 0 else float(np.median(lengths))
    module = max(module, 1.0)

    bits: List[int] = []
    for color, length in runs:
        modules = int(max(1, round(length / module)))
        bits.extend([color] * modules)

    if len(bits) < expected_modules:
        return None
    return bits


def _decode_ean13_bits(bits: List[int]) -> Optional[str]:
    if len(bits) < 95:
        return None

    A = {
        "0001101": "0",
        "0011001": "1",
        "0010011": "2",
        "0111101": "3",
        "0100011": "4",
        "0110001": "5",
        "0101111": "6",
        "0111011": "7",
        "0110111": "8",
        "0001011": "9",
    }
    B = {
        "0100111": "0",
        "0110011": "1",
        "0011011": "2",
        "0100001": "3",
        "0011101": "4",
        "0111001": "5",
        "0000101": "6",
        "0010001": "7",
        "0001001": "8",
        "0010111": "9",
    }
    C = {
        "1110010": "0",
        "1100110": "1",
        "1101100": "2",
        "1000010": "3",
        "1011100": "4",
        "1001110": "5",
        "1010000": "6",
        "1000100": "7",
        "1001000": "8",
        "1110100": "9",
    }
    parity_to_first = {
        "AAAAAA": "0",
        "AABABB": "1",
        "AABBAB": "2",
        "AABBBA": "3",
        "ABAABB": "4",
        "ABBAAB": "5",
        "ABBBAA": "6",
        "ABABAB": "7",
        "ABABBA": "8",
        "ABBABA": "9",
    }

    for start in range(0, len(bits) - 95 + 1):
        s = bits[start : start + 95]
        if s[0:3] != [1, 0, 1] or s[45:50] != [0, 1, 0, 1, 0] or s[92:95] != [1, 0, 1]:
            continue

        left_bits = s[3:45]
        right_bits = s[50:92]

        left_digits: List[str] = []
        parity: List[str] = []
        ok = True
        for i in range(6):
            pat = "".join(str(x) for x in left_bits[i * 7 : (i + 1) * 7])
            if pat in A:
                left_digits.append(A[pat])
                parity.append("A")
            elif pat in B:
                left_digits.append(B[pat])
                parity.append("B")
            else:
                ok = False
                break
        if not ok:
            continue

        right_digits: List[str] = []
        for i in range(6):
            pat = "".join(str(x) for x in right_bits[i * 7 : (i + 1) * 7])
            if pat not in C:
                ok = False
                break
            right_digits.append(C[pat])
        if not ok:
            continue

        first = parity_to_first.get("".join(parity))
        if first is None:
            continue

        code = first + "".join(left_digits) + "".join(right_digits)
        if _is_valid_ean13(code):
            return code

    return None


def _signal_decode_1d(gray: np.ndarray) -> Optional[Dict]:
    h, w = gray.shape[:2]
    if h < 20 or w < 80:
        return None

    rows = np.linspace(int(h * 0.35), int(h * 0.65), num=9, dtype=int)
    rows = np.clip(rows, 0, h - 1)
    sampled = gray[rows, :].astype(np.float32)
    profile = np.mean(sampled, axis=0)

    # Smooth 1D signal.
    kernel_size = 9 if w >= 120 else 5
    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, kernel_size), 0).reshape(-1)

    # Normalize and binarize (dark bars => 1).
    norm = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bits = (norm < thr).astype(np.uint8)

    # Transition extraction (used implicitly via run-lengths).
    transitions = np.where(np.diff(bits.astype(np.int8)) != 0)[0]
    if transitions.size < 20:
        return None

    runs = _run_lengths(bits)
    modules = _resample_to_modules(runs, expected_modules=95)
    if modules is None:
        return None

    ean13 = _decode_ean13_bits(modules)
    if not ean13:
        return None

    conf = decode_confidence(
        ean13,
        "EAN13",
        frame_quality_norm=quality_to_unit_interval(compute_frame_quality(gray).quality),
        decoder_agreement=0.7,
    )
    return {
        "decoded": ean13,
        "type": "EAN13",
        "confidence": conf,
        "decoder_agreement": 0.7,
    }
