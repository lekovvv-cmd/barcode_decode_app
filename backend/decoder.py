from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from fastapi import UploadFile
from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol

from .frame_quality import FrameQuality, compute_frame_quality, score_and_sort_frames
from .multi_frame import median_merge
from .preprocess import (
    anti_aliasing_prep,
    deskew,
    generate_variants,
    normalize_frame,
    upscale,
)
from .type_detection import estimate_code_type

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


async def decode_frames(files: List[UploadFile]) -> Dict:
    # 1. Load frames to OpenCV, limit to 10 and score quality.
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

    # Normalize size and convert to gray for quality scoring.
    normalized = [normalize_frame(im) for im in images]
    scored = score_and_sort_frames(tuple(normalized))

    # Keep best up to 10 by quality.
    best_pairs = scored[:10]
    best_frames = [img for img, _ in best_pairs]

    # 2. Type estimation using best frame.
    best_gray = best_frames[0]
    code_type_hint = estimate_code_type(best_gray)

    # 3. Orientation + deskew on each frame.
    oriented: List[np.ndarray] = []
    for img in best_frames:
        angle = 0.0  # reuse from best estimate? keep simple for now.
        if code_type_hint == "1D":
            # Try to orient once on best frame.
            angle = angle or 0.0
        rot = img  # skip heavy orientation for now; best_gray already oriented.
        rot = deskew(rot)
        oriented.append(rot)

    if code_type_hint == "2D":
        # 2D strategy: best single frame
        result = _decode_single_strategy(best_gray)
        if result:
            result["strategy"] = "best_frame"
            result["frames_used"] = 1
            return result

    # 1D strategy: try single frames then multi-frame median
    # Single-frame ensemble
    for gray in oriented:
        result = _decode_single_strategy(gray)
        if result:
            result["strategy"] = "single_frame"
            result["frames_used"] = 1
            return result

    # Multi-frame median merge
    try:
        med = median_merge(oriented)
    except ValueError:
        med = best_gray

    result = _decode_single_strategy(med)
    if result:
        result["strategy"] = "median_multiframe"
        result["frames_used"] = len(oriented)
        return result

    # Super resolution
    up = upscale(med, scale=2.5)
    result = _decode_single_strategy(up)
    if result:
        result["strategy"] = "super_resolution"
        result["frames_used"] = len(oriented)
        return result

    # Anti-aliasing pass
    aa = anti_aliasing_prep(up)
    result = _decode_single_strategy(aa)
    if result:
        result["strategy"] = "aliasing_prep"
        result["frames_used"] = len(oriented)
        return result

    # TODO: 1D signal extraction fallback (complex; omitted here but hook exists).

    return {"decoded": None}


def _decode_single_strategy(gray: np.ndarray) -> Optional[Dict]:
    """Run ensemble of decoders on one grayscale image."""
    variants = generate_variants(gray)
    for img in variants:
        # pyzbar first
        res = _decode_with_pyzbar(img)
        if res:
            return res

        # zxing-cpp if available
        if ZXING_AVAILABLE:
            res = _decode_with_zxing(img)
            if res:
                return res

        # OpenCV QRCodeDetector as backup for QR-like
        res = _decode_with_opencv(img)
        if res:
            return res

    return None


def _decode_with_pyzbar(gray: np.ndarray) -> Optional[Dict]:
    symbols = [
        ZBarSymbol.EAN13,
        ZBarSymbol.EAN8,
        ZBarSymbol.UPCA,
        ZBarSymbol.UPCE,
        ZBarSymbol.CODE128,
        ZBarSymbol.CODE39,
        ZBarSymbol.I25,
        ZBarSymbol.CODABAR,
        ZBarSymbol.QRCODE,
        ZBarSymbol.DATAMATRIX,
        ZBarSymbol.PDF417,
        ZBarSymbol.AZTEC,
    ]
    decoded = pyzbar_decode(gray, symbols=symbols)
    if not decoded:
        return None

    d = decoded[0]
    value = d.data.decode("utf-8", errors="ignore")
    btype = d.type or "UNKNOWN"
    return {
        "decoded": value,
        "type": btype,
        "confidence": 0.9,  # pyzbar has no explicit confidence, use heuristic later.
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
        "confidence": 0.9,
    }


def _decode_with_opencv(gray: np.ndarray) -> Optional[Dict]:
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(gray)
    if not data:
        return None
    return {
        "decoded": data,
        "type": "QRCODE",
        "confidence": 0.8,
    }

