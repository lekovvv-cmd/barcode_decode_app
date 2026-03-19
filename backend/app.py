import logging
import json
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from .decoder import decode_frames
except ImportError:
    from decoder import decode_frames

try:
    from .yolo_detector import warmup_models as _warmup_models
except Exception:
    try:
        from yolo_detector import warmup_models as _warmup_models
    except Exception:
        _warmup_models = None

app = FastAPI(title="Barcode Decode Backend", version="1.0.0")


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root.setLevel(logging.INFO)

    # Make sure backend module logs are visible under uvicorn.
    for name in ("backend.decoder", "backend.yolo_detector", "decoder", "yolo_detector"):
        logging.getLogger(name).setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger("backend.app")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.get("/version")
async def version() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "service": "barcode-decoder",
            "version": "0.1",
        },
    )


@app.on_event("startup")
async def startup_warmup() -> None:
    logger.info("startup: backend app initialized")
    if _warmup_models is None:
        logger.warning("startup: YOLO warmup skipped (module unavailable)")
        return
    try:
        _warmup_models()
        logger.info("startup: YOLO warmup complete")
    except Exception:
        # Keep service available even if optional YOLO fallback cannot warm up.
        logger.exception("startup: YOLO warmup failed")
        pass


@app.post("/decode")
async def decode(
    frames: List[UploadFile] = File(...),
    exclude_ids: str = Form(default="[]"),
) -> JSONResponse:
    try:
        excluded = json.loads(exclude_ids)
        if not isinstance(excluded, list):
            excluded = []
    except Exception:
        excluded = []

    logger.info("decode_request: frames=%d excluded=%d", len(frames), len(excluded))
    if not frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    # Limit frames to first 10 as per spec.
    limited_frames = frames[:10]

    try:
        decoded = await decode_frames(limited_frames, exclude_ids=excluded)
    except Exception as exc:
        # Fail soft: do not crash the service.
        logger.exception("decode_request: unhandled error")
        return JSONResponse(
            status_code=200,
            content={
                "decoded": None,
                "error": str(exc),
            },
        )

    logger.info(
        "decode_response: decoded=%s strategy=%s type=%s",
        decoded.get("decoded"),
        decoded.get("strategy"),
        decoded.get("type"),
    )
    return JSONResponse(status_code=200, content=decoded)


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
