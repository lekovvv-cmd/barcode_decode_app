from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List

from .decoder import decode_frames

app = FastAPI(title="Barcode Decode Backend", version="1.0.0")


@app.post("/decode")
async def decode(frames: List[UploadFile] = File(...)) -> JSONResponse:
    if not frames:
      raise HTTPException(status_code=400, detail="No frames provided")

    # Limit frames to first 10 as per spec.
    limited_frames = frames[:10]

    try:
        decoded = await decode_frames(limited_frames)
    except Exception as exc:
        # Fail soft: do not crash the service.
        return JSONResponse(
            status_code=200,
            content={
                "decoded": None,
                "error": str(exc),
            },
        )

    return JSONResponse(status_code=200, content=decoded)


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

