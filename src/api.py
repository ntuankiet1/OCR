"""HTTP API:  GET /health   POST /ocr (multipart field `image`)."""

import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .config import settings
from .debug import PipelineTracer
from .schemas import OcrResponse, ReadResponse

state: dict = {}
lock = threading.Lock()  # one Paddle predictor, one request at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .pipeline import OcrPipeline

    state["pipeline"] = OcrPipeline()
    yield
    state.clear()


app = FastAPI(title="TMA OCR", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if "pipeline" in state else "loading", "device": settings.device,
            "det": settings.det_model, "rec": settings.rec_dir or settings.rec_model}


def _decode(upload: UploadFile) -> np.ndarray:
    data = upload.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"image larger than {settings.max_upload_mb} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a readable image")
    return img


@app.post("/ocr", response_model=OcrResponse)
def ocr(image: UploadFile = File(...)):
    """Raw result: text lines, barcodes, timings (for debugging)."""
    img = _decode(image)
    with lock:
        return state["pipeline"].run(img)


@app.post("/v1/read", response_model=ReadResponse)
def read(
    image: UploadFile = File(...),
    property_id: str | None = Form(None),  # accepted for the contract; not used yet
    request_id: str | None = Form(None),
    capture_method: str | None = Form(None),
):
    """The label read as Parcel's fields (recipient, address, carrier, tracking, ...)."""
    img = _decode(image)
    with lock:
        return state["pipeline"].run(img)["reading"]


def require_debug() -> None:
    if not settings.enable_debug:
        raise HTTPException(404, "debug is off (set OCR_ENABLE_DEBUG=true)")


@app.get("/debug", response_class=HTMLResponse, dependencies=[Depends(require_debug)])
def debug_page():
    """Upload a photo and see every pipeline step (demo only)."""
    return Path(__file__).with_name("debug.html").read_text()


@app.post("/v1/debug", dependencies=[Depends(require_debug)])
def debug(image: UploadFile = File(...)):
    img = _decode(image)
    with lock:
        return PipelineTracer(state["pipeline"]).trace(img)
