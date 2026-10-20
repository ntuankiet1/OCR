"""HTTP API:  GET /health   POST /ocr (multipart field `image`)."""

import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from . import config
from .schemas import OcrResponse

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
    return {"status": "ok" if "pipeline" in state else "loading", "device": config.DEVICE,
            "det": config.DET_MODEL, "rec": config.REC_DIR or config.REC_MODEL}


@app.post("/ocr", response_model=OcrResponse)
def ocr(image: UploadFile = File(...)):
    data = image.file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"image larger than {config.MAX_UPLOAD_MB} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a readable image")
    with lock:
        return state["pipeline"].run(img)
