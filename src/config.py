"""Service settings, read from environment variables (see .env.example)."""

import os


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


DEVICE = os.environ.get("OCR_DEVICE", "cpu")  # "cpu" or "gpu:0"
DET_MODEL = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_server_det")
REC_MODEL = os.environ.get("OCR_REC_MODEL", "th_PP-OCRv5_mobile_rec")
REC_DIR = os.environ.get("OCR_REC_DIR") or None  # a fine-tuned recognizer, exported inference dir
UNWARP = _flag("OCR_UNWARP", "false")  # UVDoc page unwarping, slow; off by default
MIN_SCORE = float(os.environ.get("OCR_MIN_SCORE", "0.5"))
MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", "1600"))  # images are shrunk to this long side
MAX_UPLOAD_MB = int(os.environ.get("OCR_MAX_UPLOAD_MB", "15"))
