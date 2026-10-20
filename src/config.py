"""Service settings: environment variables prefixed OCR_ (or a .env file), see .env.example."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OCR_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    device: str = "cpu"  # "cpu" or "gpu:0"
    # "onnxruntime" runs ONNX copies of the Paddle models (same output, ~4x faster on CPU here);
    # "paddle" is Paddle's own engine. onnx_dir holds one folder per model (see the Dockerfile).
    engine: Literal["paddle", "onnxruntime"] = "paddle"
    onnx_dir: str = "/opt/onnx_models"
    # onnxruntime providers for the detector and the ONNX OCR models, e.g. "CUDAExecutionProvider";
    # empty = auto (CUDA if device is gpu, CoreML on native macOS, else CPU). See utils.py.
    providers: str | None = None

    det_model: str = "PP-OCRv5_server_det"
    rec_model: str = "th_PP-OCRv5_mobile_rec"
    rec_dir: str | None = None  # a fine-tuned recognizer, exported inference dir (engine=paddle)
    unwarp: bool = False  # UVDoc page unwarping, slow

    # Thresholds, per step. Raise them for fewer false positives, lower them to miss less.
    detector: str | None = None  # path to the shipping-label detector .onnx
    detector_conf: float = 0.25  # label detector: minimum box confidence
    detector_nms_iou: float = 0.5  # label detector: overlap above which boxes are merged
    locator_min_area: float = 0.15  # corner finder: smallest label, as a share of the crop
    crop_pad: float = 0.03  # share of the detected box added around it before cropping
    text_det_thresh: float | None = None  # text detector pixel threshold (None = Paddle default)
    text_det_box_thresh: float | None = None  # text detector box score (None = Paddle default)
    min_score: float = 0.5  # recognizer: lines scoring below this are dropped

    blur_threshold: float = 60.0  # Laplacian variance below this flags BLUR (see quality.py)
    model_version: str = "tma-ocr-0.1+ppocrv5-th"  # reported to Parcel, with the rules version

    enable_debug: bool = False  # serve /debug (step-by-step page): shows uploaded images, demo only

    max_side: int = 1600  # images are shrunk to this long side
    max_upload_mb: int = 15


settings = Settings()
