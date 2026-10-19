"""Download the pretrained PaddleOCR inference models once, into artifacts/base/<role>.

This is the only step that needs internet. Afterwards, bundles carry the models and the
pipeline runs fully offline (BRD §9). Run it in the train image and `dvc add artifacts/base`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

LOADERS = {
    "det": "TextDetection",
    "rec": "TextRecognition",
    "doc_orientation": "DocImgOrientationClassification",
    "textline_orientation": "TextLineOrientationClassification",
}


def fetch_base_models(models: dict[str, str], out_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    with tempfile.TemporaryDirectory() as cache:
        # Must be set before paddleocr/paddlex is imported: the cache root is read at import.
        os.environ["PADDLE_PDX_CACHE_HOME"] = cache
        import paddleocr

        for role, name in models.items():
            getattr(paddleocr, LOADERS[role])(model_name=name)  # downloads into the cache
            src = Path(cache) / "official_models" / name
            dst = out_dir / role
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            out[role] = dst
    return out
