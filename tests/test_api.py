from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src import api


class FakePipeline:
    def run(self, img):
        return {"waybill": {"found": False, "quad": None},
                "lines": [{"text": "ผู้รับ", "score": 0.9, "box": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
                "text": "ผู้รับ", "barcodes": [], "timing_ms": {"total": 1}}


@asynccontextmanager
async def _fake_lifespan(app):
    api.state["pipeline"] = FakePipeline()
    yield


def client():
    api.app.router.lifespan_context = _fake_lifespan
    return TestClient(api.app)


def test_ocr_and_health():
    png = cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))[1].tobytes()
    with client() as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/ocr", files={"image": ("a.png", png, "image/png")})
        assert r.status_code == 200 and r.json()["text"] == "ผู้รับ"
        assert c.post("/ocr", files={"image": ("a.txt", b"nope", "text/plain")}).status_code == 400


def test_debug_page_is_off_unless_enabled(monkeypatch):
    class FakeTracer:
        def __init__(self, pipeline):
            pass

        def trace(self, img):
            return {"steps": [], "total_ms": 1}

    png = cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))[1].tobytes()
    with client() as c:
        assert c.get("/debug").status_code == 404
        monkeypatch.setattr(api.settings, "enable_debug", True)
        monkeypatch.setattr(api, "PipelineTracer", FakeTracer)
        page = c.get("/debug")
        assert page.status_code == 200 and "/v1/debug" in page.text
        r = c.post("/v1/debug", files={"image": ("a.png", png, "image/png")})
        assert r.status_code == 200 and r.json() == {"steps": [], "total_ms": 1}
