"""Shipping-label detector: YOLO exported to ONNX, run with onnxruntime (no torch needed)."""

import cv2
import numpy as np
import onnxruntime as ort

SIZE = 640


class ShippingLabelDetector:
    """Finds the shipping label in a photo (models/shipping-label-detector.onnx)."""

    def __init__(
        self,
        path: str,
        conf: float = 0.25,
        providers: list[str] | None = None,
        nms_iou: float = 0.5,
    ):
        self.session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.conf = conf
        self.nms_iou = nms_iou

    def detect(self, img: np.ndarray) -> list[tuple[list[float], float]]:
        """[(x1, y1, x2, y2), score] in image pixels, best first."""
        h, w = img.shape[:2]
        scale = SIZE / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
        top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
        canvas[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh))
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        out = self.session.run(None, {self.input: blob})[0][0]  # (5, 8400): cx cy w h score
        keep = out[4] >= self.conf
        cx, cy, bw, bh = out[:4, keep]
        scores = out[4, keep]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf, self.nms_iou)
        result = []
        for i in np.array(idx).reshape(-1):
            x, y, bw_, bh_ = boxes[i]
            box = [(x - left) / scale, (y - top) / scale, (x + bw_ - left) / scale,
                   (y + bh_ - top) / scale]
            box = [min(max(v, 0), lim) for v, lim in zip(box, (w, h, w, h), strict=True)]
            result.append((box, float(scores[i])))
        return sorted(result, key=lambda r: -r[1])
