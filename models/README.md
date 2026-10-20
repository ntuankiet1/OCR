# models/

Model files are not committed (see `.gitignore`); copy them here. `docker-compose.yml` mounts this
folder read-only at `/models` in the service.

| File | What | Used by |
|---|---|---|
| `shipping-label-detector.onnx` | Finds the shipping label (one class, `waybill`) in a photo. YOLO26n, 640×640 input, output `[1, 5, 8400]` (cx, cy, w, h, score), NMS done in `src/detector.py`. | the service (`OCR_DETECTOR`) |
| `shipping-label-detector.pt` | The PyTorch weights it was exported from (keep for retraining/re-export). | `scripts/train_waybill_detector.py` |

Trained on 272 images from three public Roboflow datasets (merged to one class, see the script).
It is an Ultralytics YOLO model, so it is **AGPL-3.0**; check the licence before closed commercial
use, or retrain with RF-DETR (Apache-2.0).

Re-export after retraining:

```python
from ultralytics import YOLO
YOLO("shipping-label-detector.pt").export(format="onnx", imgsz=640, simplify=False, opset=17)
```

The PP-OCR models (`PP-OCRv5_server_det`, `th_PP-OCRv5_mobile_rec`, `PP-LCNet_x1_0_doc_ori`,
`PP-LCNet_x1_0_textline_ori`) are downloaded and converted to ONNX when the Docker image is built,
so they are not stored here. A fine-tuned recognizer can be used with `OCR_ENGINE=paddle` and
`OCR_REC_DIR`.
