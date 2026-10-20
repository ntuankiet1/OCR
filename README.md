# TMA-OCR

OCR service for Thai/English parcel waybills (Smartworld, "AP Thailand"). A photo goes in, the
waybill text and barcodes come out as JSON. Plain vision OCR (PaddleOCR PP-OCR), no VLM, runs
fully on-prem. Retraining/MLOps lives in a separate future service, not here.

```
photo → find waybill → straighten → orientation → OCR (server_det + th_PP-OCRv5_mobile_rec) → JSON
                                                  └→ barcodes (zxing-cpp)
```

## Run

```bash
make build          # PADDLE=gpu make build  for CUDA (also set OCR_DEVICE=gpu:0 in .env)
make up             # http://localhost:8000  (interactive docs at /docs)
curl -F image=@samples/kex-label.png http://localhost:8000/ocr
```

`POST /ocr` (multipart field `image`) returns `waybill` (found + corners), `lines`
(`text`, `score`, `box`), the joined `text`, `barcodes` and `timing_ms`. `GET /health` reports the
loaded models. Settings (device, models, a fine-tuned recognizer dir, thresholds) are env vars,
see [.env.example](.env.example). Models are baked into the image, so it runs offline.

## Speed (CPU, Apple-silicon Docker, 12 cores, measured through `POST /ocr`)

| Engine | KEX label (392×522) | DHL photo (1337×955 crop) |
|---|---|---|
| `OCR_ENGINE=paddle` | 2.6 s | 9.5 s |
| `OCR_ENGINE=onnxruntime` (image default) | 0.6 s | 2.0 s |

The ONNX models are converted from the same Paddle models at build time (`paddle2onnx`); the
recognized text is identical. The waybill detector adds ~30 ms. Re-measure on the real server:
Paddle's oneDNN path can do better on x86, and on GPU use `OCR_ENGINE=paddle`.

## Layout

```
src/
├── api.py        FastAPI app (/health, /ocr)
├── pipeline.py   OcrPipeline: locate → OCR → barcodes
├── locate.py     LabelLocator: crop to the detected label, find its corners, flatten it
├── detector.py   ShippingLabelDetector (YOLO, ONNX Runtime)
├── ocr.py        TextReader (PaddleOCR PP-OCRv5; Paddle or ONNX engine)
├── barcode.py    BarcodeReader (zxing-cpp)
├── utils.py      shared helpers (pick_providers: CUDA / CoreML / CPU, resize_max_side)
├── config.py     Settings (pydantic-settings, env vars OCR_*)
└── schemas.py    response models
scripts/
├── train_waybill_detector.py   download Roboflow datasets, EDA, train a nano YOLO (one class)
└── thai_ocr_benchmark.py       measure an OCR model on ThaiOCRBench (Kaggle/Colab/Docker)
models/                          shipping-label-detector.{onnx,pt} (not committed, see models/README.md)
samples/                         local test images (not committed)
tests/                           make test
```

## Waybill detector (next step)

```bash
pip install roboflow "ultralytics>=8.4" pillow matplotlib pyyaml   # Kaggle/Colab GPU
export ROBOFLOW_API_KEY=...
python scripts/train_waybill_detector.py download
python scripts/train_waybill_detector.py eda      # merges, de-duplicates, splits; eda_report.json, eda.png, samples.jpg
python scripts/train_waybill_detector.py train
```

Check the dataset versions in `SOURCES` first. Ultralytics is AGPL-3.0; for a closed commercial
service prefer RF-DETR (Apache-2.0). The data prep is reusable for either.

## Notes

- Baseline model reads a real printed Thai label almost perfectly; fine-tuning on synthetic text
  made real photos worse, so real waybill data is needed before retraining.
- Pinned to paddlepaddle 3.2.2 (3.3.x breaks CPU oneDNN).
- Never commit client documents (`docs/*.pdf`); the repo is public.
