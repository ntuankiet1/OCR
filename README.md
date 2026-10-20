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
curl -F image=@image.png http://localhost:8000/ocr
```

`POST /ocr` (multipart field `image`) returns `waybill` (found + corners), `lines`
(`text`, `score`, `box`), the joined `text`, `barcodes` and `timing_ms`. `GET /health` reports the
loaded models. Settings (device, models, a fine-tuned recognizer dir, thresholds) are env vars,
see [.env.example](.env.example). Models are baked into the image, so it runs offline.

## Layout

```
src/
├── api.py        FastAPI app (/health, /ocr)
├── pipeline.py   locate → OCR → barcodes
├── locate.py     find the waybill quad and flatten it (OpenCV now; trained detector later)
├── barcode.py    zxing-cpp
├── config.py     env settings
└── schemas.py    response models
scripts/
├── train_waybill_detector.py   download Roboflow datasets, EDA, train a nano YOLO (one class)
└── thai_ocr_benchmark.py       measure an OCR model on ThaiOCRBench (Kaggle/Colab/Docker)
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
