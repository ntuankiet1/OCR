# Smartworld Parcel OCR

OCR for parcel labels in Thai condominiums. A label photo goes in; the recipient, unit, tracking
number and carrier come out, and uncertain cases go to an officer for review. Everything runs
on-prem with **plain vision OCR models** (PaddleOCR PP-OCR): no vision-language models, no
external AI services.

The business requirements document (BRD v1.6) is not in this repo; put it under `docs/` locally
(`docs/*.pdf` is git-ignored). References like "BRD §9" or "AC01" in the code point to it.

---

## Start here: the 3 scripts used right now

The `scripts/` folder holds standalone tools. They don't need anything else from this repo, so
each one runs as is on Kaggle, Colab or in Docker.

| Script | What it does |
|---|---|
| [scripts/thai_ocr_benchmark.py](scripts/thai_ocr_benchmark.py) | Measures an OCR model on [ThaiOCRBench](https://huggingface.co/datasets/typhoon-ai/ThaiOCRBench) (public Thai benchmark, 945 images, 4 reading tasks). Prints exact match, CER, similarity per task and saves `predictions.csv` + `summary.json`. |
| [scripts/make_thai_rec_data.py](scripts/make_thai_rec_data.py) | Generates synthetic Thai + English **text lines that look like parcel labels** (names, units, addresses, phones, tracking numbers, carriers) in looped and loopless fonts, and **evaluates the dataset against our requirements** (`report.json`, `samples.jpg`). |
| [scripts/train_thai_rec.py](scripts/train_thai_rec.py) | Trains the text recognizer with PaddleOCR: `--model mobile` fine-tunes `th_PP-OCRv5_mobile_rec`; `--model server` trains the bigger `PP-OCRv5_server_rec` architecture with the Thai dictionary. Exports the model and scores it on fonts it never saw. |

### Train and compare on Kaggle or Colab (GPU)

```bash
# GPU runtime (Kaggle: Accelerator "GPU T4", Internet on; Colab: T4)
apt-get -qq install -y libraqm0 libfribidi0 fonts-thai-tlwg      # correct Thai text rendering
pip uninstall -y -q paddlepaddle                                  # never keep the CPU build
pip install -q paddlepaddle-gpu==3.2.2 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -q paddleocr==3.7.0 pythainlp fonttools rapidfuzz
git clone -q https://github.com/ntuankiet1/OCR.git
git clone -q --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR
pip install -q -r PaddleOCR/requirements.txt
cd OCR

python scripts/make_thai_rec_data.py --out ../thai_rec_synth --n 200000          # data + fit report
python scripts/train_thai_rec.py --model mobile --data ../thai_rec_synth --paddleocr ../PaddleOCR
python scripts/train_thai_rec.py --model server --data ../thai_rec_synth --paddleocr ../PaddleOCR

# before vs after, on the ThaiOCRBench box-reading task
python scripts/thai_ocr_benchmark.py --tasks "Fine-grained text recognition" --tag baseline
python scripts/thai_ocr_benchmark.py --tasks "Fine-grained text recognition" --tag mobile \
    --rec-dir rec_runs/mobile/inference
python scripts/thai_ocr_benchmark.py --tasks "Fine-grained text recognition" --tag server \
    --rec-name PP-OCRv5_server_rec --rec-dir rec_runs/server/inference
```

Baseline (current model, full benchmark): 20% exact match on box reading. The main failure is
**loopless Thai fonts read as Latin letters**, which is what the synthetic data targets.

---

## How the OCR works (plain vision, PP-OCR)

```
label photo
 → text detector   PP-OCRv5_server_det / mobile_det   finds text lines (box + score)
 → recognizer      th_PP-OCRv5_mobile_rec → fine-tuned   reads each line (text + confidence)
 → barcode reader  zxing-cpp                            tracking number from the barcode
 → field rules     recipient / unit / tracking / carrier, then match against the resident roster
 → decision        auto-register only when every check passes, otherwise officer review
```

Later: a **layout detector** (PP-DocLayout / PicoDet, trained on our labels) to find the
recipient and sender blocks before reading. This is the PP-Structure approach with our own
classes.

| Name | What it is |
|---|---|
| PaddlePaddle | Baidu's deep-learning framework (like PyTorch) |
| PaddleOCR | Baidu's OCR toolkit built on it; we use version 3.7.0 |
| PP-OCR | PaddleOCR's plain OCR system: detector + recognizer, pure vision |
| PP-Structure | Document parsing on top of PP-OCR (layout, tables); its layout method is reused later |

---

## Project skeleton

```
scripts/                 ★ standalone tools used now (benchmark, data, training)
notebooks/               Jupyter versions (benchmark with images; full-pipeline demo)

src/                     the production pipeline package, imported as `src.*`
├── cli.py               the `parcel-ocr` command (every pipeline step is a subcommand)
├── schemas.py           data contracts: label fields, label sources, review-export format
├── settings.py          paths and settings (env vars PARCEL_OCR_*)
├── inference/           runs on one label photo: quality → barcode → OCR → fields → match → decision
├── data/                ingest review exports → append-only label store → snapshots + splits → exports
├── training/            recognizer + type/size classifier training (used by dvc.yaml)
├── calibration/         turns model scores into proven auto-accept thresholds (BR05)
├── evaluation/          metrics, evaluation, champion-vs-challenger gate
├── registry/            "bundle" (a versioned set of models + thresholds) and MLflow promote/rollback
├── benchmarks/          ThaiOCRBench with its official metrics (CLI version of the script)
└── text/                Thai text normalization used when comparing text

configs/                 carriers.yaml (carrier rules), type_size.yaml (parcel size classes 1-10)
dvc.yaml, params.yaml    the retraining pipeline (stages) and its parameters
docker/, docker-compose.yml, Makefile   on-prem stack: Postgres, SeaweedFS (S3), MLflow, Label Studio
tests/                   unit tests (run in Docker: `make test`)
docs/decisions.md        decisions taken and still open
```

**Now vs later.** For the demo only `scripts/` matters. `src/` with DVC, MLflow and the gate is
the supervised retraining pipeline for production: it needs real labelled data from the pilot
and is described below.

---

## The retraining pipeline (production, later)

```
review exports ──► parcel-ocr ingest ──► versioned data (DVC)
parcel-ocr retrain: dvc repro (snapshot → train → calibrate → bundle → evaluate → gate)
                    ──► register as "challenger" ──► shadow mode
parcel-ocr promote <version> --approved-by <name>     (human sign-off → "champion")
parcel-ocr rollback --reason ... --by ...             (back to the previous champion)
```

Rules the code enforces:
- Auto-registered jobs are never used as training labels; only human-checked labels are.
- The regression and acceptance test sets are frozen, never trained on, and scored only with
  two-reviewer labels.
- Splits are grouped by parcel (property + tracking number), so retakes never leak into tests.
- A field is auto-accepted only above a threshold proven on calibration data (BR05).
- Any wrong unit/recipient assignment blocks a new model (BRD §9, AC03).

On-prem stack: `make env && make build && make test`, then `make up && make init`.
`make notebook` opens JupyterLab with the full-pipeline demo
([notebooks/run_pipeline.ipynb](notebooks/run_pipeline.ipynb)) on synthetic data.

---

## Status

| Part | State |
|---|---|
| `scripts/thai_ocr_benchmark.py` | Run on Kaggle/Colab GPU (full benchmark) and in Docker |
| `scripts/make_thai_rec_data.py`, `scripts/train_thai_rec.py` | Smoke-tested in Docker on CPU (tiny data, 1 epoch) |
| `src/` pipeline | Unit tests 52/53 pass (1 registry test to fix); not yet run end to end |
