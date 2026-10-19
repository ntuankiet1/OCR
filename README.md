# Smartworld Parcel OCR — training pipeline

Supervised, versioned retraining for the parcel-label OCR described in the business requirements
document (BRD v1.6 — not in this repo; place it under `docs/` locally, `docs/*.pdf` is git-ignored).
Section references such as "BRD §9" or "AC01" in the code point to that document.
Everything runs on-prem in Docker; no image or text is sent to an external AI service.

## Stack

| Concern | Tool | Where |
|---|---|---|
| Object storage (images, DVC remote, MLflow artifacts) | SeaweedFS (S3 API) | `seaweedfs` service |
| Data versioning | DVC (`dvc.yaml`, `dvc.lock`) | `pipeline` / `train` images |
| Experiments + model registry | MLflow 3 (Postgres backend) | `mlflow` service, :5000 |
| Annotation / adjudication | Label Studio | `label-studio` service, :8080 |
| OCR | PaddleOCR 3.x — PP-OCRv5 mobile det + `th_PP-OCRv5_mobile_rec` | `train` image |
| Type/size classifier | timm MobileNetV3 → ONNX | `train` image |

Images: `core` (data, calibration, evaluation, gate, registry — multi-arch), `test`, and
`train` (adds Paddle + PyTorch; `linux/amd64` only because Paddle has no Linux arm64 wheels).

## Quick start

```bash
make env          # .env from template — change the secrets
make build        # core, test, mlflow images
make test         # test suite in a container
make up && make init
make build-train && make base-models   # one-off, needs internet
```

## The retraining loop

```
review exports (JSONL) ──► parcel-ocr ingest ──► data/labels, data/images_index (append-only, dvc add)
                                                        │
parcel-ocr retrain:  trigger? ─► evaluate-champion ─► dvc repro ─► gate ─► register (alias: challenger)
                                                                                 │ shadow mode
                                         parcel-ocr promote <v> --approved-by <name>   (alias: champion)
                                         parcel-ocr rollback --reason ... --by ...     (back to previous)
```

`dvc repro` stages: snapshot → export → train_rec / train_type_size → assemble_provisional →
predict_calibration → calibrate → assemble → predict_regression → evaluate → gate.

**Trigger** (`params.yaml: trigger`): ≥1,000 new human-verified labels, or 7 days with new labels.

**Gate** (`params.yaml: gate`): any wrong unit/recipient assignment blocks; no field may drop
more than 0.2 pp or be significantly worse (paired McNemar); no carrier/capture-mode/property
subgroup may drop more than 1 pp; auto rate and p95 latency must hold.

**Promotion is always manual** — a named approver after the shadow period.

## Public Thai benchmark (ThaiOCRBench)

Paddle publishes no public Thai benchmark (its Thai numbers come from in-house sets), so we use
[ThaiOCRBench](https://huggingface.co/datasets/typhoon-ai/ThaiOCRBench) (IJCNLP-AACL 2025,
CC-BY-SA-4.0, revision pinned in `src/benchmarks/thaiocrbench.py`). An OCR pipeline can answer 4
of its 13 tasks: fine-grained text recognition (text inside a given box), text recognition,
full-page OCR, and handwritten content extraction. Scores use the benchmark's own metrics
(BMFL / ANLS, re-implemented from its eval code) plus our CER and exact match.

```bash
make build-train
make benchmark-thai LIMIT=20                       # smoke run, 20 samples per task
make benchmark-thai                                # full run, ~945 images
make benchmark-thai ARGS="--bundle artifacts/bundle --mlflow"   # a fine-tuned bundle, logged
```

Output: `reports/benchmarks/thaiocrbench/<run>/report.md`, `metrics.json` and
`predictions.parquet` (each sample's answer, prediction and scores).

This is general Thai text, not parcel labels: use it to choose between off-the-shelf models and
to catch **forgetting** after fine-tuning on parcels. It is not an acceptance test.

## Data rules that the code enforces

- **Auto-registered jobs are never training labels.** No human checked them. Only officer
  edits/confirmations, annotator, audit and adjudicated labels count.
- **Frozen sets** (`data/frozen/regression.txt`, `data/frozen/acceptance.txt`, in git) are
  immutable, disjoint, and scored only with two-reviewer (`adjudicated`) labels. Export for
  training refuses them; prediction on `acceptance` needs `--allow-acceptance` (UAT only).
- **Splits are a salted hash of `group_key`** (property + tracking number), so retakes of one
  parcel never straddle train and test, and a parcel keeps its split across snapshots.
- **Calibrated thresholds**: a field is auto-accepted only above a score whose calibration subset
  has a Clopper-Pearson lower bound ≥ 99.5% (BR05). No threshold → that field always goes to review.

## Bundle = the unit of versioning

`artifacts/bundle/` holds every model, the calibration, `pipeline.json` (runtime params) and the
rule files, with a content-hash `bundle_id` stamped on every job (BR16). `verify_bundle()` detects
any modified file. Serving loads `models:/parcel-ocr-bundle@champion`.

## Review-service export contract

One JSON object per line — see `ReviewEvent` in [src/schemas.py](src/schemas.py). `predicted` is
what the bundle produced, `final` is what was saved after review. `final` must carry the
**printed text** for edited fields (not only the chosen resident), otherwise the recognizer
cannot learn from the correction.

## Layout

```
src/            package (imported as `src.*`), CLI entry point `parcel-ocr` → src/cli.py
  data/         ingest, append-only label store, splits, snapshots, exports
  training/     PaddleOCR rec fine-tune, type/size classifier, base-model download
  calibration/  threshold selection, temperature / isotonic calibration
  evaluation/   metrics, evaluate, gate, predict
  registry/     bundle assembly + integrity, MLflow register / promote / rollback
  inference/    quality, barcode, OCR adapter, rule KIE, roster matching, decision rules
configs/        carriers.yaml (rule KIE), type_size.yaml (BRD classes 1-10)
docker/         Dockerfiles, training requirements (cpu/gpu), postgres init
docs/           BRD, decisions.md
```
