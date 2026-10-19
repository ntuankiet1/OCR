# Decisions log

## Taken by default when the repo was scaffolded (2026-10-19) — revisit if wrong

| # | Decision | Default taken | Why |
|---|---|---|---|
| D1 | Training compute | On-prem; GPU variant of the `train` image (`docker-compose.gpu.yml`) | BRD §3/§9: images may not leave the organization |
| D2 | Inference location | Central CPU server | 1,100 parcels/day/property is low volume |
| D3 | Object storage | SeaweedFS | MinIO CE is archived and ships no images since Oct 2025 |
| D4 | Orchestration | `parcel-ocr retrain` + `dvc repro`, scheduled by cron/CI | Smallest moving parts for the 6-week plan; Prefect/Dagster can wrap it later |
| D5 | Model scope | One global bundle; per-property data = rosters + config only | Per-property models multiply versioning and gating |
| D6 | Retrain trigger | ≥1,000 new verified labels or 7 days with new labels; manual `--force` | Each cycle must finish calibration, gate and shadow |
| D7 | Promotion | Automatic up to `challenger`; human sign-off to `champion` | "Supervised" requirement |
| D8 | Gate tolerances | See `params.yaml: gate` | Proposal — confirm with the business |
| D9 | Text normalization | `src/text/normalize.py`, version `0.1-proposal` | AC02 requires agreement *before* testing |

## Still open (blocks something)

| Topic | Blocks | Owner |
|---|---|---|
| Who labels the ≥5,000 acceptance + ~2-3k regression images (two reviewers) | Frozen sets, gate, UAT | ? |
| PDPA legal basis + retention for using label images in training | Ingesting real images | ? |
| Review UI captures corrected *printed text* (not only the chosen resident) | Recognizer learning from corrections | Product |
| Carrier tracking-number formats (`configs/carriers.yaml`) | OCR-only tracking extraction | Label survey, week 1 |
| Type/size definitions + reference images + acceptance criteria (BRD §12) | Gating the type/size model | Business |
| Resident roster export format + refresh cadence (`data/rosters/<property>.json`) | Offline matching evaluation | Smartworld |
| GPU server CUDA version (`docker/train/requirements-gpu.txt`) | GPU training image | Infra |
