# Model improvement pipeline (MLOps): plan

Parcel label reading, on-premises. Proposal for discussion.

## 1. Purpose

We start from pretrained models and **fine-tune them on your own parcel labels**, so accuracy
matches what your property really receives: your carriers, your label layouts and your photo
conditions (webcam, phone, lighting, labels stuck on boxes). After go-live the same pipeline keeps
the models improving from the checks your officers already do. Everything runs inside your
organisation: no external AI service is called, and images and recipient data never leave it.

## 2. What is fine-tuned

| Model | Job |
|---|---|
| Label detector | Finds the shipping label in a photo and its four corners, so it can be straightened |
| Text recognizer | Reads Thai and English text on the label |
| Field reading | Turns the text into carrier, tracking number, recipient and address |
| Confidence | Calibrated so that a stated confidence matches how often the field is really right |

## 3. Overview

```
Officer checks in the registration app
   │  photo, what the system read, what the officer saved
   ▼
1 Collect ─► 2 Curate and label ─► 3 Version the data ─► 4 Fine-tune ─► 5 Evaluate and gate
                                                                              │
7 Release and monitor ◄───────────── 6 Register and shadow ◄─────────────────┘
   │
   └── officers' corrections and drift signals ─► back to 1
```

## 4. Stages

**1. Collect.** Finished jobs from the registration app: the photo, what the system read, and what
the officer finally saved. Only values a person checked count as correct answers. Jobs the system
registered by itself are never used as answers.

**2. Curate and label.** Remove duplicates and unreadable photos, review a sample, and mark the
label corners where officer data does not provide them. Retakes of the same parcel stay together.
Where real photos are scarce, extra training photos may be generated; these are always marked and
never used for testing.

**3. Version the data.** Every dataset and every train, validation and test split is a numbered
version. The test sets are real photos, labelled by two reviewers, frozen, and never trained on.

**4. Fine-tune.** Training runs as a container on a GPU server, on a chosen data version. Each run
records its data version, code version, settings and results, so any model can be reproduced.

**5. Evaluate and gate.** The new model and the one in production are scored on the same frozen
tests, per field and per carrier. A new model moves on only if all hold:
- no field is worse than the current model beyond an agreed margin;
- no increase in the critical error: a wrong unit or recipient reported with high confidence;
- speed stays within budget on the target server;
- stated confidence matches measured accuracy.

The exact thresholds are agreed once, from the first real test set, and then fixed.

**6. Register and shadow.** The passing model is stored as the challenger and runs next to the live
model on real traffic. Its results are recorded but not used. A named approver reads the comparison
and decides whether to promote it.

**7. Release and monitor.** A release is one versioned bundle (detector, recognizer, field reading,
calibration, thresholds). Rolling back means pointing to the previous bundle: no retraining needed.
In production we watch, per carrier and per week: how often officers correct each field, how many
jobs are held for review, low-confidence rate, speed, and drift (new carriers, new label layouts,
darker or blurrier photos). A rising correction rate or a new layout opens a data-collection task;
it never retrains by itself.

## 5. Data protection

- All data, training and serving stay on-premises.
- Access is by role; recipient details are not written to general logs.
- Photos are kept only for the agreed retention period.
- Test data is never mixed into training.

## 6. Tools

| Need | Tool |
|---|---|
| Data and pipeline versioning | DVC, Git |
| Experiments and model registry | MLflow, PostgreSQL |
| Storage for datasets and bundles | MinIO (S3-compatible) |
| Labelling | Label Studio |
| Training and release | Docker, an on-premises image registry (Harbor) |

## 7. Roadmap

| Phase | Deliverable | What we need |
|---|---|---|
| 0 | Real test set and automatic scoring of every change | Real labelled photos |
| 1 | Export of finished jobs and dataset v1 | Access to the officers' checked jobs |
| 2 | Fine-tuned label detector, straightening before reading | Phase 0–1 data, GPU server |
| 3 | Fine-tuned text recognizer | Enough officer-checked jobs |
| 4 | Quality gate, release bundle, shadow run, promote and rollback | Phases 0–3 |
| 5 | Monitoring and drift alerts | Live traffic |
| 6 | Learned field reading | Enough corrected jobs |

## 8. To agree together

1. Success measures per field, and the margins for the gate.
2. Who labels and reviews, and how many photos per week they can handle.
3. How long photos are kept, and approval to use officer-checked jobs for training.
4. Who approves a promotion to production.
5. GPU server access: a booking window or a dedicated card.
