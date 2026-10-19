"""Run the OCR pipeline (PP-OCRv5 det + Thai rec) on ThaiOCRBench.

Dataset: typhoon-ai/ThaiOCRBench (IJCNLP-AACL 2025, CC-BY-SA-4.0), 2,808 samples / 13 tasks.
A detection + recognition pipeline can only answer the 4 pure-reading tasks:

  Fine-grained text recognition (206)  read the text inside a given box  ← closest to a field crop
  Text recognition              (333)  read the text in a sign / photo
  Full-page OCR                 (197)  read a whole document page
  Handwritten content extraction(209)  handwriting (outside the BRD guarantee; informative)

The other 9 tasks (VQA, KIE as JSON, tables, charts...) need a language model and are skipped.
The dataset revision is pinned so results stay comparable over time.

This is public, general Thai text — NOT parcel labels. Use it to (1) pick between off-the-shelf
models and (2) check that fine-tuning on parcels did not destroy general Thai reading
("forgetting"). It is not an acceptance test: AC01 needs our own 5,000 real labels.
"""

from __future__ import annotations

import json
import re
import time
import traceback
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.benchmarks.thaiocrbench_metrics import SUPPORTED_TASKS, official_score, our_metrics
from src.inference.ocr import OcrLine

REPO = "typhoon-ai/ThaiOCRBench"
REVISION = "ca610d1ab33059e721251137ab1f19f936b87648"  # 2025-12-02
N_SHARDS = 4
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/data/test-{i:05d}-of-{n:05d}.parquet"

DEFAULT_TASKS = (
    "Fine-grained text recognition",
    "Text recognition",
    "Full-page OCR",
    "Handwritten content extraction",
)
_BBOX = re.compile(
    r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]"
)


# --------------------------------------------------------------------------- data
def download(cache_dir: Path, revision: str = REVISION) -> list[Path]:
    """Fetch the parquet shards once (public data coming in; nothing of ours goes out)."""
    out = []
    for i in range(N_SHARDS):
        dst = cache_dir / revision[:8] / f"test-{i:05d}-of-{N_SHARDS:05d}.parquet"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(HF_URL.format(repo=REPO, rev=revision, i=i, n=N_SHARDS), tmp)
            tmp.rename(dst)
        out.append(dst)
    return out


def load(shards: list[Path], tasks: list[str], limit: int | None = None) -> pd.DataFrame:
    unknown = set(tasks) - SUPPORTED_TASKS
    if unknown:
        raise ValueError(f"not an OCR task (needs a VLM): {sorted(unknown)}")
    cols = ["Id", "Task", "category", "question", "answer", "image"]
    df = pd.concat(
        (
            pq.read_table(s, columns=cols, filters=[("Task", "in", tasks)]).to_pandas()
            for s in shards
        ),
        ignore_index=True,
    ).sort_values(["Task", "Id"])
    if limit:
        df = df.groupby("Task", group_keys=False).head(limit)
    return df.reset_index(drop=True)


def decode_image(cell) -> np.ndarray:
    data = cell["bytes"] if isinstance(cell, dict) else cell
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("undecodable image")
    return img


# --------------------------------------------------------------------------- geometry
def parse_bbox(question: str) -> tuple[float, float, float, float] | None:
    """Fine-grained questions give [x1, y1, x2, y2] on a 0-1000 grid (OCRBench-v2 convention)."""
    m = _BBOX.search(question)
    return tuple(float(v) for v in m.groups()) if m else None


def crop_normalized(
    image: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.1
) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000)
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    x1, x2 = int(max(0, x1 - px)), int(min(w, x2 + px))
    y1, y2 = int(max(0, y1 - py)), int(min(h, y2 + py))
    return image[y1:y2, x1:x2]


def reading_order(lines: list[OcrLine]) -> str:
    """Group detected lines into rows (top→bottom), then left→right within a row."""
    if not lines:
        return ""
    items = []
    for ln in lines:
        xs, ys = [p[0] for p in ln.polygon], [p[1] for p in ln.polygon]
        items.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(xs), ln.text))
    items.sort(key=lambda it: it[0])
    heights = sorted(it[1] for it in items)
    tol = 0.5 * (heights[len(heights) // 2] or 1.0)
    rows, current = [], [items[0]]
    for it in items[1:]:
        row_cy = sum(c[0] for c in current) / len(current)
        if abs(it[0] - row_cy) <= tol:
            current.append(it)
        else:
            rows.append(current)
            current = [it]
    rows.append(current)
    return "\n".join(" ".join(it[3] for it in sorted(r, key=lambda it: it[2])) for r in rows)


# --------------------------------------------------------------------------- run
@dataclass
class ModelSpec:
    models_dir: Path | None  # None → PaddleOCR downloads by name (benchmark only)
    det: str
    rec: str
    doc_orientation: str
    textline_orientation: str
    label: str


def predict_sample(engine, task: str, question: str, image: np.ndarray) -> tuple[str, str]:
    if task == "Fine-grained text recognition":
        box = parse_bbox(question)
        if box is None:
            raise ValueError("no bbox in question")
        crop = crop_normalized(image, box)
        text = reading_order(engine.read(crop))
        if text:
            return text, "crop+det+rec"
        return engine.recognize(crop)[0], "crop+rec"
    return reading_order(engine.read(image)), "page"


def run(engine, df: pd.DataFrame, progress_every: int = 25) -> pd.DataFrame:
    rows = []
    for n, r in enumerate(df.itertuples(), 1):
        rec = {
            "id": r.Id,
            "task": r.Task,
            "category": r.category,
            "question": r.question,
            "answer": r.answer,
        }
        t0 = time.perf_counter()
        try:
            pred, mode = predict_sample(engine, r.Task, r.question, decode_image(r.image))
            rec.update(prediction=pred, mode=mode, error=None)
        except Exception as exc:  # one bad sample must not kill a long run
            rec.update(prediction="", mode="error", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        rec["latency_ms"] = 1000 * (time.perf_counter() - t0)
        rec["official_score"] = official_score(r.Task, rec["prediction"], r.answer)
        rec.update(our_metrics(rec["prediction"], r.answer))
        rows.append(rec)
        if n % progress_every == 0 or n == len(df):
            running = sum(x["official_score"] for x in rows) / len(rows)
            print(f"[{n}/{len(df)}] {r.Task}: running official={running:.3f}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report
def summarize(pred: pd.DataFrame, min_category_n: int = 10) -> dict:
    tasks = {}
    for task, p in pred.groupby("task"):
        tasks[task] = {
            "n": len(p),
            "official_score": float(p["official_score"].mean()),
            "official_metric": "ANLS" if task == "Text recognition" else "BMFL",
            "cer_mean": float(p["cer"].mean()),
            "cer_median": float(p["cer"].median()),
            "exact_match": float(p["exact"].mean()),
            "latency_ms_p50": float(p["latency_ms"].quantile(0.5)),
            "latency_ms_p95": float(p["latency_ms"].quantile(0.95)),
            "errors": int(p["error"].notna().sum()),
        }
    by_cat = (
        pred.groupby(["task", "category"])["official_score"].agg(["count", "mean"]).reset_index()
    )
    by_cat = by_cat[by_cat["count"] >= min_category_n]
    return {
        "tasks": tasks,
        "by_category": [
            {
                "task": r["task"],
                "category": r["category"],
                "n": int(r["count"]),
                "official_score": float(r["mean"]),
            }
            for _, r in by_cat.iterrows()
        ],
    }


def _short(s: str, n: int = 90) -> str:
    s = (s or "").replace("\n", " ⏎ ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def write_report(
    out_dir: Path, pred: pd.DataFrame, summary: dict, spec: ModelSpec, meta: dict
) -> Path:
    source = spec.models_dir or "PaddleOCR download"
    lines = [
        f"# ThaiOCRBench — {spec.label}",
        "",
        f"- Run: {meta['created_at']} · dataset `{REPO}` @ `{REVISION[:8]}` "
        f"· limit per task: {meta['limit'] or 'none'}",
        f"- Models: det `{spec.det}` · rec `{spec.rec}` · from `{source}`",
        "- Official score = the benchmark's own metric (BMFL or ANLS, 0-1, higher is better).",
        "  CER and exact match use the project's normalization and ignore whitespace.",
        "",
        "| Task | n | Official | Metric | CER mean | CER median | Exact | p50 ms | Errors |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for task, t in summary["tasks"].items():
        cells = [
            task,
            t["n"],
            f"{t['official_score']:.3f}",
            t["official_metric"],
            f"{t['cer_mean']:.3f}",
            f"{t['cer_median']:.3f}",
            f"{100 * t['exact_match']:.1f}%",
            f"{t['latency_ms_p50']:.0f}",
            t["errors"],
        ]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    if summary["by_category"]:
        lines += [
            "",
            "## By category (n ≥ 10)",
            "",
            "| Task | Category | n | Official |",
            "|---|---|---:|---:|",
        ]
        for c in summary["by_category"]:
            lines.append(
                f"| {c['task']} | {c['category']} | {c['n']} | {c['official_score']:.3f} |"
            )
    for task, p in pred.groupby("task"):
        lines += [
            "",
            f"## Worst 10 — {task}",
            "",
            "| Id | CER | Answer | Prediction |",
            "|---|---:|---|---|",
        ]
        for r in p.sort_values("cer", ascending=False).head(10).itertuples():
            lines.append(f"| {r.id} | {r.cer:.2f} | {_short(r.answer)} | {_short(r.prediction)} |")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def benchmark(
    spec: ModelSpec,
    cache_dir: Path,
    out_root: Path,
    tasks: list[str],
    limit: int | None,
    log_mlflow: bool,
) -> Path:
    from src.inference.ocr import PaddleOcrEngine

    df = load(download(cache_dir), tasks, limit)
    print(f"ThaiOCRBench: {len(df)} samples · {df['Task'].value_counts().to_dict()}", flush=True)
    engine = PaddleOcrEngine(
        spec.models_dir, spec.det, spec.rec, spec.doc_orientation, spec.textline_orientation
    )

    pred = run(engine, df)
    summary = summarize(pred)
    meta = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "limit": limit,
        "dataset": REPO,
        "revision": REVISION,
        "models": spec.__dict__ | {"models_dir": str(spec.models_dir)},
    }

    out_dir = (
        out_root / f"{datetime.now(UTC):%Y%m%d-%H%M}-{re.sub(r'[^A-Za-z0-9_.-]+', '_', spec.label)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(meta | summary, indent=2, ensure_ascii=False))
    report = write_report(out_dir, pred, summary, spec, meta)

    if log_mlflow:
        import mlflow

        mlflow.set_experiment("benchmark-thaiocrbench")
        with mlflow.start_run(run_name=spec.label):
            mlflow.log_params(
                {
                    "det": spec.det,
                    "rec": spec.rec,
                    "revision": REVISION[:8],
                    "limit": limit or "none",
                }
            )
            for task, t in summary["tasks"].items():
                key = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")
                mlflow.log_metrics(
                    {f"{key}.official": t["official_score"], f"{key}.cer": t["cer_mean"]}
                )
            mlflow.log_artifact(str(report))
            mlflow.log_artifact(str(out_dir / "metrics.json"))
    return report
