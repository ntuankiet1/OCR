"""Evaluate the Thai OCR model on ThaiOCRBench.

1. Load the benchmark (typhoon-ai/ThaiOCRBench, the 4 tasks an OCR model can answer)
2. Load the model: PaddleOCR PP-OCRv5_mobile_det + th_PP-OCRv5_mobile_rec
3. Run inference on every image
4. Report accuracy per task and save the predictions

Setup (Python 3.10-3.13):
    pip install paddleocr==3.7.0 pandas pyarrow rapidfuzz
    # GPU (e.g. Colab T4). paddlepaddle-gpu 3.x is only on Paddle's index, not PyPI:
    pip install paddlepaddle-gpu==3.2.2 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
    # or CPU:
    pip install paddlepaddle==3.2.2
    # Never install both: they are the same `paddle` module. The CPU one never uses the GPU.

Run:
    python thai_ocr_benchmark.py --limit 20   # quick check: 20 images per task
    python thai_ocr_benchmark.py              # full benchmark: 945 images (downloads 1.9 GB once)
"""

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import paddle
import pandas as pd
from rapidfuzz.distance import Levenshtein

REPO = "typhoon-ai/ThaiOCRBench"
REVISION = "ca610d1ab33059e721251137ab1f19f936b87648"  # pinned dataset version
SHARD_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/{rev}/data/test-{i:05d}-of-00004.parquet"
)
TASKS = [
    "Fine-grained text recognition",  # read the text inside a given box
    "Text recognition",  # read the text on a sign / photo
    "Full-page OCR",  # read a whole document page
    "Handwritten content extraction",  # handwriting
]
DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "th_PP-OCRv5_mobile_rec"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


# --------------------------------------------------------------------------- 1. data
def load_benchmark(cache_dir: Path, limit: int | None) -> pd.DataFrame:
    """Download the 4 parquet shards once (1.9 GB) and keep only the OCR tasks."""
    frames = []
    for i in range(4):
        path = cache_dir / f"test-{i:05d}-of-00004.parquet"
        if not path.exists():
            print(f"downloading shard {i + 1}/4 ...", flush=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(SHARD_URL.format(repo=REPO, rev=REVISION, i=i), tmp)
            tmp.rename(path)
        columns = ["Id", "Task", "category", "question", "answer", "image"]
        frames.append(pd.read_parquet(path, columns=columns, filters=[("Task", "in", TASKS)]))
    df = pd.concat(frames, ignore_index=True).sort_values(["Task", "Id"])
    if limit:
        df = df.groupby("Task").head(limit)
    return df.reset_index(drop=True)


def to_bgr(image_cell) -> np.ndarray:
    data = image_cell["bytes"] if isinstance(image_cell, dict) else image_cell
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def crop_box(image: np.ndarray, question: str) -> np.ndarray:
    """Fine-grained questions give the region as [x1, y1, x2, y2] on a 0-1000 grid."""
    x1, y1, x2, y2 = map(float, re.search(r"\[([\d.\s,]+)\]", question).group(1).split(","))
    h, w = image.shape[:2]
    x1, x2, y1, y2 = x1 * w / 1000, x2 * w / 1000, y1 * h / 1000, y2 * h / 1000
    pad_x, pad_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    return image[
        int(max(0, y1 - pad_y)) : int(min(h, y2 + pad_y)),
        int(max(0, x1 - pad_x)) : int(min(w, x2 + pad_x)),
    ]


# --------------------------------------------------------------------------- 2. model
def hide_broken_torch() -> None:
    """paddleocr imports paddlex → modelscope, whose logger imports torch whenever it is
    installed. On Colab, installing paddlepaddle-gpu replaces CUDA libraries (NCCL) that the
    preinstalled torch needs, so `import torch` fails ("undefined symbol ..."). This script
    never uses torch: if it cannot be imported, make it look absent (modelscope skips it)."""
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):  # OSError: a CUDA library file is missing
        sys.modules["torch"] = None  # importlib.util.find_spec("torch") now returns None


def load_model(device: str, det_name: str, rec_name: str, rec_dir: Path | None):
    """rec_dir: a fine-tuned recognizer exported by train_thai_rec.py (None = Paddle's model).
    rec_name must be the architecture it was trained from (e.g. PP-OCRv5_server_rec)."""
    hide_broken_torch()
    from paddleocr import PaddleOCR, TextRecognition

    source = rec_dir or "official weights"
    print(f"loading {det_name} + {rec_name} ({source}) on {device} (paddle {paddle.__version__})")
    common = {
        "device": device,
        # paddle 3.3.x crashes on CPU with oneDNN ("ConvertPirAttribute2RuntimeAttribute")
        "enable_mkldnn": device == "cpu" and not paddle.__version__.startswith("3.3"),
    }
    rec_dir_kw = {"text_recognition_model_dir": str(rec_dir)} if rec_dir else {}
    ocr = PaddleOCR(
        text_detection_model_name=det_name,
        text_recognition_model_name=rec_name,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        **rec_dir_kw,
        **common,
    )
    # For crops where the detector finds no text.
    rec_only = TextRecognition(
        model_name=rec_name, **({"model_dir": str(rec_dir)} if rec_dir else {}), **common
    )
    return ocr, rec_only


# --------------------------------------------------------------------------- 3. inference
def read_text(ocr, image: np.ndarray) -> str:
    return "\n".join(text for res in ocr.predict(image) for text in res["rec_texts"])


def predict(ocr, rec_only, task: str, question: str, image: np.ndarray) -> str:
    if task == "Fine-grained text recognition":
        crop = crop_box(image, question)
        text = read_text(ocr, crop)
        if not text:
            text = next(iter(rec_only.predict(crop)))["rec_text"]
        return text
    return read_text(ocr, image)


# --------------------------------------------------------------------------- 4. metrics
def normalize(text: str) -> str:
    """Ignore whitespace and unify Thai digits / SARA AM so only real reading errors count."""
    text = unicodedata.normalize("NFC", text or "").translate(THAI_DIGITS)
    text = text.replace("\u0e4d\u0e32", "\u0e33")  # NIKHAHIT + SARA AA == SARA AM
    return re.sub(r"\s+", "", text)


def score(prediction: str, answer: str) -> dict:
    p, g = normalize(prediction), normalize(answer)
    dist = Levenshtein.distance(p, g)
    return {
        "exact": p == g,  # whole answer read exactly right
        # Character error rate, lower is better. Capped at 1.0 (= completely wrong): reading
        # extra text around a short answer would otherwise push it far above 1.
        "cer": min(dist / max(len(g), 1), 1.0),
        "similarity": 1 - dist / max(len(p), len(g), 1),  # normalized edit similarity
        "answer_found": bool(g) and g in p,  # answer appears somewhere in the output
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="images per task (default: all)")
    parser.add_argument("--device", help="gpu:0 or cpu (default: GPU if available)")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "thaiocrbench")
    parser.add_argument("--det-name", default=DET_MODEL, help="e.g. PP-OCRv5_server_det")
    parser.add_argument("--rec-name", default=REC_MODEL, help="architecture of the recognizer")
    parser.add_argument("--rec-dir", type=Path, help="fine-tuned recognizer (exported folder)")
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS, help="subset of tasks (default: all 4)"
    )
    parser.add_argument("--tag", help="results go to <out>/<tag> (e.g. baseline, mobile-finetuned)")
    args = parser.parse_args()
    out = args.out / args.tag if args.tag else args.out

    has_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    device = args.device or ("gpu:0" if has_gpu else "cpu")

    df = load_benchmark(args.cache, args.limit)
    if args.tasks:
        df = df[df["Task"].isin(args.tasks)].reset_index(drop=True)
    print(f"{len(df)} images: {df['Task'].value_counts().to_dict()}")
    ocr, rec_only = load_model(device, args.det_name, args.rec_name, args.rec_dir)

    rows = []
    start = time.perf_counter()
    for n, r in enumerate(df.itertuples(), 1):
        t0 = time.perf_counter()
        try:
            prediction, error = predict(ocr, rec_only, r.Task, r.question, to_bgr(r.image)), ""
        except Exception as exc:  # keep going; a failed image scores 0
            prediction, error = "", f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "id": r.Id,
                "task": r.Task,
                "category": r.category,
                "answer": r.answer,
                "prediction": prediction,
                "seconds": time.perf_counter() - t0,
                "error": error,
                **score(prediction, r.answer),
            }
        )
        if n % 25 == 0 or n == len(df):
            print(f"[{n}/{len(df)}] {time.perf_counter() - start:.0f}s elapsed")

    results = pd.DataFrame(rows)
    summary = results.groupby("task").agg(
        images=("id", "size"),
        exact_match=("exact", "mean"),
        cer_mean=("cer", "mean"),
        cer_median=("cer", "median"),
        similarity=("similarity", "mean"),
        answer_found=("answer_found", "mean"),
        sec_per_image=("seconds", "mean"),
        errors=("error", lambda e: int((e != "").sum())),
    )
    print("\n" + summary.round(3).to_string())

    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "predictions.csv", index=False)
    meta = {
        "dataset": REPO,
        "revision": REVISION,
        "det": args.det_name,
        "rec": args.rec_name,
        "rec_dir": str(args.rec_dir) if args.rec_dir else None,
        "device": device,
        "paddle": paddle.__version__,
        "limit": args.limit,
    }
    (out / "summary.json").write_text(
        json.dumps({**meta, "tasks": summary.round(4).to_dict(orient="index")}, indent=2)
    )
    print(f"\nsaved {out / 'predictions.csv'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
