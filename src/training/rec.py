"""Fine-tune th_PP-OCRv5_mobile_rec with PaddleOCR's own trainer (train image only).

We start from the upstream config and override only what we must, so upgrading PaddleOCR keeps
its tuned augmentations and schedule. The character dictionary is NOT changed: a new dictionary
would reset the output layer and throw away the pretrained Thai head.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import mlflow
import yaml


def _o(key: str, value) -> list[str]:
    return ["-o", f"{key}={json.dumps(value) if isinstance(value, list) else value}"]


def check_charset(gt_files: list[Path], dict_path: Path) -> dict[str, int]:
    """Characters in our labels that the model cannot emit (fine-tuning cannot fix these)."""
    vocab = set(dict_path.read_text(encoding="utf-8").splitlines()) | {" "}
    oov: Counter[str] = Counter()
    for f in gt_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            _, _, text = line.partition("\t")
            oov.update(ch for ch in text if ch not in vocab)
    return dict(oov.most_common(50))


def train_rec(params: dict, export_dir: Path, out_dir: Path, paddleocr_home: Path) -> Path:
    base_cfg = paddleocr_home / params["base_config"]
    if not base_cfg.exists():
        raise FileNotFoundError(f"{base_cfg} not found — is this the train image?")
    cfg = yaml.safe_load(base_cfg.read_text())
    dict_path = paddleocr_home / cfg["Global"]["character_dict_path"]

    train_gt, val_gt = export_dir / "rec_gt_train.txt", export_dir / "rec_gt_val.txt"
    if not train_gt.exists() or not train_gt.read_text().strip():
        raise RuntimeError("no recognition training lines exported — nothing to train")
    oov = check_charset([train_gt, val_gt], dict_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    pretrained = out_dir.parent / ".pretrained" / Path(params["pretrained_url"]).name
    if not pretrained.exists():
        pretrained.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(params["pretrained_url"], pretrained)

    save_dir = out_dir / "checkpoints"
    overrides = [
        *_o("Global.pretrained_model", str(pretrained.with_suffix(""))),
        *_o("Global.epoch_num", params["epochs"]),
        *_o("Global.save_model_dir", str(save_dir)),
        *_o("Global.use_gpu", str(params["use_gpu"]).lower()),
        *_o("Global.distributed", "false"),
        *_o("Global.save_epoch_step", max(1, params["epochs"] // 3)),
        *_o("Optimizer.lr.learning_rate", params["learning_rate"]),
        *_o("Optimizer.lr.warmup_epoch", params["warmup_epochs"]),
        *_o("Train.dataset.data_dir", str(export_dir)),
        *_o("Train.dataset.label_file_list", [str(train_gt)]),
        *_o("Train.sampler.first_bs", params["batch_size"]),
        *_o("Train.loader.batch_size_per_card", params["batch_size"]),
        *_o("Train.loader.num_workers", params["num_workers"]),
        *_o("Eval.dataset.data_dir", str(export_dir)),
        *_o("Eval.dataset.label_file_list", [str(val_gt)]),
        *_o("Eval.loader.num_workers", params["num_workers"]),
    ]

    mlflow.set_experiment("rec")
    with mlflow.start_run(run_name="th_PP-OCRv5_mobile_rec-finetune") as run:
        mlflow.log_params({k: v for k, v in params.items() if k != "pretrained_url"})
        mlflow.log_dict({"oov_chars": oov}, "oov_chars.json")
        # cwd = PaddleOCR repo: the upstream config uses repo-relative paths (dictionary).
        subprocess.run(
            [sys.executable, "tools/train.py", "-c", str(base_cfg), *overrides],
            cwd=paddleocr_home,
            check=True,
        )
        inference_dir = out_dir / "inference"
        subprocess.run(
            [
                sys.executable,
                "tools/export_model.py",
                "-c",
                str(base_cfg),
                *_o("Global.pretrained_model", str(save_dir / "best_accuracy")),
                *_o("Global.save_inference_dir", str(inference_dir)),
            ],
            cwd=paddleocr_home,
            check=True,
        )
        best = save_dir / "best_accuracy.states"
        if best.exists():
            mlflow.log_artifact(str(best))
        (out_dir / "run.json").write_text(
            json.dumps({"mlflow_run_id": run.info.run_id, "oov_chars": oov}, indent=2)
        )
    return inference_dir
