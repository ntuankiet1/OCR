"""Fine-tune a Thai+English text-line recognizer with PaddleOCR (plain OCR, no VLM).

  --model mobile  th_PP-OCRv5_mobile_rec: starts from Paddle's Thai weights, same dictionary.
  --model server  PP-OCRv5_server_rec architecture with the Thai dictionary
                  (ppocrv5_th_dict.txt: Thai + Latin + digits). Starts from Paddle's server
                  weights; the output layers do not match the new dictionary, so PaddleOCR
                  skips them ("shape ... not matched") and they are trained from scratch.

Input: a folder made by make_thai_rec_data.py (train.txt, val.txt, images/).
Output: <out>/checkpoints (training) and <out>/inference (exported model for PaddleOCR).

Setup (Colab GPU):
    pip install paddlepaddle-gpu==3.2.2 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
    git clone --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR
    pip install -r PaddleOCR/requirements.txt
Run:
    python scripts/train_thai_rec.py --model mobile --data data/thai_rec_synth
    python scripts/train_thai_rec.py --model server --data data/thai_rec_synth
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

PRETRAINED = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/{}_pretrained.pdparams"
MODELS = {
    "mobile": {
        "config": "configs/rec/PP-OCRv5/multi_language/th_PP-OCRv5_mobile_rec.yaml",
        "pretrained": "th_PP-OCRv5_mobile_rec",
        "model_name": "th_PP-OCRv5_mobile_rec",
        "epochs": 20,
        "batch_size": 128,
        "lr": 0.0002,  # fine-tune: lower LR than upstream 5e-4
        "extra": {},
    },
    "server": {
        "config": "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml",
        "pretrained": "PP-OCRv5_server_rec",
        "model_name": "PP-OCRv5_server_rec",
        "epochs": 12,
        "batch_size": 64,
        "lr": 0.0005,  # new output layer: upstream LR
        "extra": {"Global.character_dict_path": "./ppocr/utils/dict/ppocrv5_th_dict.txt"},
    },
}


def overrides(pairs: dict) -> list[str]:
    """PaddleOCR's `-o` is argparse nargs="+": a REPEATED `-o` replaces the previous one, so
    every key=value must follow a single `-o` (as in PaddleOCR's own docs)."""
    kv = [f"{k}={json.dumps(v) if isinstance(v, list) else v}" for k, v in pairs.items()]
    return ["-o", *kv]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument(
        "--data", type=Path, required=True, help="folder with train.txt, val.txt, images/"
    )
    ap.add_argument("--out", type=Path, help="default: rec_runs/<model>")
    ap.add_argument(
        "--paddleocr", type=Path, default=Path("PaddleOCR"), help="PaddleOCR git repo (v3.7.0)"
    )
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    m = MODELS[args.model]
    repo, data = args.paddleocr.resolve(), args.data.resolve()
    out = (args.out or Path("rec_runs") / args.model).resolve()
    if not (repo / "tools" / "train.py").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v3.7.0",
                "https://github.com/PaddlePaddle/PaddleOCR",
                str(repo),
            ],
            check=True,
        )
    for f in ("train.txt", "val.txt"):
        if not (data / f).exists():
            raise SystemExit(f"{data / f} not found: run make_thai_rec_data.py first")

    import paddle

    use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    n_train = len((data / "train.txt").read_text(encoding="utf-8").splitlines())
    batch = max(1, min(args.batch_size or m["batch_size"], n_train))  # drop_last: never 0 batches
    epochs = args.epochs or m["epochs"]
    steps_per_epoch = max(1, n_train // batch)

    pretrained = out / "pretrained" / f"{m['pretrained']}_pretrained.pdparams"
    if not pretrained.exists():
        pretrained.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {pretrained.name} ...", flush=True)
        urllib.request.urlretrieve(PRETRAINED.format(m["pretrained"]), pretrained)

    ckpt_dir = out / "checkpoints"
    train_opts = overrides(
        {
            "Global.pretrained_model": pretrained.with_suffix(""),
            "Global.epoch_num": epochs,
            "Global.use_gpu": str(use_gpu).lower(),
            "Global.distributed": "false",
            "Global.save_model_dir": str(ckpt_dir),
            "Global.save_epoch_step": max(1, epochs // 4),
            "Global.eval_batch_step": [0, steps_per_epoch],  # evaluate once per epoch
            "Global.print_batch_step": 50,
            "Optimizer.lr.learning_rate": args.lr or m["lr"],
            "Optimizer.lr.warmup_epoch": 1,
            "Train.dataset.data_dir": str(data),
            "Train.dataset.label_file_list": [str(data / "train.txt")],
            "Train.sampler.first_bs": batch,
            "Train.loader.batch_size_per_card": batch,
            "Train.loader.num_workers": args.workers,
            "Eval.dataset.data_dir": str(data),
            "Eval.dataset.label_file_list": [str(data / "val.txt")],
            "Eval.loader.batch_size_per_card": min(batch, 64),
            "Eval.loader.num_workers": args.workers,
            **m["extra"],
        }
    )

    print(
        f"training {args.model}: {n_train} lines, batch {batch}, {epochs} epochs, "
        f"{'GPU' if use_gpu else 'CPU'}",
        flush=True,
    )
    # cwd = PaddleOCR repo: configs use repo-relative dictionary paths.
    subprocess.run(
        [sys.executable, "tools/train.py", "-c", m["config"], *train_opts],
        cwd=repo,
        check=True,
    )

    best = ckpt_dir / "best_accuracy"
    ckpt = best if best.with_suffix(".pdparams").exists() else ckpt_dir / "latest"
    inference = out / "inference"
    # The export must build the same output size as training (same dictionary).
    export_opts = overrides(
        {"Global.pretrained_model": ckpt, "Global.save_inference_dir": inference, **m["extra"]}
    )
    subprocess.run(
        [sys.executable, "tools/export_model.py", "-c", m["config"], *export_opts],
        cwd=repo,
        check=True,
    )

    # Accuracy on test.txt: fonts never used in train/val (made by make_thai_rec_data.py).
    if (data / "test.txt").exists():
        eval_opts = overrides(
            {
                "Global.checkpoints": ckpt,
                "Global.use_gpu": str(use_gpu).lower(),
                "Global.distributed": "false",
                "Eval.dataset.data_dir": str(data),
                "Eval.dataset.label_file_list": [str(data / "test.txt")],
                "Eval.loader.num_workers": args.workers,
                **m["extra"],
            }
        )
        print("\n=== accuracy on held-out fonts (test.txt):", flush=True)
        subprocess.run(
            [sys.executable, "tools/eval.py", "-c", m["config"], *eval_opts], cwd=repo, check=False
        )

    print(f"\nexported {ckpt.name} to {inference}")
    print(
        "evaluate it:\n  python scripts/thai_ocr_benchmark.py "
        f"--rec-name {m['model_name']} --rec-dir {inference} --tag {args.model}-finetuned"
    )


if __name__ == "__main__":
    main()
