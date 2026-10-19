"""Train the 10-class parcel type/size classifier (timm, train image only).

Outputs: model.onnx (serving), best.pt, calibration_logits.npz (for temperature scaling).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

NUM_CLASSES = 10


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dataset(csv: Path, image_size: int, train: bool):
    import torch
    from PIL import Image
    from torchvision.transforms import v2 as T

    df = pd.read_csv(csv)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = T.Compose(
        [
            T.ToImage(),
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomRotation(15),
            T.ColorJitter(0.3, 0.3, 0.2),
            T.ToDtype(torch.float32, scale=True),
            norm,
        ]
        if train
        else [
            T.ToImage(),
            T.Resize((image_size, image_size)),
            T.ToDtype(torch.float32, scale=True),
            norm,
        ]
    )

    class DS(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(df)

        def __getitem__(self, i: int):
            row = df.iloc[i]
            return tf(Image.open(row.image_path).convert("RGB")), int(row.label_index)

    return DS()


def _logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    out, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            out.append(model(x.to(device)).float().cpu().numpy())
            ys.append(y.numpy())
    if not out:
        return np.zeros((0, NUM_CLASSES), np.float32), np.zeros((0,), np.int64)
    return np.concatenate(out), np.concatenate(ys)


def train_type_size(params: dict, export_dir: Path, out_dir: Path) -> Path:
    import timm
    import torch

    _seed_all(params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size, bs, nw = params["image_size"], params["batch_size"], params["num_workers"]
    train_csv = export_dir / "type_size_train.csv"
    if not train_csv.exists() or len(pd.read_csv(train_csv)) == 0:
        raise RuntimeError("no type/size training images exported — nothing to train")

    loaders = {
        split: torch.utils.data.DataLoader(
            _dataset(export_dir / f"type_size_{split}.csv", size, train=split == "train"),
            batch_size=bs,
            shuffle=split == "train",
            num_workers=nw,
        )
        for split in ("train", "val", "calibration")
    }
    model = timm.create_model(params["model"], pretrained=True, num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params["epochs"])
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])

    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, best_path = -1.0, out_dir / "best.pt"
    mlflow.set_experiment("type_size")
    with mlflow.start_run(run_name=params["model"]) as run:
        mlflow.log_params(params)
        for epoch in range(params["epochs"]):
            model.train()
            for x, y in loaders["train"]:
                opt.zero_grad()
                loss = loss_fn(model(x.to(device)), y.to(device))
                loss.backward()
                opt.step()
            sched.step()
            logits, y = _logits(model, loaders["val"], device)
            acc = float((logits.argmax(1) == y).mean()) if len(y) else 0.0
            mlflow.log_metrics({"val_acc": acc, "train_loss": float(loss)}, step=epoch)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_path)

        model.load_state_dict(torch.load(best_path, map_location=device))
        logits, y = _logits(model, loaders["calibration"], device)
        np.savez(out_dir / "calibration_logits.npz", logits=logits, labels=y)

        model.eval().cpu()
        onnx_path = out_dir / "model.onnx"
        # Fixed batch of 1 (serving classifies one parcel at a time). torch >= 2.9 exports via
        # the dynamo exporter by default, which needs `onnxscript` (in the train requirements).
        torch.onnx.export(
            model,
            (torch.randn(1, 3, size, size),),
            str(onnx_path),
            input_names=["image"],
            output_names=["logits"],
            dynamo=True,
        )
        mlflow.log_metric("best_val_acc", best_acc)
        mlflow.log_artifact(str(onnx_path))
        (out_dir / "run.json").write_text(
            json.dumps({"mlflow_run_id": run.info.run_id, "best_val_acc": best_acc}, indent=2)
        )
    return onnx_path
