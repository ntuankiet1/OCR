"""`parcel-ocr` command line. Heavy imports stay inside commands so the core image (no
Paddle/PyTorch) can run everything except training and prediction."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from src.settings import get_settings, load_params

app = typer.Typer(no_args_is_help=True, help="Smartworld Parcel OCR — supervised training pipeline")
infra = typer.Typer(no_args_is_help=True, help="Infrastructure helpers")
app.add_typer(infra, name="infra")

S = get_settings()


def _echo(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _store():
    from src.data.store import ImageStore

    return ImageStore(S.image_cache_dir, S.s3_endpoint_url)


def _skip_training(out: Path, reason: str) -> None:
    """Leave a marker so the DVC stage still has an output and the bundle falls back."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps({"trained": False, "reason": reason}))
    typer.echo(reason)


# --------------------------------------------------------------------------- infra
@infra.command("init-buckets")
def init_buckets() -> None:
    """Create the S3 buckets (idempotent)."""
    from src.data.store import ensure_buckets

    _echo({"created": ensure_buckets(S.buckets, S.s3_endpoint_url)})


@infra.command("init-dirs")
def init_dirs() -> None:
    """Create the data directories DVC stages depend on."""
    for d in (
        S.labels_dir,
        S.images_index_dir,
        S.frozen_dir,
        S.data_dir / "rosters",
        S.artifacts_dir / "base",
    ):
        d.mkdir(parents=True, exist_ok=True)
        (d / ".keep").touch()
    _echo({"ok": True})


# --------------------------------------------------------------------------- data
@app.command()
def ingest(path: Path) -> None:
    """Ingest a review-service export (JSONL of ReviewEvent) into the append-only stores."""
    from src.data.ingest import ingest_jsonl

    stats = ingest_jsonl(path, S.images_index_dir, S.labels_dir)
    _echo(stats.__dict__)


@app.command()
def snapshot() -> None:
    """Resolve labels, assign splits and write data/snapshots/current."""
    from src.data.snapshot import build_snapshot

    _echo(
        build_snapshot(
            S.images_index_dir, S.labels_dir, S.frozen_dir, S.snapshot_dir, load_params()["split"]
        )
    )


@app.command()
def freeze(
    split: Annotated[str, typer.Argument(help="regression | acceptance")],
    keys_file: Annotated[Path, typer.Option(help="text file, one group_key per line")],
    note: Annotated[str, typer.Option(help="why / how this set was sampled")],
) -> None:
    """Create a frozen evaluation set (immutable once written)."""
    from src.data.split import write_frozen
    from src.schemas import Split

    keys = [k.strip() for k in keys_file.read_text().splitlines() if k.strip()]
    _echo({"written": str(write_frozen(S.frozen_dir, Split(split), keys, note)), "n": len(keys)})


@app.command()
def export() -> None:
    """Export train/val/calibration to PaddleOCR rec format and the type/size CSVs."""
    from src.data.export import export_all

    _echo(export_all(S.snapshot_dir, _store(), S.exports_dir))


# --------------------------------------------------------------------------- training (train image)
@app.command("fetch-base-models")
def fetch_base_models() -> None:
    """Download pretrained PaddleOCR inference models into artifacts/base (needs internet once)."""
    from src.training.base_models import fetch_base_models as fetch

    _echo(
        {
            k: str(v)
            for k, v in fetch(load_params()["base_models"], S.artifacts_dir / "base").items()
        }
    )


@app.command("train-rec")
def train_rec() -> None:
    """Fine-tune the Thai text recognizer."""
    from src.training.rec import train_rec as train

    out = S.artifacts_dir / "rec"
    if not (S.exports_dir / "rec" / "rec_gt_train.txt").read_text().strip():
        _skip_training(
            out, "no recognition training data yet — bundle keeps the pretrained recognizer"
        )
        return
    _echo(
        {
            "inference_dir": str(
                train(load_params()["train_rec"], S.exports_dir / "rec", out, S.paddleocr_home)
            )
        }
    )


@app.command("train-type-size")
def train_type_size() -> None:
    """Train the parcel type/size classifier."""
    import pandas as pd

    from src.training.type_size import train_type_size as train

    out = S.artifacts_dir / "type_size"
    csv = S.exports_dir / "type_size" / "type_size_train.csv"
    if len(pd.read_csv(csv)) == 0:
        _skip_training(
            out, "no type/size training data yet — bundle leaves type/size blank (review)"
        )
        return
    _echo({"onnx": str(train(load_params()["train_type_size"], S.exports_dir / "type_size", out))})


# --------------------------------------------------------------------------- bundle
@app.command()
def assemble(
    out: Annotated[Path, typer.Option()] = Path("artifacts/bundle"),
    provisional: Annotated[
        bool, typer.Option(help="no thresholds: used to score the calibration split")
    ] = False,
) -> None:
    """Assemble models + calibration + rules into a versioned bundle directory."""
    from src.calibration.run import empty_calibration
    from src.registry.bundle import assemble_bundle, hash_path

    params = load_params()
    base, art = S.artifacts_dir / "base", S.artifacts_dir
    rec_ft = art / "rec" / "inference"
    components = {
        "det": (base / "det", "paddle-inference", f"pretrained:{params['base_models']['det']}"),
        "rec": (
            rec_ft,
            "paddle-inference",
            "fine-tuned:" + json.loads((art / "rec" / "run.json").read_text())["mlflow_run_id"],
        )
        if rec_ft.exists()
        else (base / "rec", "paddle-inference", f"pretrained:{params['base_models']['rec']}"),
        "doc_orientation": (
            base / "doc_orientation",
            "paddle-inference",
            f"pretrained:{params['base_models']['doc_orientation']}",
        ),
        "textline_orientation": (
            base / "textline_orientation",
            "paddle-inference",
            f"pretrained:{params['base_models']['textline_orientation']}",
        ),
    }
    onnx = art / "type_size" / "model.onnx"
    if onnx.exists():
        run_id = json.loads((art / "type_size" / "run.json").read_text())["mlflow_run_id"]
        components["type_size"] = (onnx, "onnx", f"fine-tuned:{run_id}")
    missing = [n for n, (p, _, _) in components.items() if not p.exists()]
    if missing:
        raise typer.BadParameter(
            f"missing components {missing} — run `parcel-ocr fetch-base-models` first"
        )

    if provisional:
        calibration_file = art / "calibration" / "provisional.json"
        empty_calibration(calibration_file, params["calibration"]["fields"])
    else:
        calibration_file = art / "calibration" / "calibration.json"

    meta = {
        "det_name": params["base_models"]["det"],
        "rec_name": params["base_models"]["rec"],
        "doc_orientation_name": params["base_models"]["doc_orientation"],
        "textline_orientation_name": params["base_models"]["textline_orientation"],
        "type_size_image_size": params["train_type_size"]["image_size"],
        "params": {"quality": params["quality"], "matching": params["matching"]},
    }
    manifest = assemble_bundle(
        out,
        components,
        calibration_file,
        [S.configs_dir / "carriers.yaml", S.configs_dir / "type_size.yaml"],
        data_snapshot=hash_path(S.snapshot_dir)[:16] if S.snapshot_dir.exists() else None,
        pipeline_meta=meta,
    )
    _echo(
        {
            "bundle_id": manifest.bundle_id,
            "bundle_version": manifest.bundle_version,
            "out": str(out),
        }
    )


@app.command()
def predict(
    bundle: Annotated[Path, typer.Option()],
    split: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    allow_acceptance: Annotated[bool, typer.Option(help="UAT only")] = False,
) -> None:
    """Run a bundle over one snapshot split (train image)."""
    from src.evaluation.predict import predict_split

    _echo(
        predict_split(
            bundle, S.snapshot_dir, split, _store(), S.data_dir / "rosters", out, allow_acceptance
        )
    )


@app.command()
def calibrate() -> None:
    """Fit calibrators and auto-accept thresholds on the calibration split."""
    from src.calibration.run import calibrate as run
    from src.calibration.run import threshold_summary

    res = run(
        S.artifacts_dir / "predictions" / "calibration",
        S.snapshot_dir,
        S.artifacts_dir / "type_size" / "calibration_logits.npz",
        load_params()["calibration"],
        S.artifacts_dir / "calibration" / "calibration.json",
    )
    _echo(threshold_summary(res))


# --------------------------------------------------------------------------- evaluation + gate
@app.command()
def evaluate(
    pred: Annotated[Path, typer.Option()],
    split: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
) -> None:
    """Score predictions against gold labels."""
    from src.evaluation.evaluate import evaluate as run

    m = run(pred, S.snapshot_dir, split, S.frozen_dir, out)
    _echo(
        {
            k: m[k]
            for k in (
                "n_jobs",
                "auto_rate",
                "wrong_assignments",
                "whole_record_accuracy",
                "latency_ms",
            )
        }
    )


@app.command("evaluate-champion")
def evaluate_champion(out: Annotated[Path, typer.Option()] = Path("reports/champion")) -> None:
    """Re-score the current champion on the current regression set (train image)."""
    from src.data.split import frozen_set_hash
    from src.evaluation.evaluate import evaluate as run
    from src.evaluation.predict import predict_split
    from src.registry.bundle import read_manifest
    from src.registry.mlflow_registry import CHAMPION, download_bundle
    from src.schemas import Split

    tmp = S.artifacts_dir / "champion"
    shutil.rmtree(tmp, ignore_errors=True)
    bundle = download_bundle(S.registered_model_name, CHAMPION, tmp)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    if bundle is None:
        (out / "NONE").write_text("no champion yet — gate runs in first_model mode\n")
        _echo({"champion": None})
        return
    pred_dir = tmp / "predictions"
    predict_split(
        bundle, S.snapshot_dir, Split.REGRESSION.value, _store(), S.data_dir / "rosters", pred_dir
    )
    m = run(pred_dir, S.snapshot_dir, Split.REGRESSION.value, S.frozen_dir, out)
    _echo(
        {
            "champion": read_manifest(bundle).bundle_id,
            "eval_set_hash": frozen_set_hash(S.frozen_dir, Split.REGRESSION),
            "auto_rate": m["auto_rate"],
        }
    )


@app.command()
def gate(
    challenger: Annotated[Path, typer.Option()] = Path("reports/challenger"),
    champion: Annotated[Path, typer.Option()] = Path("reports/champion"),
    out: Annotated[Path, typer.Option()] = Path("reports/gate.json"),
    strict: Annotated[bool, typer.Option(help="exit 1 when the gate fails (CI)")] = False,
) -> None:
    """Compare challenger vs champion; write the gate report."""
    from src.evaluation.gate import EvalRun, GateConfig, run_gate

    chal = EvalRun.load(challenger)
    if chal is None:
        raise typer.BadParameter(f"no metrics.json in {challenger}")
    report = run_gate(chal, EvalRun.load(champion), GateConfig(**load_params()["gate"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    for c in report.checks:
        typer.echo(f"{'PASS' if c.passed else 'FAIL'}  {c.name:<32} {c.detail}")
    typer.echo(f"\nGATE {'PASSED' if report.passed else 'FAILED'} ({report.mode})")
    if strict and not report.passed:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- registry
@app.command()
def register(
    bundle: Annotated[Path, typer.Option()] = Path("artifacts/bundle"),
    gate_report: Annotated[Path, typer.Option()] = Path("reports/gate.json"),
    metrics: Annotated[Path, typer.Option()] = Path("reports/challenger/metrics.json"),
) -> None:
    """Register a gated bundle in MLflow and set it as `challenger` (→ shadow mode)."""
    from src.registry.mlflow_registry import register_bundle

    _echo(
        {
            "challenger_version": register_bundle(
                bundle, gate_report, metrics, S.registered_model_name
            )
        }
    )


@app.command()
def promote(
    version: str,
    approved_by: Annotated[str, typer.Option(help="person signing off after shadow mode")],
    note: Annotated[str, typer.Option()] = "",
) -> None:
    """Promote a registered version to `champion` (human sign-off required)."""
    from src.registry.mlflow_registry import promote as run

    _echo(run(S.registered_model_name, version, approved_by, note))


@app.command()
def rollback(
    reason: Annotated[str, typer.Option()],
    by: Annotated[str, typer.Option()],
) -> None:
    """Point `champion` back to the previous champion."""
    from src.registry.mlflow_registry import rollback as run

    _echo(run(S.registered_model_name, reason, by))


@app.command()
def status() -> None:
    """Show champion / challenger / previous and promotion history."""
    from src.registry.mlflow_registry import status as run

    _echo(run(S.registered_model_name))


# ------------------------------------------------------------------- benchmarks (train image)
@app.command("benchmark-thai")
def benchmark_thai(
    limit: Annotated[int | None, typer.Option(help="samples per task (smoke run)")] = None,
    task: Annotated[
        list[str] | None, typer.Option(help="repeatable; default: the 4 OCR tasks")
    ] = None,
    bundle: Annotated[
        Path | None, typer.Option(help="evaluate a bundle's models (e.g. a challenger)")
    ] = None,
    det_model: Annotated[
        str | None, typer.Option(help="PaddleOCR model name; downloads it")
    ] = None,
    rec_model: Annotated[
        str | None, typer.Option(help="PaddleOCR model name; downloads it")
    ] = None,
    mlflow_log: Annotated[bool, typer.Option("--mlflow/--no-mlflow")] = False,
) -> None:
    """Score the OCR pipeline on ThaiOCRBench (public Thai benchmark, 4 reading tasks)."""
    from src.benchmarks.thaiocrbench import DEFAULT_TASKS, ModelSpec, benchmark

    base = load_params()["base_models"]
    names = {
        "det": base["det"],
        "rec": base["rec"],
        "doc_orientation": base["doc_orientation"],
        "textline_orientation": base["textline_orientation"],
    }
    if bundle is not None:
        meta = json.loads((bundle / "pipeline.json").read_text())
        from src.registry.bundle import read_manifest

        names = {
            "det": meta["det_name"],
            "rec": meta["rec_name"],
            "doc_orientation": meta["doc_orientation_name"],
            "textline_orientation": meta["textline_orientation_name"],
        }
        models_dir, label = bundle / "models", f"bundle-{read_manifest(bundle).bundle_id[:8]}"
    elif det_model or rec_model:
        names |= {k: v for k, v in {"det": det_model, "rec": rec_model}.items() if v}
        models_dir, label = None, f"{names['det']}+{names['rec']}"
    elif (S.artifacts_dir / "base" / "det").exists():
        models_dir, label = S.artifacts_dir / "base", f"base-{names['rec']}"
    else:
        models_dir, label = None, f"{names['det']}+{names['rec']}"

    report = benchmark(
        ModelSpec(models_dir=models_dir, label=label, **names),
        cache_dir=S.data_dir / "benchmarks" / "thaiocrbench",
        out_root=S.reports_dir / "benchmarks" / "thaiocrbench",
        tasks=list(task or DEFAULT_TASKS),
        limit=limit,
        log_mlflow=mlflow_log,
    )
    typer.echo(report.read_text())
    typer.echo(f"report: {report}")


# --------------------------------------------------------------------------- orchestration
def trigger_state(params: dict) -> dict:
    from src.data.labelstore import read_labels
    from src.schemas import LabelSource

    labels = read_labels(S.labels_dir)
    verified = int((labels["source"] != LabelSource.SYNTHETIC.value).sum())
    stats_file = S.snapshot_dir / "stats.json"
    last = json.loads(stats_file.read_text()) if stats_file.exists() else {}
    new = verified - last.get("n_verified_labels", 0)
    days = (
        (datetime.now(UTC) - datetime.fromisoformat(last["created_at"])).days
        if "created_at" in last
        else None
    )
    t = params["trigger"]
    due = new >= t["min_new_verified_labels"] or (
        new > 0 and (days is None or days >= t["max_days_since_last_snapshot"])
    )
    return {
        "verified_labels": verified,
        "new_since_snapshot": new,
        "days_since_snapshot": days,
        "due": due,
    }


def _run(cmd: list[str]) -> None:
    typer.echo(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.command()
def retrain(
    force: Annotated[bool, typer.Option(help="ignore the trigger")] = False,
    push: Annotated[bool, typer.Option(help="dvc push data + artifacts to the remote")] = True,
) -> None:
    """One supervised cycle: trigger → version data → re-score champion → dvc repro → gate →
    register challenger. Promotion stays manual (`parcel-ocr promote`)."""
    state = trigger_state(load_params())
    _echo(state)
    if not (state["due"] or force):
        typer.echo("trigger not met — nothing to do")
        return
    _run(["dvc", "add", str(S.labels_dir), str(S.images_index_dir)])
    evaluate_champion()
    _run(["dvc", "repro"])
    report = json.loads(Path("reports/gate.json").read_text())
    if report["passed"]:
        register()
    else:
        typer.echo("gate failed — champion unchanged; see reports/gate.json")
    if push:
        _run(["dvc", "push"])
    typer.echo("Commit dvc.lock and data/*.dvc so this cycle's data + model lineage is in git.")


if __name__ == "__main__":
    app()
