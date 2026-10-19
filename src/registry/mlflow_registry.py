"""MLflow model registry: register, promote (human sign-off) and roll back bundles.

Aliases on the registered model `parcel-ocr-bundle`:
  champion    — serving. The inference service loads models:/parcel-ocr-bundle@champion.
  challenger  — passed the gate; running in shadow mode.
  previous    — the champion before the current one (one-step rollback target).

Promotion history is kept as a registered-model tag, so rollback can walk back more than one
step and every move is attributable (who, when, why).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.evaluation.gate import GateReport
from src.registry.bundle import read_manifest, verify_bundle

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous"
HISTORY_TAG = "champion_history"
EXPERIMENT = "parcel-ocr-bundles"


class RegistryError(RuntimeError):
    pass


class BundleModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper so `mlflow.pyfunc.load_model("models:/parcel-ocr-bundle@champion")` works.
    Input: DataFrame with `image_path` (+ optional `property_id`); output: one row per job."""

    def load_context(self, context) -> None:
        self.bundle_dir = Path(context.artifacts["bundle"])
        self._pipeline = None

    def predict(self, context, model_input, params=None):
        import pandas as pd

        from src.inference.pipeline import InferencePipeline  # heavy deps: train/serve image only

        if self._pipeline is None:
            self._pipeline = InferencePipeline.from_bundle(self.bundle_dir)
        rows = [self._pipeline.run(p).to_record() for p in model_input["image_path"]]
        return pd.DataFrame(rows)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, alias).version
    except MlflowException:
        return None


def _history(client: MlflowClient, name: str) -> list[str]:
    tags = client.get_registered_model(name).tags
    return json.loads(tags.get(HISTORY_TAG, "[]"))


def register_bundle(bundle_dir: Path, gate_report_file: Path, metrics_file: Path, name: str) -> str:
    """Register a gated bundle and point `challenger` at it. Refuses failed or corrupt bundles."""
    report = GateReport.model_validate_json(gate_report_file.read_text())
    if not report.passed:
        raise RegistryError("gate failed — bundle not registered (see gate report)")
    problems = verify_bundle(bundle_dir)
    if problems:
        raise RegistryError(f"bundle integrity: {problems}")

    manifest = read_manifest(bundle_dir)
    metrics = json.loads(metrics_file.read_text())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"bundle-{manifest.bundle_version}"):
        mlflow.set_tags(
            {
                "bundle_id": manifest.bundle_id,
                "git_commit": manifest.git_commit or "unknown",
                "data_snapshot": manifest.data_snapshot or "unknown",
            }
        )
        mlflow.log_metrics(
            {
                "auto_rate": metrics["auto_rate"],
                "wrong_assignments": metrics["wrong_assignments"],
                "whole_record_accuracy": metrics["whole_record_accuracy"],
                "latency_p95_ms": metrics["latency_ms"]["p95"],
                **{f"acc_{f}": v["accuracy"] for f, v in metrics["fields"].items()},
            }
        )
        mlflow.log_artifact(str(gate_report_file), "gate")
        mlflow.log_artifact(str(metrics_file), "evaluation")
        info = mlflow.pyfunc.log_model(
            name="bundle",
            python_model=BundleModel(),
            artifacts={"bundle": str(bundle_dir)},
            registered_model_name=name,
            pip_requirements=[],  # runtime deps come from the docker image, not pip inference
        )

    version = str(info.registered_model_version)
    client = MlflowClient()
    for key, value in {
        "bundle_id": manifest.bundle_id,
        "bundle_version": manifest.bundle_version,
        "gate_passed": "true",
        "eval_set_hash": str(metrics.get("eval_set_hash")),
        "registered_at": _now(),
    }.items():
        client.set_model_version_tag(name, version, key, value)
    client.set_registered_model_alias(name, CHALLENGER, version)
    return version


def promote(name: str, version: str, approved_by: str, note: str = "") -> dict:
    """Make `version` the champion. Requires a named approver (supervised promotion)."""
    if not approved_by.strip():
        raise RegistryError("promotion needs --approved-by")
    client = MlflowClient()
    mv = client.get_model_version(name, version)
    if mv.tags.get("gate_passed") != "true":
        raise RegistryError(f"version {version} has no passing gate report")

    current = _alias_version(client, name, CHAMPION)
    if current == version:
        raise RegistryError(f"version {version} is already champion")
    if current is not None:
        client.set_registered_model_alias(name, PREVIOUS, current)
    client.set_registered_model_alias(name, CHAMPION, version)
    if _alias_version(client, name, CHALLENGER) == version:
        client.delete_registered_model_alias(name, CHALLENGER)

    history = _history(client, name) + [version]
    client.set_registered_model_tag(name, HISTORY_TAG, json.dumps(history))
    for key, value in {
        "promoted_at": _now(),
        "approved_by": approved_by,
        "promotion_note": note,
    }.items():
        client.set_model_version_tag(name, version, key, value)
    return {"champion": version, "previous": current}


def rollback(name: str, reason: str, by: str) -> dict:
    """Point `champion` back at the previous champion. Instant: serving reloads the alias."""
    if not reason.strip() or not by.strip():
        raise RegistryError("rollback needs --reason and --by")
    client = MlflowClient()
    history = _history(client, name)
    if len(history) < 2:
        raise RegistryError("no earlier champion to roll back to")

    bad, target = history[-1], history[-2]
    client.set_registered_model_alias(name, CHAMPION, target)
    if len(history) >= 3:
        client.set_registered_model_alias(name, PREVIOUS, history[-3])
    else:
        client.delete_registered_model_alias(name, PREVIOUS)
    client.set_registered_model_tag(name, HISTORY_TAG, json.dumps(history[:-1]))
    for key, value in {
        "rolled_back_at": _now(),
        "rolled_back_by": by,
        "rollback_reason": reason,
    }.items():
        client.set_model_version_tag(name, bad, key, value)
    return {"champion": target, "rolled_back": bad}


def status(name: str) -> dict:
    client = MlflowClient()
    try:
        client.get_registered_model(name)
    except MlflowException:
        return {"registered_model": name, "exists": False}
    out: dict = {"registered_model": name, "exists": True, "history": _history(client, name)}
    for alias in (CHAMPION, CHALLENGER, PREVIOUS):
        v = _alias_version(client, name, alias)
        out[alias] = None if v is None else {"version": v, **client.get_model_version(name, v).tags}
    return out


def download_bundle(name: str, alias: str, dst: Path) -> Path | None:
    """Fetch the bundle directory behind an alias (e.g. to re-evaluate the champion)."""
    version = _alias_version(MlflowClient(), name, alias)
    if version is None:
        return None
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{version}", dst_path=str(dst)
    )
    return next(Path(local).rglob("bundle.json")).parent
