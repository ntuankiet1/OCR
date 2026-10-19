"""The bundle: the unit we version, gate, promote and roll back.

A bundle is a directory holding every model, calibrator, threshold and rule the inference
pipeline needs — nothing is downloaded at runtime (BRD §9: test with internet disabled).
`bundle_id` is a content hash, stamped on every registration job for audit (BR16).

  bundle/
    bundle.json            BundleManifest
    models/<component>/    det, rec, doc_orientation, textline_orientation, type_size
    calibration.json       per-field calibrators + thresholds
    pipeline.json          model names + runtime params (quality, matching)
    rules/                 carriers.yaml, type_size.yaml (copied from configs/)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.text.normalize import NORMALIZATION_VERSION

MANIFEST = "bundle.json"


def hash_path(path: Path) -> str:
    """Stable SHA-256 over a file or a directory tree (relative paths + contents)."""
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        h.update(str(f.relative_to(path) if path.is_dir() else f.name).encode())
        h.update(b"\0")
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout
        return out + ("-dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class ComponentRef(BaseModel):
    name: str
    path: str  # relative to the bundle root
    sha256: str
    framework: str  # paddle-inference | onnx
    source: str  # "fine-tuned:<run id>" or "pretrained:<model name>"


class BundleManifest(BaseModel):
    bundle_id: str
    bundle_version: str  # human-readable: <date>-<bundle_id[:8]>
    created_at: str
    git_commit: str | None
    data_snapshot: str | None  # snapshot hash (dvc.lock) the challenger was trained on
    normalization_version: str = NORMALIZATION_VERSION
    components: list[ComponentRef]
    rules_sha256: dict[str, str]
    calibration_sha256: str
    thresholds: dict[str, float | None] = Field(default_factory=dict)

    def component(self, name: str) -> ComponentRef:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"bundle has no component {name!r}")


def assemble_bundle(
    out_dir: Path,
    components: dict[str, tuple[Path, str, str]],  # name -> (src path, framework, source)
    calibration_file: Path,
    rule_files: list[Path],
    data_snapshot: str | None,
    pipeline_meta: dict,  # model names + runtime params (quality, matching) the pipeline reads
) -> BundleManifest:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "models").mkdir(parents=True)
    (out_dir / "rules").mkdir()

    refs = []
    for name, (src, framework, source) in sorted(components.items()):
        dst = out_dir / "models" / name
        shutil.copytree(src, dst) if src.is_dir() else (
            dst.mkdir(),
            shutil.copy2(src, dst / src.name),
        )
        refs.append(
            ComponentRef(
                name=name,
                path=f"models/{name}",
                sha256=hash_path(dst),
                framework=framework,
                source=source,
            )
        )

    rules = {}
    for f in rule_files:
        shutil.copy2(f, out_dir / "rules" / f.name)
        rules[f.name] = hash_path(out_dir / "rules" / f.name)

    shutil.copy2(calibration_file, out_dir / "calibration.json")
    calibration = json.loads(calibration_file.read_text())
    thresholds = {f: c.get("threshold") for f, c in calibration.get("fields", {}).items()}
    (out_dir / "pipeline.json").write_text(json.dumps(pipeline_meta, indent=2, sort_keys=True))
    rules["pipeline.json"] = hash_path(out_dir / "pipeline.json")

    identity = json.dumps(
        {
            "components": {r.name: r.sha256 for r in refs},
            "rules": rules,
            "calibration": hash_path(out_dir / "calibration.json"),
            "normalization": NORMALIZATION_VERSION,
        },
        sort_keys=True,
    )
    bundle_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    now = datetime.now(UTC)
    manifest = BundleManifest(
        bundle_id=bundle_id,
        bundle_version=f"{now:%Y.%m.%d}-{bundle_id[:8]}",
        created_at=now.isoformat(timespec="seconds"),
        git_commit=git_commit(),
        data_snapshot=data_snapshot,
        components=refs,
        rules_sha256=rules,
        calibration_sha256=hash_path(out_dir / "calibration.json"),
        thresholds=thresholds,
    )
    write_manifest(out_dir, manifest)
    return manifest


def write_manifest(bundle_dir: Path, manifest: BundleManifest) -> None:
    (bundle_dir / MANIFEST).write_text(manifest.model_dump_json(indent=2))


def read_manifest(bundle_dir: Path) -> BundleManifest:
    return BundleManifest.model_validate_json((bundle_dir / MANIFEST).read_text())


def verify_bundle(bundle_dir: Path) -> list[str]:
    """Return a list of integrity problems (empty = OK). Run before serving a bundle."""
    m = read_manifest(bundle_dir)
    problems = []
    for c in m.components:
        p = bundle_dir / c.path
        if not p.exists():
            problems.append(f"missing component {c.name}")
        elif hash_path(p) != c.sha256:
            problems.append(f"component {c.name} hash mismatch")
    if hash_path(bundle_dir / "calibration.json") != m.calibration_sha256:
        problems.append("calibration.json hash mismatch")
    for name, sha in m.rules_sha256.items():
        p = bundle_dir / name if name == "pipeline.json" else bundle_dir / "rules" / name
        if not p.exists() or hash_path(p) != sha:
            problems.append(f"rule file {name} missing or modified")
    return problems
