"""Split assignment, frozen evaluation sets and leakage guards.

* Frozen sets (acceptance, regression) are plain-text lists of group keys under data/frozen/,
  tracked by git. They are created once and never edited in place — a new version is a new file.
* Everything else is assigned by a salted hash of group_key, so a parcel keeps its split across
  snapshots and retakes of one parcel can never straddle train and test.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.schemas import FROZEN_SPLITS, Split


class LeakageError(RuntimeError):
    pass


class ForbiddenSplitError(RuntimeError):
    pass


def hash_unit(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def frozen_path(frozen_dir: Path, split: Split) -> Path:
    return frozen_dir / f"{split.value}.txt"


def load_frozen(frozen_dir: Path) -> dict[Split, set[str]]:
    sets: dict[Split, set[str]] = {}
    for split in FROZEN_SPLITS:
        path = frozen_path(frozen_dir, split)
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            sets[split] = {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}
        else:
            sets[split] = set()
    overlap = sets[Split.REGRESSION] & sets[Split.ACCEPTANCE]
    if overlap:
        raise LeakageError(f"{len(overlap)} group keys are in both frozen sets")
    return sets


def frozen_set_hash(frozen_dir: Path, split: Split) -> str:
    """Identity of a frozen set version; stored with every evaluation so the gate can refuse
    to compare results computed on different set versions."""
    keys = sorted(load_frozen(frozen_dir)[split])
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def write_frozen(frozen_dir: Path, split: Split, group_keys: list[str], note: str) -> Path:
    if split not in FROZEN_SPLITS:
        raise ValueError(f"{split} is not a frozen split")
    path = frozen_path(frozen_dir, split)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Frozen sets are immutable: archive it (git keeps history) "
            "and re-evaluate the champion on the new version before gating against it."
        )
    other = load_frozen(frozen_dir)[
        Split.ACCEPTANCE if split == Split.REGRESSION else Split.REGRESSION
    ]
    clash = set(group_keys) & other
    if clash:
        raise LeakageError(f"{len(clash)} keys already belong to the other frozen set")
    frozen_dir.mkdir(parents=True, exist_ok=True)
    header = [
        f"# frozen split: {split.value}",
        f"# created_at: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"# n_groups: {len(group_keys)}",
        f"# note: {note}",
    ]
    path.write_text("\n".join(header + sorted(group_keys)) + "\n", encoding="utf-8")
    return path


def assign_splits(
    images: pd.DataFrame,
    frozen: dict[Split, set[str]],
    salt: str,
    val_fraction: float,
    calibration_fraction: float,
) -> pd.Series:
    """Return the split for each row of `images` (needs a `group_key` column)."""

    def pick(key: str) -> str:
        for split in FROZEN_SPLITS:
            if key in frozen.get(split, ()):
                return split.value
        if key.startswith("synthetic:"):
            return Split.TRAIN.value
        u = hash_unit(key, salt)
        if u < calibration_fraction:
            return Split.CALIBRATION.value
        if u < calibration_fraction + val_fraction:
            return Split.VAL.value
        return Split.TRAIN.value

    return images["group_key"].map(pick)


def check_no_leakage(images: pd.DataFrame) -> None:
    per_group = images.groupby("group_key")["split"].nunique()
    leaked = per_group[per_group > 1]
    if not leaked.empty:
        raise LeakageError(
            f"{len(leaked)} groups span several splits, e.g. {leaked.index[:3].tolist()}"
        )


def guard_training_splits(splits: list[str]) -> None:
    """Training/export code calls this before touching data: frozen sets never leave the vault."""
    bad = [s for s in splits if s in {f.value for f in FROZEN_SPLITS}]
    if bad:
        raise ForbiddenSplitError(f"refusing to export frozen split(s) for training: {bad}")
