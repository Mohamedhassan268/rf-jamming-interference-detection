"""Append-only experiment metadata and run artifact helpers."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

RESULT_COLUMNS = [
    "experiment_id", "date", "git_commit", "seed", "model", "parameter_count",
    "dataset", "task_type", "train_classes", "test_domain", "normalization",
    "augmentation", "learning_rate", "batch_size", "best_epoch", "accuracy",
    "macro_f1", "weighted_f1", "ece", "notes",
]


def git_commit() -> str | None:
    """Return the current commit hash, or `None` outside a Git work tree."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def create_run_directory(root: str | Path, experiment_id: str) -> Path:
    """Create a non-overwriting timestamped run directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / f"{timestamp}_{experiment_id}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_run_metadata(run_dir: str | Path, config: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Save resolved config and JSON-safe experiment metadata."""
    root = Path(run_dir)
    with (root / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), **metadata}
    with (root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_result(path: str | Path, row: dict[str, Any]) -> None:
    """Append one result without truncating prior experiments."""
    unknown = set(row) - set(RESULT_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown results columns: {sorted(unknown)}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for result records."""
    return datetime.now(timezone.utc).isoformat()
