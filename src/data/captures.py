"""Strict adapter for session-based captured or shifted RF arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CaptureSession:
    samples: np.ndarray
    labels: pd.DataFrame
    metadata: dict[str, Any]


def _to_n2t(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 3:
        raise ValueError(f"Expected a 3-D capture array, got {samples.shape}.")
    if samples.shape[1] == 2:
        return samples
    if samples.shape[2] == 2:
        return np.transpose(samples, (0, 2, 1))
    raise ValueError(f"Expected [N, 2, T] or [N, T, 2], got {samples.shape}.")


def load_capture_session(session_dir: str | Path, mmap_mode: str | None = "r") -> CaptureSession:
    """Load a session while preserving unknown metadata and label columns."""
    root = Path(session_dir)
    required = [root / "metadata.json", root / "samples.npy", root / "labels.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Capture session is incomplete; missing: " + ", ".join(missing))
    with required[0].open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    samples = _to_n2t(np.load(required[1], mmap_mode=mmap_mode, allow_pickle=False))
    if not np.issubdtype(samples.dtype, np.floating):
        raise TypeError(f"Capture samples must be floating-point, got {samples.dtype}.")
    labels = pd.read_csv(required[2])
    if "label" not in labels.columns:
        raise ValueError("labels.csv must contain a 'label' column.")
    if len(labels) != len(samples):
        raise ValueError(f"Sample/label count mismatch: {len(samples)} vs {len(labels)}.")
    return CaptureSession(samples=samples, labels=labels, metadata=metadata)


class CapturedRFDataset(Dataset):
    """PyTorch dataset returning `(sample, label, metadata)` per window."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session = load_capture_session(session_dir)

    def __len__(self) -> int:
        return len(self.session.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        row = self.session.labels.iloc[index].to_dict()
        metadata = {**self.session.metadata, **row, "sample_index": index}
        return torch.from_numpy(np.array(self.session.samples[index], copy=True)), row["label"], metadata


class MappedCapturedRFDataset(Dataset):
    """Map capture labels to a source-trained classifier's exact class order."""

    def __init__(self, session_dir: str | Path, class_names: list[str]) -> None:
        self.dataset = CapturedRFDataset(session_dir)
        self.class_names = list(class_names)
        self.label_to_id = {name: index for index, name in enumerate(self.class_names)}
        observed = set(self.dataset.session.labels["label"].astype(str))
        unknown = observed - set(self.label_to_id)
        if unknown:
            raise ValueError(
                "Capture labels do not exist in the checkpoint class mapping: "
                f"{sorted(unknown)}. Supply a verified mapping; labels are never guessed."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        sample, label, metadata = self.dataset[index]
        label_name = str(label)
        metadata["source_class_name"] = label_name
        return sample, self.label_to_id[label_name], metadata
