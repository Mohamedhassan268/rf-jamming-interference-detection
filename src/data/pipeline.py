"""Configuration-driven datasets and I/Q transformations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import ConfigError, require_resolved

from .preprocessing import (
    PreprocessingConfig,
    add_awgn,
    amplitude_scale,
    frequency_offset,
    phase_rotate,
    preprocess,
)
from .radioml import RadioML2018Dataset, RadioMLSchema


def _is_unresolved(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().upper() == "TBD")


def resolve_dataset_path(config: dict[str, Any]) -> Path:
    """Resolve the configured RadioML file without guessing among candidates."""
    require_resolved(config, ["dataset.path", "dataset.file"])
    dataset = config["dataset"]
    filename = Path(str(dataset["file"]))
    return filename if filename.is_absolute() else Path(dataset["path"]) / filename


def build_radioml_dataset(config: dict[str, Any]) -> RadioML2018Dataset:
    """Build a filtered RadioML adapter from verified configuration fields."""
    require_resolved(
        config,
        [
            "dataset.task_type",
            "dataset.classes",
            "dataset.input_length",
            "dataset.schema.samples_key",
            "dataset.schema.labels_key",
            "dataset.schema.class_names",
        ],
    )
    dataset_config = config["dataset"]
    schema_config = dataset_config["schema"]
    snr_key = schema_config.get("snr_key")
    if _is_unresolved(snr_key):
        snr_key = None
    snr_min = dataset_config.get("snr_min")
    snr_max = dataset_config.get("snr_max")
    dataset = RadioML2018Dataset(
        resolve_dataset_path(config),
        RadioMLSchema(schema_config["samples_key"], schema_config["labels_key"], snr_key),
        classes=dataset_config["classes"],
        class_names=schema_config["class_names"],
        snr_min=None if _is_unresolved(snr_min) else float(snr_min),
        snr_max=None if _is_unresolved(snr_max) else float(snr_max),
        max_examples_per_class=dataset_config.get("max_examples_per_class"),
        seed=int(dataset_config.get("sample_seed", config["seed"])),
    )
    if not len(dataset):
        raise ConfigError("The configured RadioML filters selected zero samples.")
    first, _, _ = dataset[0]
    expected = int(dataset_config["input_length"])
    if tuple(first.shape) != (2, expected):
        raise ConfigError(
            f"Configured input_length={expected}, but selected samples have shape {tuple(first.shape)}."
        )
    return dataset


class TransformedSubset(Dataset):
    """Index subset applying deterministic preprocessing and optional augmentation."""

    def __init__(
        self,
        dataset: Dataset,
        indices: np.ndarray,
        preprocessing_config: dict[str, Any],
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.config = preprocessing_config
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample, label, metadata = self.dataset[int(self.indices[index])]
        array = sample.numpy()
        array = preprocess(
            array,
            PreprocessingConfig(
                normalization=self.config.get("normalization", "none"),
                remove_dc=bool(self.config.get("remove_dc", False)),
            ),
        )
        if self.augment:
            array = self._augment(array)
        return torch.from_numpy(np.ascontiguousarray(array)), label, metadata

    def _augment(self, array: np.ndarray) -> np.ndarray:
        amplitude = self.config.get("amplitude_scale", {})
        if amplitude.get("enabled", False):
            array = amplitude_scale(array, self.rng.uniform(float(amplitude["min"]), float(amplitude["max"])))
        phase = self.config.get("phase_rotation", {})
        if phase.get("enabled", False):
            array = phase_rotate(array, self.rng.uniform(-np.pi, np.pi))
        awgn = self.config.get("awgn", {})
        if awgn.get("enabled", False):
            for field in ("snr_db_min", "snr_db_max"):
                if _is_unresolved(awgn.get(field)):
                    raise ConfigError(f"preprocessing.awgn.{field} must be resolved when AWGN is enabled.")
            array = add_awgn(
                array,
                self.rng.uniform(float(awgn["snr_db_min"]), float(awgn["snr_db_max"])),
                self.rng,
            )
        offset = self.config.get("frequency_offset", {})
        if offset.get("enabled", False):
            if _is_unresolved(offset.get("max_normalized_offset")):
                raise ConfigError(
                    "preprocessing.frequency_offset.max_normalized_offset must be resolved when enabled."
                )
            maximum = float(offset["max_normalized_offset"])
            array = frequency_offset(array, self.rng.uniform(-maximum, maximum))
        return array


def collate_iq(batch):
    """Stack tensors and labels while preserving heterogeneous metadata dictionaries."""
    samples, labels, metadata = zip(*batch)
    return torch.stack(samples), torch.as_tensor(labels, dtype=torch.long), list(metadata)
