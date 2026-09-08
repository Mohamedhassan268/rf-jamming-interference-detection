"""RadioML 2018.01A HDF5 adapter with explicit schema and shape validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RadioMLSchema:
    samples_key: str
    labels_key: str
    snr_key: str | None = None


def inspect_radioml_hdf5(path: str | Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return every HDF5 dataset's shape and dtype without loading arrays."""
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    with h5py.File(path, "r") as handle:
        def visitor(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset):
                result[name] = (tuple(item.shape), str(item.dtype))
        handle.visititems(visitor)
    return result


def _validate_sample_shape(shape: Sequence[int]) -> str:
    if len(shape) != 3:
        raise ValueError(f"RadioML samples must be 3-D, got {tuple(shape)}.")
    if shape[1] == 2:
        return "N2T"
    if shape[2] == 2:
        return "NT2"
    raise ValueError(f"Cannot identify I/Q axis in sample shape {tuple(shape)}.")


class RadioML2018Dataset(Dataset):
    """Lazy, filtered RadioML adapter returning `[2, T]` plus source metadata.

    The caller supplies explicit dataset keys because source copies may differ.
    Labels may be integer IDs or one-hot rows; semantic class names must be
    supplied when filtering one-hot labels. No class meaning is inferred.
    """

    def __init__(
        self,
        path: str | Path,
        schema: RadioMLSchema,
        classes: Sequence[int | str] | None = None,
        class_names: Sequence[str] | None = None,
        snr_min: float | None = None,
        snr_max: float | None = None,
        max_examples_per_class: int | None = None,
        seed: int = 42,
    ) -> None:
        self.path = Path(path)
        self.schema = schema
        if not self.path.is_file():
            raise FileNotFoundError(f"RadioML file not found: {self.path}")
        with h5py.File(self.path, "r") as handle:
            for key in (schema.samples_key, schema.labels_key):
                if key not in handle:
                    raise KeyError(f"HDF5 dataset {key!r} not found. Available: {list(handle.keys())}")
            self.layout = _validate_sample_shape(handle[schema.samples_key].shape)
            raw_labels = np.asarray(handle[schema.labels_key])
            if len(raw_labels) != handle[schema.samples_key].shape[0]:
                raise ValueError("RadioML sample and label counts differ.")
            label_ids = raw_labels.argmax(axis=1) if raw_labels.ndim == 2 else raw_labels.reshape(-1)
            if raw_labels.ndim not in (1, 2):
                raise ValueError(f"Unsupported label shape: {raw_labels.shape}.")
            self.class_names = list(class_names) if class_names is not None else None
            if self.class_names is not None and max(label_ids, default=-1) >= len(self.class_names):
                raise ValueError("class_names does not cover all numeric labels.")
            indices = np.arange(len(label_ids))
            if classes is not None:
                wanted_ids: list[int] = []
                for value in classes:
                    if isinstance(value, str):
                        if self.class_names is None:
                            raise ValueError("class_names is required for string class filtering.")
                        wanted_ids.append(self.class_names.index(value))
                    else:
                        wanted_ids.append(int(value))
                indices = indices[np.isin(label_ids, wanted_ids)]
            self.snrs: np.ndarray | None = None
            if schema.snr_key is not None:
                if schema.snr_key not in handle:
                    raise KeyError(f"SNR dataset {schema.snr_key!r} was requested but not found.")
                self.snrs = np.asarray(handle[schema.snr_key]).reshape(-1)
                if len(self.snrs) != len(label_ids):
                    raise ValueError("RadioML sample and SNR counts differ.")
                mask = np.ones(len(indices), dtype=bool)
                if snr_min is not None:
                    mask &= self.snrs[indices] >= snr_min
                if snr_max is not None:
                    mask &= self.snrs[indices] <= snr_max
                indices = indices[mask]
            elif snr_min is not None or snr_max is not None:
                raise ValueError("SNR filtering requested without an snr_key.")
            if max_examples_per_class is not None:
                if max_examples_per_class <= 0:
                    raise ValueError("max_examples_per_class must be positive.")
                rng = np.random.default_rng(seed)
                selected = []
                for class_id in np.unique(label_ids[indices]):
                    candidates = indices[label_ids[indices] == class_id]
                    selected.extend(rng.choice(candidates, min(len(candidates), max_examples_per_class), replace=False))
                indices = np.asarray(sorted(selected), dtype=np.int64)
            self.indices = indices.astype(np.int64)
            self.labels = label_ids.astype(np.int64)
            self.selected_class_ids = np.unique(self.labels[self.indices]).astype(np.int64)
            self.label_to_local = {
                int(source_id): local_id for local_id, source_id in enumerate(self.selected_class_ids)
            }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        source_index = int(self.indices[index])
        with h5py.File(self.path, "r") as handle:
            sample = np.asarray(handle[self.schema.samples_key][source_index])
        if self.layout == "NT2":
            sample = sample.T
        if sample.ndim != 2 or sample.shape[0] != 2:
            raise ValueError(f"Invalid sample at index {source_index}: {sample.shape}.")
        source_label_id = int(self.labels[source_index])
        label_id = self.label_to_local[source_label_id]
        metadata: dict[str, Any] = {
            "sample_id": source_index,
            "source_dataset": "RadioML 2018.01A",
            "label_id": label_id,
            "source_label_id": source_label_id,
            "class_name": self.class_names[source_label_id] if self.class_names else None,
            "snr_db": float(self.snrs[source_index]) if self.snrs is not None else None,
        }
        return torch.from_numpy(sample), label_id, metadata

    @property
    def selected_labels(self) -> np.ndarray:
        """Return contiguous training labels aligned with this filtered dataset."""
        return np.asarray([self.label_to_local[int(value)] for value in self.labels[self.indices]], dtype=np.int64)

    @property
    def selected_class_names(self) -> list[str]:
        """Return names in contiguous classifier-output order."""
        if self.class_names is None:
            return [str(value) for value in self.selected_class_ids]
        return [self.class_names[int(value)] for value in self.selected_class_ids]

