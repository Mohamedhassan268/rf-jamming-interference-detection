"""Reproducible split helpers with optional group isolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def radioml_strata(dataset) -> np.ndarray:
    """Return joint modulation/source-SNR strata aligned with a RadioML adapter."""
    modulation_ids = dataset.selected_labels
    if dataset.snrs is None:
        return modulation_ids
    selected_snrs = np.asarray(dataset.snrs[dataset.indices])
    _, snr_ids = np.unique(selected_snrs, return_inverse=True)
    return modulation_ids * (snr_ids.max() + 1) + snr_ids


def make_splits(
    labels: np.ndarray,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> SplitIndices:
    """Create deterministic splits, isolating parent/session groups if supplied."""
    total = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0) or min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Split fractions must be positive and sum to 1.")
    y = np.asarray(labels)
    indices = np.arange(len(y))
    if groups is not None:
        group_array = np.asarray(groups)
        first = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
        train, remainder = next(first.split(indices, y, group_array))
        relative_test = test_fraction / (validation_fraction + test_fraction)
        second = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
        val_local, test_local = next(second.split(remainder, y[remainder], group_array[remainder]))
        return SplitIndices(indices[train], remainder[val_local], remainder[test_local])
    train, remainder = train_test_split(
        indices, train_size=train_fraction, random_state=seed, stratify=y
    )
    relative_test = test_fraction / (validation_fraction + test_fraction)
    validation, test = train_test_split(
        remainder, test_size=relative_test, random_state=seed, stratify=y[remainder]
    )
    return SplitIndices(train, validation, test)
