"""Audits and plots for the first reproducible binary-jamming baseline."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.jamming import PairedBinaryJammingDataset, complex_power


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_audit(dataset, splits, seed: int) -> dict[str, Any]:
    """Describe source splits and reject source-window leakage."""
    arrays = {
        "train": np.asarray(splits.train, dtype=np.int64),
        "validation": np.asarray(splits.validation, dtype=np.int64),
        "test": np.asarray(splits.test, dtype=np.int64),
    }
    overlap = {
        "train_validation": int(len(np.intersect1d(arrays["train"], arrays["validation"]))),
        "train_test": int(len(np.intersect1d(arrays["train"], arrays["test"]))),
        "validation_test": int(len(np.intersect1d(arrays["validation"], arrays["test"]))),
    }
    if any(overlap.values()):
        raise ValueError(f"Source-window leakage detected: {overlap}")
    if len(np.unique(np.concatenate(tuple(arrays.values())))) != len(dataset):
        raise ValueError("Source splits do not form a complete, unique partition.")

    reports: dict[str, Any] = {}
    selected_labels = dataset.selected_labels
    selected_snrs = None if dataset.snrs is None else np.asarray(dataset.snrs[dataset.indices])
    for name, indices in arrays.items():
        modulation = Counter(dataset.selected_class_names[int(value)] for value in selected_labels[indices])
        snr = Counter(float(value) for value in selected_snrs[indices]) if selected_snrs is not None else {}
        reports[name] = {
            "source_windows": int(len(indices)),
            "modulation_distribution": dict(sorted(modulation.items())),
            "source_snr_distribution": {f"{key:g}": value for key, value in sorted(snr.items())},
        }
    return {
        "seed": int(seed),
        "stratification": "RadioML modulation class + source SNR",
        "splits": reports,
        "pairwise_source_position_overlap": overlap,
        "leakage_check_passed": True,
    }


def generation_audit(
    source_dataset,
    paired_train: PairedBinaryJammingDataset,
    paired_validation: PairedBinaryJammingDataset,
) -> dict[str, Any]:
    """Check binary/condition balance, deterministic generation, and measured JSR."""
    condition_reports: dict[str, Any] = {}
    for name, paired in (("train", paired_train), ("validation", paired_validation)):
        counts = paired.condition_counts()
        values = list(counts.values())
        condition_reports[name] = {
            "generated_examples": len(paired),
            "clean_examples": len(paired) // 2,
            "jammed_examples": len(paired) // 2,
            "clean_fraction": 0.5,
            "jammed_fraction": 0.5,
            "jammer_jsr_counts": counts,
            "condition_count_range": [min(values), max(values)],
            "condition_balance_passed": max(values) - min(values) <= 1,
        }
        if max(values) - min(values) > 1:
            raise ValueError(f"{name} jammer conditions are not approximately balanced.")

    first, _, first_metadata = paired_train[1]
    repeat, _, repeat_metadata = paired_train[1]
    deterministic = bool(np.array_equal(first.numpy(), repeat.numpy()) and first_metadata == repeat_metadata)
    if not deterministic:
        raise ValueError("Synthetic jammer generation is not deterministic.")

    errors = []
    sample_count = min(30, len(paired_train) // 2)
    for source_offset in range(sample_count):
        source_position = int(paired_train.source_indices[source_offset])
        source, _, _ = source_dataset[source_position]
        mixed, _, metadata = paired_train[2 * source_offset + 1]
        interference = mixed.numpy().astype(np.float64) - source.numpy().astype(np.float64)
        measured = 10.0 * np.log10(complex_power(interference) / complex_power(source.numpy()))
        errors.append(abs(measured - float(metadata["jsr_db_requested"])))
    tolerance_db = 1e-4
    maximum_error = max(errors, default=0.0)
    if maximum_error > tolerance_db:
        raise ValueError(f"Measured JSR error {maximum_error:g} dB exceeds {tolerance_db:g} dB.")
    return {
        **condition_reports,
        "determinism": {"sample_generated_twice_identically": deterministic},
        "jsr_check": {
            "samples_checked": sample_count,
            "tolerance_db": tolerance_db,
            "maximum_absolute_error_db": maximum_error,
            "passed": True,
        },
    }


def plot_training_history(history: list[dict[str, float | int]], output: str | Path) -> None:
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["validation_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["validation_accuracy"], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Training accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
