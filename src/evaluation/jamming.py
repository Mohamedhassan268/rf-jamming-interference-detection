"""Task-specific metrics for binary synthetic-jamming detection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _rate(predicted: np.ndarray, mask: np.ndarray, positive_label: int) -> dict[str, float | int]:
    return {
        "rate": float(np.mean(predicted[mask] == positive_label)),
        "support": int(mask.sum()),
    }


def binary_jamming_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metadata: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Report clean false alarms and jammed detection rates by generation condition."""
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if truth.ndim != 1 or truth.shape != predicted.shape or len(metadata) != len(truth):
        raise ValueError("Labels, predictions, and metadata must describe the same 1-D examples.")
    if set(np.unique(truth)) - {0, 1} or set(np.unique(predicted)) - {0, 1}:
        raise ValueError("Binary jamming metrics require labels and predictions in {0, 1}.")
    clean_mask = truth == 0
    jammed_mask = truth == 1
    if not clean_mask.any() or not jammed_mask.any():
        raise ValueError("Binary jamming metrics require both clean and jammed examples.")

    jammer_types = np.asarray([item.get("jammer_type") for item in metadata], dtype=object)
    jsr_values = np.asarray([item.get("jsr_db_requested") for item in metadata], dtype=object)
    if any(value is None for value in jammer_types[jammed_mask]):
        raise ValueError("Every jammed example must record jammer_type.")
    if any(value is None for value in jsr_values[jammed_mask]):
        raise ValueError("Every jammed example must record jsr_db_requested.")

    by_jammer = {}
    for jammer_type in sorted(set(jammer_types[jammed_mask])):
        mask = jammed_mask & (jammer_types == jammer_type)
        by_jammer[str(jammer_type)] = _rate(predicted, mask, positive_label=1)

    numeric_jsr = np.asarray(
        [np.nan if value is None else float(value) for value in jsr_values], dtype=np.float64
    )
    by_jsr_db = {}
    by_jammer_and_jsr_db = {}
    for jsr_db in sorted(set(numeric_jsr[jammed_mask])):
        jsr_mask = jammed_mask & np.isclose(numeric_jsr, jsr_db)
        by_jsr_db[f"{jsr_db:g}"] = _rate(predicted, jsr_mask, positive_label=1)
    for jammer_type in sorted(set(jammer_types[jammed_mask])):
        type_mask = jammed_mask & (jammer_types == jammer_type)
        by_jammer_and_jsr_db[str(jammer_type)] = {
            f"{jsr_db:g}": _rate(
                predicted,
                type_mask & np.isclose(numeric_jsr, jsr_db),
                positive_label=1,
            )
            for jsr_db in sorted(set(numeric_jsr[type_mask]))
        }

    return {
        "clean_false_positive_rate": _rate(predicted, clean_mask, positive_label=1),
        "jammed_detection_rate": _rate(predicted, jammed_mask, positive_label=1),
        "detection_rate_by_jammer": by_jammer,
        "detection_rate_by_jsr_db": by_jsr_db,
        "detection_rate_by_jammer_and_jsr_db": by_jammer_and_jsr_db,
    }
