"""Per-SNR performance summaries when SNR metadata is available."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .metrics import classification_metrics


def metrics_per_snr(
    y_true: Sequence[int], y_pred: Sequence[int], snr_db: Sequence[float | None]
) -> dict[float, dict[str, Any]]:
    """Compute classification metrics for each observed finite SNR value."""
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    if len(truth) != len(predicted) or len(truth) != len(snr_db):
        raise ValueError("Labels, predictions, and SNR metadata must have equal length.")
    snr = np.asarray([np.nan if value is None else value for value in snr_db], dtype=float)
    labels = np.unique(np.concatenate((truth, predicted)))
    output: dict[float, dict[str, Any]] = {}
    for value in np.unique(snr[np.isfinite(snr)]):
        mask = snr == value
        output[float(value)] = classification_metrics(truth[mask], predicted[mask], labels=labels)
    return output
