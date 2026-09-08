"""Basic closed-set confidence and calibration utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("logits must have shape [N, C] with C >= 2.")
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def maximum_softmax_confidence(logits: np.ndarray) -> np.ndarray:
    """Return maximum class probability for each sample."""
    return _softmax(logits).max(axis=1)


def reliability_bins(logits: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> list[dict[str, Any]]:
    """Build equal-width reliability-bin data using `(lower, upper]` bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    probabilities = _softmax(logits)
    truth = np.asarray(y_true)
    if truth.ndim != 1 or len(truth) != len(probabilities):
        raise ValueError("y_true must be one-dimensional and match logits rows.")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for index in range(n_bins):
        mask = (confidence >= edges[index]) & (confidence <= edges[index + 1] if index == n_bins - 1 else confidence < edges[index + 1])
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    return bins


def expected_calibration_error(logits: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Compute sample-weighted expected calibration error."""
    bins = reliability_bins(logits, y_true, n_bins=n_bins)
    total = sum(item["count"] for item in bins)
    if total == 0:
        raise ValueError("Cannot compute ECE for zero samples.")
    return float(sum(item["count"] / total * abs(item["accuracy"] - item["mean_confidence"]) for item in bins if item["count"]))

