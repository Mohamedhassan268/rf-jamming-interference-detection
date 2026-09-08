"""Honest held-out classification summaries."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute aggregate, confusion-matrix, and per-class metrics."""
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal shape.")
    label_values = np.asarray(labels if labels is not None else np.unique(np.concatenate((truth, predicted))))
    if class_names is not None and len(class_names) != len(label_values):
        raise ValueError("class_names length must match labels length.")
    macro = precision_recall_fscore_support(truth, predicted, labels=label_values, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(truth, predicted, labels=label_values, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(truth, predicted, labels=label_values, average=None, zero_division=0)
    names = class_names if class_names is not None else [str(value) for value in label_values]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=label_values).tolist(),
        "per_class": {
            name: {
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
            for index, name in enumerate(names)
        },
    }

