"""Evaluate a saved model exactly once on its preserved RadioML test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.data.jamming import PairedBinaryJammingDataset
from src.data.pipeline import TransformedSubset, build_radioml_dataset, collate_iq
from src.evaluation import (
    binary_jamming_metrics,
    classification_metrics,
    expected_calibration_error,
    metrics_per_snr,
)
from src.models import CompactRFNet
from src.utils.config import ConfigError, load_config
from src.utils.logging import append_result, git_commit, utc_timestamp


def _fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _model(config: dict) -> CompactRFNet:
    model = config["model"]
    return CompactRFNet(
        num_classes=int(model["num_classes"]),
        input_length=int(config["dataset"]["input_length"]),
        conv_channels=model["conv_channels"],
        kernel_sizes=model["kernel_sizes"],
        pooling=model["pooling"],
        dropout=float(model["dropout"]),
        classifier_size=int(model["classifier_size"]),
    )


def _augmentation_summary(config: dict) -> str:
    preprocessing = config.get("preprocessing", {})
    names = ("amplitude_scale", "phase_rotation", "awgn", "frequency_offset")
    enabled = [name for name in names if preprocessing.get(name, {}).get("enabled", False)]
    return "+".join(enabled) if enabled else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-id", default="baseline_source_test")
    parser.add_argument("--results-csv", default="experiments/results.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise ConfigError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("config") != config:
            raise ConfigError(
                "The supplied configuration differs from the configuration embedded in the checkpoint. "
                "Evaluate with the exact resolved training config."
            )
        dataset = build_radioml_dataset(config)
        if _fingerprint(dataset.indices) != checkpoint.get("selected_source_indices_sha256"):
            raise ConfigError("Selected dataset indices differ from training; test identity is not preserved.")
        test_indices = np.asarray(checkpoint["splits"]["test"], dtype=np.int64)
        if len(test_indices) == 0 or test_indices.max() >= len(dataset):
            raise ConfigError("Checkpoint test indices are empty or incompatible with the dataset.")
        generated_test = PairedBinaryJammingDataset(dataset, test_indices, config["label_generation"])
        test_data = TransformedSubset(
            generated_test,
            np.arange(len(generated_test)),
            config.get("preprocessing", {}),
            augment=False,
            seed=int(config["seed"]),
        )
        loader = DataLoader(
            test_data,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["training"].get("num_workers", 0)),
            collate_fn=collate_iq,
        )
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ConfigError("CUDA was requested but is not available.")
        model = _model(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logits_parts, truth_parts, snrs, example_metadata = [], [], [], []
        with torch.no_grad():
            for samples, labels, metadata in loader:
                logits_parts.append(model(samples.to(device=device, dtype=torch.float32)).cpu().numpy())
                truth_parts.append(labels.numpy())
                snrs.extend(item.get("snr_db") for item in metadata)
                example_metadata.extend(metadata)
        logits = np.concatenate(logits_parts)
        truth = np.concatenate(truth_parts)
        predicted = logits.argmax(axis=1)
        class_names = checkpoint["class_names"]
        metrics = classification_metrics(
            truth, predicted, labels=np.arange(len(class_names)), class_names=class_names
        )
        metrics["ece"] = expected_calibration_error(logits, truth)
        metrics["per_snr"] = metrics_per_snr(truth, predicted, snrs) if any(value is not None for value in snrs) else None
        metrics["jamming"] = binary_jamming_metrics(truth, predicted, example_metadata)
        output_path = checkpoint_path.parent / "source_test_metrics.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        display = ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=class_names)
        display.plot(xticks_rotation=45, colorbar=False)
        plt.tight_layout()
        figure_path = checkpoint_path.parent / "confusion_matrix_source.png"
        plt.savefig(figure_path, dpi=160)
        plt.close()
        append_result(
            args.results_csv,
            {
                "experiment_id": args.experiment_id,
                "date": utc_timestamp(),
                "git_commit": git_commit() or "",
                "seed": config["seed"],
                "model": config["model"]["name"],
                "parameter_count": checkpoint["parameter_count"],
                "dataset": config["dataset"]["name"],
                "task_type": config["dataset"]["task_type"],
                "train_classes": "|".join(class_names),
                "test_domain": "RadioML held-out test",
                "normalization": config.get("preprocessing", {}).get("normalization", "none"),
                "augmentation": _augmentation_summary(config),
                "learning_rate": config["training"]["learning_rate"],
                "batch_size": config["training"]["batch_size"],
                "best_epoch": checkpoint["best_epoch"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "ece": metrics["ece"],
                "notes": "Held-out source test; appended by scripts/evaluate.py",
            },
        )
        print(f"Saved held-out metrics to {output_path} and confusion matrix to {figure_path}.")
    except (ConfigError, FileNotFoundError, KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
