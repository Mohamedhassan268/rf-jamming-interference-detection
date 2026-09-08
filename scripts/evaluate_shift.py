"""Evaluate an untouched source-trained model on a labeled capture session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.data.captures import MappedCapturedRFDataset
from src.data.pipeline import TransformedSubset, collate_iq
from src.evaluation import classification_metrics, expected_calibration_error, metrics_per_snr
from src.models import CompactRFNet
from src.utils.config import ConfigError, load_config, require_resolved
from src.utils.logging import append_result, git_commit, utc_timestamp


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-id", default="baseline_target_test")
    parser.add_argument("--results-csv", default="experiments/results.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        require_resolved(config, ["evaluation.capture_session"])
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise ConfigError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        training_config = checkpoint["config"]
        class_names = checkpoint["class_names"]
        capture_path = Path(config["dataset"]["target_path"]) / config["evaluation"]["capture_session"]
        captured = MappedCapturedRFDataset(capture_path, class_names)
        first, _, _ = captured[0]
        expected_length = int(training_config["dataset"]["input_length"])
        if tuple(first.shape) != (2, expected_length):
            raise ConfigError(
                f"Capture shape {tuple(first.shape)} does not match trained input [2, {expected_length}]. "
                "Cropping, padding, or resampling must be explicitly configured and justified."
            )
        target_data = TransformedSubset(
            captured,
            np.arange(len(captured)),
            training_config.get("preprocessing", {}),
            augment=False,
            seed=int(training_config["seed"]),
        )
        loader = DataLoader(
            target_data,
            batch_size=int(training_config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(training_config["training"].get("num_workers", 0)),
            collate_fn=collate_iq,
        )
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ConfigError("CUDA was requested but is not available.")
        model = _model(training_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logits_parts, truth_parts, snrs = [], [], []
        with torch.no_grad():
            for samples, labels, metadata in loader:
                logits_parts.append(model(samples.to(device=device, dtype=torch.float32)).cpu().numpy())
                truth_parts.append(labels.numpy())
                snrs.extend(item.get("snr_db") for item in metadata)
        logits = np.concatenate(logits_parts)
        truth = np.concatenate(truth_parts)
        predicted = logits.argmax(axis=1)
        metrics = classification_metrics(
            truth, predicted, labels=np.arange(len(class_names)), class_names=class_names
        )
        metrics["ece"] = expected_calibration_error(logits, truth)
        metrics["per_snr"] = metrics_per_snr(truth, predicted, snrs) if any(value is not None for value in snrs) else None
        output_path = checkpoint_path.parent / "target_test_metrics.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        display = ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=class_names)
        display.plot(xticks_rotation=45, colorbar=False)
        plt.tight_layout()
        figure_path = checkpoint_path.parent / "confusion_matrix_target.png"
        plt.savefig(figure_path, dpi=160)
        plt.close()
        preprocessing = training_config.get("preprocessing", {})
        enabled = [
            name
            for name in ("amplitude_scale", "phase_rotation", "awgn", "frequency_offset")
            if preprocessing.get(name, {}).get("enabled", False)
        ]
        append_result(
            args.results_csv,
            {
                "experiment_id": args.experiment_id,
                "date": utc_timestamp(),
                "git_commit": git_commit() or "",
                "seed": training_config["seed"],
                "model": training_config["model"]["name"],
                "parameter_count": checkpoint["parameter_count"],
                "dataset": training_config["dataset"]["name"],
                "task_type": training_config["dataset"]["task_type"],
                "train_classes": "|".join(class_names),
                "test_domain": f"captured_rf:{config['evaluation']['capture_session']}",
                "normalization": preprocessing.get("normalization", "none"),
                "augmentation": "+".join(enabled) if enabled else "none",
                "learning_rate": training_config["training"]["learning_rate"],
                "batch_size": training_config["training"]["batch_size"],
                "best_epoch": checkpoint["best_epoch"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "ece": metrics["ece"],
                "notes": "Untouched source-trained model on labeled captured/shifted RF",
            },
        )
        print(f"Saved target metrics to {output_path} and confusion matrix to {figure_path}.")
    except (ConfigError, FileNotFoundError, KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
