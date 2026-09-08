"""Evaluate every configured jammer/JSR condition on preserved test sources only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.jamming import FullJammerStressDataset
from src.data.pipeline import TransformedSubset, build_radioml_dataset, collate_iq
from src.models import CompactRFNet, count_trainable_parameters
from src.utils.config import ConfigError, load_config


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


def _plot_jsr(summary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    display = {"tone": "tone", "chirp": "chirp", "barrage": "barrage noise"}
    for jammer_type, group in summary.groupby("jammer_type", sort=False):
        ordered = group.sort_values("jsr_db")
        axis.plot(
            ordered["jsr_db"], ordered["detection_rate"], marker="o", label=display[jammer_type]
        )
    axis.set(
        xlabel="JSR (dB)",
        ylabel="Detection rate",
        title="Jamming detection rate vs. jammer-to-signal ratio",
        xticks=sorted(summary["jsr_db"].unique()),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_snr(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("source_snr_db")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(ordered["source_snr_db"], ordered["detection_rate"], marker="o")
    axis.set(
        xlabel="Original RadioML source SNR (dB)",
        ylabel="Detection rate",
        title="Jamming detection rate vs. source-signal SNR",
        xticks=ordered["source_snr_db"].tolist(),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        checkpoint_path = Path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("config") != config:
            raise ConfigError("Configuration differs from the checkpoint training configuration.")
        dataset = build_radioml_dataset(config)
        if _fingerprint(dataset.indices) != checkpoint.get("selected_source_indices_sha256"):
            raise ConfigError("Selected dataset indices differ from training.")
        test_indices = np.asarray(checkpoint["splits"]["test"], dtype=np.int64)
        stress = FullJammerStressDataset(dataset, test_indices, config["label_generation"])
        transformed = TransformedSubset(
            stress, np.arange(len(stress)), config.get("preprocessing", {}), augment=False,
            seed=int(config["seed"]),
        )
        loader = DataLoader(
            transformed,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["training"].get("num_workers", 0)),
            collate_fn=collate_iq,
        )
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ConfigError("CUDA was requested but is not available.")
        model = _model(config).to(device)
        if count_trainable_parameters(model) != checkpoint["parameter_count"] or checkpoint["parameter_count"] != 240_962:
            raise ConfigError("Checkpoint/model parameter count is not the required 240,962.")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        rows = []
        with torch.no_grad():
            for samples, _labels, metadata in loader:
                logits = model(samples.to(device=device, dtype=torch.float32)).cpu().numpy()
                shifted = logits - logits.max(axis=1, keepdims=True)
                probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
                predicted = logits.argmax(axis=1)
                for index, item in enumerate(metadata):
                    rows.append(
                        {
                            "modulation": item["class_name"],
                            "source_snr_db": float(item["snr_db"]),
                            "jammer_type": item["jammer_type"],
                            "jsr_db": float(item["jsr_db_requested"]),
                            "detected": int(predicted[index] == 1),
                            "jam_probability": float(probabilities[index, 1]),
                        }
                    )
        frame = pd.DataFrame(rows)
        detailed = (
            frame.groupby(["modulation", "source_snr_db", "jammer_type", "jsr_db"], sort=True)
            .agg(
                num_samples=("detected", "size"),
                detection_rate=("detected", "mean"),
                mean_jam_probability=("jam_probability", "mean"),
            )
            .reset_index()
        )
        jsr_summary = (
            frame.groupby(["jammer_type", "jsr_db"], sort=True)
            .agg(
                num_samples=("detected", "size"),
                detection_rate=("detected", "mean"),
                mean_jam_probability=("jam_probability", "mean"),
            )
            .reset_index()
        )
        snr_summary = (
            frame.groupby("source_snr_db", sort=True)
            .agg(
                num_samples=("detected", "size"),
                detection_rate=("detected", "mean"),
                mean_jam_probability=("jam_probability", "mean"),
            )
            .reset_index()
        )
        run_dir = checkpoint_path.parent
        detailed.to_csv(run_dir / "jammer_jsr_results.csv", index=False)
        jsr_summary.to_csv(run_dir / "jammer_jsr_summary.csv", index=False)
        snr_summary.to_csv(run_dir / "source_snr_results.csv", index=False)
        _plot_jsr(jsr_summary, run_dir / "detection_rate_vs_jsr.png")
        _plot_snr(snr_summary, run_dir / "detection_rate_vs_source_snr.png")
        stress_metrics = {
            "test_source_windows": int(len(test_indices)),
            "jammed_variants": int(len(frame)),
            "overall_detection_rate": float(frame["detected"].mean()),
            "overall_mean_jam_probability": float(frame["jam_probability"].mean()),
            "condition_grid_size": int(len(stress.conditions)),
        }
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metrics["stress_test_metrics"] = stress_metrics
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(
            f"Saved {len(detailed):,} detailed groups from {len(frame):,} stress variants "
            f"to {run_dir}."
        )
    except (ConfigError, FileNotFoundError, KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
