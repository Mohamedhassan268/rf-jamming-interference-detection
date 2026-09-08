"""Train the configured compact RF baseline after all task facts are resolved."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.pipeline import TransformedSubset, build_radioml_dataset, collate_iq
from src.data.splitting import make_splits
from src.models import CompactRFNet, count_trainable_parameters
from src.training import train_model
from src.utils.config import ConfigError, load_config, require_resolved
from src.utils.logging import create_run_directory, save_run_metadata
from src.utils.seed import seed_everything


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError("CUDA was requested but is not available.")
    return device


def _index_fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", default="baseline")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        require_resolved(
            config,
            ["dataset.task_type", "dataset.classes", "dataset.input_length", "model.num_classes"],
        )
        seed = int(config["seed"])
        seed_everything(seed, bool(config.get("deterministic", False)))
        dataset = build_radioml_dataset(config)
        model_config = config["model"]
        if int(model_config["num_classes"]) != len(dataset.selected_class_ids):
            raise ConfigError(
                f"model.num_classes={model_config['num_classes']} but filters selected "
                f"{len(dataset.selected_class_ids)} classes: {dataset.selected_class_names}."
            )
        splits = make_splits(
            dataset.selected_labels,
            float(config["dataset"]["train_fraction"]),
            float(config["dataset"]["validation_fraction"]),
            float(config["dataset"]["test_fraction"]),
            seed,
        )
        preprocessing = config.get("preprocessing", {})
        train_data = TransformedSubset(dataset, splits.train, preprocessing, augment=True, seed=seed)
        validation_data = TransformedSubset(dataset, splits.validation, preprocessing, augment=False, seed=seed)
        generator = torch.Generator().manual_seed(seed)
        batch_size = int(config["training"]["batch_size"])
        workers = int(config["training"].get("num_workers", 0))
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=generator,
            collate_fn=collate_iq,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_iq,
        )
        model = CompactRFNet(
            num_classes=int(model_config["num_classes"]),
            input_length=int(config["dataset"]["input_length"]),
            conv_channels=model_config["conv_channels"],
            kernel_sizes=model_config["kernel_sizes"],
            pooling=model_config["pooling"],
            dropout=float(model_config["dropout"]),
            classifier_size=int(model_config["classifier_size"]),
        )
        parameter_count = count_trainable_parameters(model)
        device = _device(str(config["training"].get("device", "auto")))
        print(f"device={device} trainable_parameters={parameter_count:,}")
        result = train_model(model, train_loader, validation_loader, config["training"], device)
        run_dir = create_run_directory(args.runs_dir, args.experiment_id)
        split_payload = {"train": splits.train, "validation": splits.validation, "test": splits.test}
        np.savez_compressed(run_dir / "splits.npz", **split_payload)
        common_checkpoint = {
            "config": config,
            "class_names": dataset.selected_class_names,
            "selected_source_indices_sha256": _index_fingerprint(dataset.indices),
            "splits": {name: values.tolist() for name, values in split_payload.items()},
            "parameter_count": parameter_count,
            "best_epoch": result.best_epoch,
        }
        torch.save({**common_checkpoint, "model_state_dict": result.best_state}, run_dir / "best_model.pt")
        torch.save({**common_checkpoint, "model_state_dict": result.last_state}, run_dir / "last_model.pt")
        pd.DataFrame(result.history).to_csv(run_dir / "history.csv", index=False)
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "selection_metric": "validation_loss",
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": min(row["validation_loss"] for row in result.history),
                    "test_metrics": None,
                },
                handle,
                indent=2,
            )
        save_run_metadata(
            run_dir,
            config,
            {
                "seed": seed,
                "model": model.__class__.__name__,
                "parameter_count": parameter_count,
                "class_names": dataset.selected_class_names,
                "dataset_selected_samples": len(dataset),
                "split_counts": {name: len(values) for name, values in split_payload.items()},
            },
        )
        print(f"Training complete. Artifacts saved to {run_dir}. The test split was not evaluated.")
    except (ConfigError, FileNotFoundError, KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
