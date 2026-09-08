"""Run dataset-backed split, generation, checksum, JSR, and model sanity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.jamming import PairedBinaryJammingDataset
from src.data.pipeline import build_radioml_dataset
from src.data.splitting import make_splits, radioml_strata
from src.evaluation import file_sha256, generation_audit, split_audit
from src.models import CompactRFNet, count_trainable_parameters
from src.utils.config import ConfigError, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        dataset = build_radioml_dataset(config)
        seed = int(config["seed"])
        splits = make_splits(
            radioml_strata(dataset),
            float(config["dataset"]["train_fraction"]),
            float(config["dataset"]["validation_fraction"]),
            float(config["dataset"]["test_fraction"]),
            seed,
        )
        train = PairedBinaryJammingDataset(dataset, splits.train, config["label_generation"])
        validation = PairedBinaryJammingDataset(dataset, splits.validation, config["label_generation"])
        model_config = config["model"]
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
        if parameter_count != 240_962:
            raise ConfigError(f"Expected 240,962 parameters, got {parameter_count:,}.")
        checksum = file_sha256(dataset.path)
        expected = config["dataset"]["subset"]["local_file_sha256"]
        if checksum != expected:
            raise ConfigError(f"Dataset checksum mismatch: expected {expected}, got {checksum}.")
        report = {
            "dataset_subset_sha256": checksum,
            "source_split": split_audit(dataset, splits, seed),
            "generation": generation_audit(dataset, train, validation),
            "model": {"trainable_parameters": parameter_count, "passed": True},
        }
        print(json.dumps(report, indent=2))
        print("BASELINE PREFLIGHT PASSED")
    except (ConfigError, FileNotFoundError, KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
