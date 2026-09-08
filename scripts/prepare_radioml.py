"""Inspect and validate a RadioML 2018.01A source file before preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.radioml import RadioML2018Dataset, RadioMLSchema, inspect_radioml_hdf5
from src.utils.config import ConfigError, load_config, require_resolved


def _is_tbd(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip().upper() == "TBD")


def _configured_input(config: dict) -> Path | None:
    dataset = config.get("dataset", {})
    filename = dataset.get("file")
    if _is_tbd(filename):
        return None
    candidate = Path(filename)
    return candidate if candidate.is_absolute() else Path(dataset.get("path", "data/raw")) / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", help="Explicit RadioML HDF5 file; overrides dataset.file")
    parser.add_argument(
        "--validate-configured-schema",
        action="store_true",
        help="Instantiate the configured adapter after inspection; requires resolved schema and task fields",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        input_path = Path(args.input) if args.input else _configured_input(config)
        if input_path is None:
            raise ConfigError(
                "Provide --input or set dataset.file to the actual RadioML HDF5 file. "
                "Inspection does not require task labels or other experimental assumptions."
            )
        inventory = inspect_radioml_hdf5(input_path)
        print(json.dumps({name: {"shape": shape, "dtype": dtype} for name, (shape, dtype) in inventory.items()}, indent=2))
        if not args.validate_configured_schema:
            print(
                "Inspection complete. Record verified dataset.schema keys and class semantics in the config; "
                "no key or label meaning was inferred."
            )
            return
        require_resolved(
            config,
            [
                "dataset.task_type",
                "dataset.classes",
                "dataset.input_length",
                "dataset.schema.samples_key",
                "dataset.schema.labels_key",
            ],
        )
        dataset_config = config["dataset"]
        schema_config = dataset_config["schema"]
        snr_key = schema_config.get("snr_key")
        if _is_tbd(snr_key):
            snr_key = None
        snr_min = None if _is_tbd(dataset_config.get("snr_min")) else dataset_config["snr_min"]
        snr_max = None if _is_tbd(dataset_config.get("snr_max")) else dataset_config["snr_max"]
        dataset = RadioML2018Dataset(
            input_path,
            RadioMLSchema(
                samples_key=schema_config["samples_key"],
                labels_key=schema_config["labels_key"],
                snr_key=snr_key,
            ),
            classes=None if str(dataset_config["classes"]).lower() == "all" else dataset_config["classes"],
            class_names=None if _is_tbd(schema_config.get("class_names")) else schema_config["class_names"],
            snr_min=snr_min,
            snr_max=snr_max,
            max_examples_per_class=dataset_config.get("max_examples_per_class"),
            seed=int(dataset_config.get("sample_seed", config["seed"])),
        )
        if len(dataset) == 0:
            raise ConfigError("The configured filters selected zero RadioML samples.")
        sample, label, metadata = dataset[0]
        expected_length = int(dataset_config["input_length"])
        if sample.shape != (2, expected_length):
            raise ConfigError(
                f"Configured input_length={expected_length}, but the first selected sample is {tuple(sample.shape)}."
            )
        print(f"Validated {len(dataset)} selected samples; first label={label}, metadata={metadata}.")
    except (ConfigError, FileNotFoundError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
