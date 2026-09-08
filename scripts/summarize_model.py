"""Summarize the provisional reconstruction baseline without training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import ProvisionalCompactRFNet, count_trainable_parameters
from src.utils.config import ConfigError, load_config


def _build(config: dict, num_classes: int) -> ProvisionalCompactRFNet:
    model = config["model"]
    input_length = config["dataset"].get("input_length") or 8
    return ProvisionalCompactRFNet(
        num_classes=num_classes,
        input_length=int(input_length),
        conv_channels=model["conv_channels"],
        kernel_sizes=model["kernel_sizes"],
        pooling=model["pooling"],
        dropout=float(model["dropout"]),
        classifier_size=int(model["classifier_size"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--num-classes", type=int, help="Verified output-class count for an exact total")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        count_two = count_trainable_parameters(_build(config, 2))
        count_three = count_trainable_parameters(_build(config, 3))
        per_class = count_three - count_two
        fixed = count_two - 2 * per_class
        print("Model: ProvisionalCompactRFNet")
        print("Historical fidelity: not established")
        print(f"Conv channels: {config['model']['conv_channels']}")
        print(f"Kernel sizes: {config['model']['kernel_sizes']}")
        print(f"Pooling: {config['model']['pooling']}")
        print(f"Classifier size: {config['model']['classifier_size']}")
        print(f"Trainable-parameter formula: {fixed:,} + {per_class:,} * num_classes")
        if args.num_classes is None:
            print("Exact total: unresolved until the verified label count is supplied")
        else:
            if args.num_classes < 2:
                raise ConfigError("--num-classes must be at least 2.")
            exact = count_trainable_parameters(_build(config, args.num_classes))
            print(f"Exact total for num_classes={args.num_classes}: {exact:,}")
    except (ConfigError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
