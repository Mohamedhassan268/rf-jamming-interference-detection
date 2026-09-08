"""Create descriptive source/target I/Q distribution plots from `.npy` arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np


def _load(path: str) -> np.ndarray:
    samples = np.load(path, mmap_mode="r", allow_pickle=False)
    if samples.ndim != 3:
        raise ValueError(f"Expected 3-D array at {path}, got {samples.shape}.")
    if samples.shape[1] == 2:
        return samples
    if samples.shape[2] == 2:
        return np.transpose(samples, (0, 2, 1))
    raise ValueError(f"No I/Q axis of length 2 at {path}: {samples.shape}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", default="figures/source_target_distributions.png")
    parser.add_argument("--max-values", type=int, default=500_000)
    args = parser.parse_args()
    try:
        source, target = _load(args.source), _load(args.target)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for values, name in ((source, "source"), (target, "target")):
        flat_i = np.asarray(values[:, 0, :]).reshape(-1)[: args.max_values]
        flat_q = np.asarray(values[:, 1, :]).reshape(-1)[: args.max_values]
        axes[0].hist(flat_i, bins=100, density=True, alpha=0.5, label=name)
        axes[1].hist(flat_q, bins=100, density=True, alpha=0.5, label=name)
        axes[2].hist(np.hypot(flat_i, flat_q), bins=100, density=True, alpha=0.5, label=name)
    for axis, title in zip(axes, ("I amplitude", "Q amplitude", "Magnitude")):
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    print(f"Saved descriptive comparison to {destination}. It is not proof of causality or adaptation success.")


if __name__ == "__main__":
    main()
