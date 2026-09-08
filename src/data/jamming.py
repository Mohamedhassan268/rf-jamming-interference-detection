"""Explicit synthetic-jammer generation for the new binary detection task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_JAMMERS = ("tone", "chirp", "barrage")


def _condition_grid(config: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    jammer_types = tuple(config.get("jammer_types", ()))
    if not jammer_types or set(jammer_types) - set(SUPPORTED_JAMMERS):
        raise ValueError(f"jammer_types must be a non-empty subset of {SUPPORTED_JAMMERS}.")
    jsr_values = np.asarray(config.get("jsr_db_values", ()), dtype=np.float64)
    if not len(jsr_values) or not np.isfinite(jsr_values).all():
        raise ValueError("jsr_db_values must contain finite values.")
    return tuple((jammer, float(jsr)) for jammer in jammer_types for jsr in jsr_values)


def _jammer_rng(seed: int, source_position: int, condition_index: int) -> np.random.Generator:
    """Create a stable waveform RNG independent of iteration and worker order."""
    return np.random.default_rng(
        np.random.SeedSequence([seed, source_position, condition_index, 0x4A5352])
    )


def _validate_window(sample: np.ndarray) -> np.ndarray:
    array = np.asarray(sample)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"Expected one I/Q window with shape [2, T], got {array.shape}.")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("I/Q samples must use a floating-point dtype.")
    if not np.isfinite(array).all():
        raise ValueError("I/Q samples must be finite before jammer injection.")
    return array


def complex_power(sample: np.ndarray) -> float:
    """Return mean `I² + Q²` power for one I/Q window."""
    array = _validate_window(sample)
    return float(np.mean(np.sum(array.astype(np.float64) ** 2, axis=0)))


def _unit_power(sample: np.ndarray) -> np.ndarray:
    power = complex_power(sample)
    if power <= np.finfo(np.float64).tiny:
        raise ValueError("Generated jammer has zero power.")
    return sample / np.asarray(np.sqrt(power), dtype=sample.dtype)


def _sample_tone_frequency(config: dict[str, Any], rng: np.random.Generator) -> float:
    minimum = float(config["normalized_frequency_min"])
    maximum = float(config["normalized_frequency_max"])
    exclusion = float(config.get("excluded_dc_half_width", 0.0))
    if not (-0.5 <= minimum < maximum <= 0.5):
        raise ValueError("Tone normalized-frequency bounds must satisfy -0.5 <= min < max <= 0.5.")
    if exclusion < 0 or exclusion * 2 >= maximum - minimum:
        raise ValueError("Tone DC exclusion is incompatible with its frequency interval.")
    allowed = []
    if minimum < -exclusion:
        allowed.append((minimum, min(maximum, -exclusion)))
    if maximum > exclusion:
        allowed.append((max(minimum, exclusion), maximum))
    widths = np.asarray([high - low for low, high in allowed], dtype=np.float64)
    if not len(allowed) or np.any(widths <= 0):
        raise ValueError("Tone configuration leaves no allowed normalized frequency.")
    choice = int(rng.choice(len(allowed), p=widths / widths.sum()))
    return float(rng.uniform(*allowed[choice]))


def generate_jammer(
    jammer_type: str,
    length: int,
    dtype: np.dtype,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate a unit-power synthetic jammer and its sampled parameters."""
    if jammer_type not in SUPPORTED_JAMMERS:
        raise ValueError(f"Unsupported jammer type {jammer_type!r}; choose from {SUPPORTED_JAMMERS}.")
    if length < 2:
        raise ValueError("Jammer windows require at least two temporal samples.")
    output_dtype = np.dtype(dtype)
    n = np.arange(length, dtype=np.float64)
    initial_phase = float(rng.uniform(-np.pi, np.pi))
    if jammer_type == "tone":
        frequency = _sample_tone_frequency(config["tone"], rng)
        phase = 2.0 * np.pi * frequency * n + initial_phase
        metadata = {"normalized_frequency": frequency, "initial_phase_rad": initial_phase}
        jammer = np.stack((np.cos(phase), np.sin(phase))).astype(output_dtype)
    elif jammer_type == "chirp":
        chirp = config["chirp"]
        start = float(rng.uniform(float(chirp["normalized_start_min"]), float(chirp["normalized_start_max"])))
        sweep_magnitude = float(
            rng.uniform(float(chirp["normalized_sweep_min"]), float(chirp["normalized_sweep_max"]))
        )
        sweep = sweep_magnitude * (-1.0 if rng.random() < 0.5 else 1.0)
        phase_cycles = start * n + 0.5 * sweep * n * n / (length - 1)
        phase = 2.0 * np.pi * phase_cycles + initial_phase
        metadata = {
            "normalized_start_frequency": start,
            "normalized_sweep": sweep,
            "initial_phase_rad": initial_phase,
        }
        jammer = np.stack((np.cos(phase), np.sin(phase))).astype(output_dtype)
    else:
        jammer = rng.normal(size=(2, length)).astype(output_dtype)
        metadata = {}
    return _unit_power(jammer), metadata


def inject_jammer(
    sample: np.ndarray,
    jammer: np.ndarray,
    jsr_db: float,
) -> tuple[np.ndarray, float]:
    """Add a jammer at the requested jammer-to-signal power ratio in dB."""
    signal = _validate_window(sample)
    interference = _validate_window(jammer)
    if signal.shape != interference.shape:
        raise ValueError(f"Signal and jammer shapes differ: {signal.shape} vs {interference.shape}.")
    if not np.isfinite(jsr_db):
        raise ValueError("jsr_db must be finite.")
    signal_power = complex_power(signal)
    jammer_power = complex_power(interference)
    if signal_power <= np.finfo(np.float64).tiny:
        raise ValueError("Cannot define JSR for a zero-power source window.")
    desired_ratio = 10.0 ** (jsr_db / 10.0)
    scale = np.sqrt(signal_power * desired_ratio / jammer_power)
    scaled_jammer = interference * np.asarray(scale, dtype=signal.dtype)
    mixed = signal + scaled_jammer
    achieved_jsr = 10.0 * np.log10(complex_power(scaled_jammer) / signal_power)
    return mixed.astype(signal.dtype, copy=False), float(achieved_jsr)


class PairedBinaryJammingDataset(Dataset):
    """Return one clean and one deterministic jammed example per source window.

    `source_indices` must already belong to exactly one train/validation/test split.
    This enforces splitting before label generation so paired versions of a source
    window cannot leak across splits.
    """

    def __init__(
        self,
        source_dataset: Dataset,
        source_indices: Sequence[int],
        config: dict[str, Any],
    ) -> None:
        self.source_dataset = source_dataset
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        self.config = config
        if config.get("strategy") != "paired_clean_jammed":
            raise ValueError("label_generation.strategy must be 'paired_clean_jammed'.")
        if config.get("labels") != {"clean": 0, "jammed": 1}:
            raise ValueError("Binary label mapping must be exactly clean=0 and jammed=1.")
        self.seed = int(config["seed"])
        self.conditions = _condition_grid(config)
        # Shuffle source positions once, then cycle through the condition grid.
        # Every condition count therefore differs from every other by at most one.
        order = np.random.default_rng(self.seed).permutation(len(self.source_indices))
        assigned = np.empty(len(self.source_indices), dtype=np.int64)
        assigned[order] = np.arange(len(order), dtype=np.int64) % len(self.conditions)
        self.assigned_condition_indices = assigned

    def condition_counts(self) -> dict[str, int]:
        counts = np.bincount(self.assigned_condition_indices, minlength=len(self.conditions))
        return {
            f"{jammer}|{jsr:g}": int(counts[index])
            for index, (jammer, jsr) in enumerate(self.conditions)
        }

    def __len__(self) -> int:
        return 2 * len(self.source_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        source_offset = index // 2
        source_position = int(self.source_indices[source_offset])
        sample, _source_label, source_metadata = self.source_dataset[source_position]
        array = sample.numpy()
        is_jammed = bool(index % 2)
        metadata = {
            **source_metadata,
            "binary_label_name": "jammed" if is_jammed else "clean",
            "is_jammed": is_jammed,
            "label_generation_seed": self.seed,
            "jammer_type": None,
            "jsr_db_requested": None,
            "jsr_db_achieved": None,
            "jammer_parameters": None,
        }
        if not is_jammed:
            return sample.clone(), 0, metadata
        condition_index = int(self.assigned_condition_indices[source_offset])
        jammer_type, jsr_db = self.conditions[condition_index]
        rng = _jammer_rng(self.seed, source_position, condition_index)
        jammer, parameters = generate_jammer(jammer_type, array.shape[-1], array.dtype, rng, self.config)
        mixed, achieved_jsr = inject_jammer(array, jammer, jsr_db)
        metadata.update(
            {
                "jammer_type": jammer_type,
                "jsr_db_requested": jsr_db,
                "jsr_db_achieved": achieved_jsr,
                "jammer_parameters": parameters,
            }
        )
        return torch.from_numpy(np.ascontiguousarray(mixed)), 1, metadata


class FullJammerStressDataset(Dataset):
    """Return all configured jammer/JSR variants for source windows in one split."""

    def __init__(
        self,
        source_dataset: Dataset,
        source_indices: Sequence[int],
        config: dict[str, Any],
    ) -> None:
        self.source_dataset = source_dataset
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        self.config = config
        self.seed = int(config["seed"])
        self.conditions = _condition_grid(config)

    def __len__(self) -> int:
        return len(self.source_indices) * len(self.conditions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        source_offset, condition_index = divmod(index, len(self.conditions))
        source_position = int(self.source_indices[source_offset])
        sample, _source_label, source_metadata = self.source_dataset[source_position]
        array = sample.numpy()
        jammer_type, jsr_db = self.conditions[condition_index]
        rng = _jammer_rng(self.seed, source_position, condition_index)
        jammer, parameters = generate_jammer(jammer_type, array.shape[-1], array.dtype, rng, self.config)
        mixed, achieved_jsr = inject_jammer(array, jammer, jsr_db)
        metadata = {
            **source_metadata,
            "binary_label_name": "jammed",
            "is_jammed": True,
            "label_generation_seed": self.seed,
            "jammer_type": jammer_type,
            "jsr_db_requested": jsr_db,
            "jsr_db_achieved": achieved_jsr,
            "jammer_parameters": parameters,
        }
        return torch.from_numpy(np.ascontiguousarray(mixed)), 1, metadata
