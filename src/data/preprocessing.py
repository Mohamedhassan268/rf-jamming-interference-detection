"""Composable preprocessing and augmentation for `[2, T]` I/Q windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Normalization = Literal["none", "rms"]


def _validate_iq(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x)
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected [2, T] or [N, 2, T], got {array.shape}.")
    channel_axis = 0 if array.ndim == 2 else 1
    if array.shape[channel_axis] != 2:
        raise ValueError(f"I/Q channel dimension must be 2, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("I/Q arrays must use a floating-point dtype.")
    return array


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Subtract the temporal mean independently from I and Q."""
    array = _validate_iq(x)
    return array - array.mean(axis=-1, keepdims=True, dtype=array.dtype)


def rms_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each I/Q window by `sqrt(mean(I^2 + Q^2))` safely."""
    array = _validate_iq(x)
    power = np.sum(array * array, axis=-2)
    rms = np.sqrt(np.mean(power, axis=-1, keepdims=True))
    if array.ndim == 2:
        denominator = np.maximum(rms, np.asarray(eps, dtype=array.dtype))
    else:
        denominator = np.maximum(rms, np.asarray(eps, dtype=array.dtype))[:, None, :]
    return array / denominator


def normalize(x: np.ndarray, method: Normalization = "none", eps: float = 1e-8) -> np.ndarray:
    """Apply the selected normalization without changing shape or dtype."""
    array = _validate_iq(x)
    if method == "none":
        return array.copy()
    if method == "rms":
        return rms_normalize(array, eps=eps)
    raise ValueError(f"Unknown normalization method: {method!r}.")


def amplitude_scale(x: np.ndarray, scale: float) -> np.ndarray:
    """Multiply an I/Q window by a scalar amplitude factor."""
    array = _validate_iq(x)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative.")
    return array * np.asarray(scale, dtype=array.dtype)


def phase_rotate(x: np.ndarray, radians: float) -> np.ndarray:
    """Rotate I/Q samples by a constant phase angle."""
    array = _validate_iq(x)
    cosine = np.asarray(np.cos(radians), dtype=array.dtype)
    sine = np.asarray(np.sin(radians), dtype=array.dtype)
    i, q = array[..., 0, :], array[..., 1, :]
    return np.stack((i * cosine - q * sine, i * sine + q * cosine), axis=-2)


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add white Gaussian noise at a window-relative SNR in dB."""
    array = _validate_iq(x)
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")
    generator = rng or np.random.default_rng()
    signal_power = np.mean(np.sum(array * array, axis=-2), axis=-1, keepdims=True)
    noise_complex_power = signal_power / np.asarray(10.0 ** (snr_db / 10.0), dtype=array.dtype)
    std = np.sqrt(noise_complex_power / np.asarray(2.0, dtype=array.dtype))
    if array.ndim == 3:
        std = std[:, None, :]
    noise = generator.normal(size=array.shape).astype(array.dtype, copy=False) * std
    return array + noise


def frequency_offset(x: np.ndarray, normalized_offset: float) -> np.ndarray:
    """Apply cycles-per-sample carrier offset to an I/Q window."""
    array = _validate_iq(x)
    if not np.isfinite(normalized_offset):
        raise ValueError("normalized_offset must be finite.")
    phase = 2.0 * np.pi * normalized_offset * np.arange(array.shape[-1])
    cosine = np.cos(phase).astype(array.dtype, copy=False)
    sine = np.sin(phase).astype(array.dtype, copy=False)
    i, q = array[..., 0, :], array[..., 1, :]
    return np.stack((i * cosine - q * sine, i * sine + q * cosine), axis=-2)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Independently switchable preprocessing parameters."""

    normalization: Normalization = "none"
    remove_dc: bool = False


def preprocess(x: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply deterministic preprocessing in a documented order."""
    output = _validate_iq(x).copy()
    if config.remove_dc:
        output = remove_dc(output)
    return normalize(output, method=config.normalization)

