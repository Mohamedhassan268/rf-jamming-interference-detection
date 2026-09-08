import numpy as np
import pytest

from src.data.preprocessing import (
    add_awgn,
    amplitude_scale,
    frequency_offset,
    normalize,
    phase_rotate,
    remove_dc,
    rms_normalize,
)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: normalize(x, "none"),
        rms_normalize,
        remove_dc,
        lambda x: amplitude_scale(x, 0.7),
        lambda x: phase_rotate(x, 0.4),
        lambda x: add_awgn(x, 10.0, np.random.default_rng(7)),
        lambda x: frequency_offset(x, 0.01),
    ],
)
def test_operations_preserve_shape_dtype_and_finiteness(operation):
    signal = np.random.default_rng(1).normal(size=(4, 2, 128)).astype(np.float32)
    output = operation(signal)
    assert output.shape == signal.shape
    assert output.dtype == signal.dtype
    assert np.isfinite(output).all()


@pytest.mark.parametrize("magnitude", [0.0, 1e-30])
def test_rms_normalization_is_stable_near_zero(magnitude):
    signal = np.full((2, 64), magnitude, dtype=np.float32)
    output = rms_normalize(signal)
    assert output.shape == signal.shape
    assert output.dtype == signal.dtype
    assert np.isfinite(output).all()


def test_invalid_shape_is_rejected():
    with pytest.raises(ValueError, match="I/Q"):
        rms_normalize(np.zeros((3, 128), dtype=np.float32))

