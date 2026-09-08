from types import SimpleNamespace

import numpy as np
import pytest

from scripts.download_radioml_subset import (
    ALL_SNRS,
    CLASS_NAMES,
    FRAMES_PER_CONDITION,
    _condition_ranges,
)
from scripts.train import _split_strata


def test_condition_ranges_are_balanced_deterministic_and_in_bounds():
    first = _condition_ranges([-20, 0, 30], samples_per_condition=8, seed=7)
    second = _condition_ranges([-20, 0, 30], samples_per_condition=8, seed=7)
    assert first == second
    assert len(first) == len(CLASS_NAMES) * 3
    for class_id, snr_db, start, stop in first:
        condition = class_id * len(ALL_SNRS) + ALL_SNRS.index(snr_db)
        lower = condition * FRAMES_PER_CONDITION
        assert lower <= start < stop <= lower + FRAMES_PER_CONDITION
        assert stop - start == 8


def test_condition_ranges_reject_unsupported_snr():
    with pytest.raises(ValueError, match="SNR values"):
        _condition_ranges([-21], samples_per_condition=8, seed=7)


def test_training_strata_combine_modulation_and_snr():
    dataset = SimpleNamespace(
        selected_labels=np.array([0, 0, 1, 1]),
        snrs=np.array([-10, 0, -10, 0]),
        indices=np.arange(4),
    )
    strata = _split_strata(dataset)
    assert len(np.unique(strata)) == 4
