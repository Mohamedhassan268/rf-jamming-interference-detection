import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.data.jamming import (
    FullJammerStressDataset,
    PairedBinaryJammingDataset,
    complex_power,
    generate_jammer,
    inject_jammer,
)


CONFIG = {
    "strategy": "paired_clean_jammed",
    "labels": {"clean": 0, "jammed": 1},
    "jammer_types": ["tone", "chirp", "barrage"],
    "jsr_db_values": [-10.0, 0.0, 10.0],
    "seed": 42,
    "tone": {
        "normalized_frequency_min": -0.45,
        "normalized_frequency_max": 0.45,
        "excluded_dc_half_width": 0.02,
    },
    "chirp": {
        "normalized_start_min": -0.40,
        "normalized_start_max": 0.40,
        "normalized_sweep_min": 0.05,
        "normalized_sweep_max": 0.40,
    },
}


class SourceDataset(Dataset):
    def __init__(self):
        self.samples = torch.ones(3, 2, 64, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], index % 2, {"sample_id": 100 + index, "snr_db": 10.0}


@pytest.mark.parametrize("jammer_type", ["tone", "chirp", "barrage"])
def test_generated_jammers_have_unit_power_shape_and_dtype(jammer_type):
    jammer, metadata = generate_jammer(
        jammer_type, 128, np.dtype("float32"), np.random.default_rng(9), CONFIG
    )
    assert jammer.shape == (2, 128)
    assert jammer.dtype == np.float32
    assert np.isfinite(jammer).all()
    assert complex_power(jammer) == pytest.approx(1.0, rel=1e-5)
    assert isinstance(metadata, dict)


@pytest.mark.parametrize("jsr_db", [-10.0, 0.0, 10.0])
def test_injection_achieves_requested_jsr(jsr_db):
    rng = np.random.default_rng(2)
    signal = rng.normal(size=(2, 256)).astype(np.float32)
    jammer, _ = generate_jammer("barrage", 256, signal.dtype, rng, CONFIG)
    mixed, achieved = inject_jammer(signal, jammer, jsr_db)
    assert mixed.shape == signal.shape
    assert mixed.dtype == signal.dtype
    assert achieved == pytest.approx(jsr_db, abs=1e-5)


def test_paired_dataset_is_balanced_and_reproducible():
    first = PairedBinaryJammingDataset(SourceDataset(), [0, 2], CONFIG)
    second = PairedBinaryJammingDataset(SourceDataset(), [0, 2], CONFIG)
    assert len(first) == 4
    labels = [first[index][1] for index in range(len(first))]
    assert labels == [0, 1, 0, 1]
    torch.testing.assert_close(first[1][0], second[1][0])
    assert first[1][2]["jammer_type"] == second[1][2]["jammer_type"]
    assert first[1][2]["jsr_db_requested"] == second[1][2]["jsr_db_requested"]


def test_paired_condition_assignment_is_balanced():
    dataset = SourceDataset()
    paired = PairedBinaryJammingDataset(dataset, list(range(len(dataset))), CONFIG)
    counts = list(paired.condition_counts().values())
    assert max(counts) - min(counts) <= 1


def test_full_stress_dataset_emits_every_condition_per_source():
    stress = FullJammerStressDataset(SourceDataset(), [0, 2], CONFIG)
    assert len(stress) == 2 * 3 * 3
    metadata = [stress[index][2] for index in range(9)]
    conditions = {(item["jammer_type"], item["jsr_db_requested"]) for item in metadata}
    assert conditions == {
        (jammer, jsr) for jammer in CONFIG["jammer_types"] for jsr in CONFIG["jsr_db_values"]
    }
    torch.testing.assert_close(stress[0][0], FullJammerStressDataset(SourceDataset(), [0, 2], CONFIG)[0][0])


def test_zero_power_source_is_rejected_for_jamming():
    jammer = np.ones((2, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="zero-power source"):
        inject_jammer(np.zeros_like(jammer), jammer, 0.0)
