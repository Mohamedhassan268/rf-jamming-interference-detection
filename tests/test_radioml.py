import h5py
import numpy as np
import pytest
import torch

from src.data.radioml import RadioML2018Dataset, RadioMLSchema, inspect_radioml_hdf5


def _write_fixture(path, shape=(8, 16, 2)):
    with h5py.File(path, "w") as handle:
        samples = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        labels = np.eye(2, dtype=np.float32)[np.arange(shape[0]) % 2]
        snr = np.array([-10, -10, 0, 0, 10, 10, 20, 20], dtype=np.float32)[: shape[0], None]
        handle.create_dataset("X", data=samples)
        handle.create_dataset("Y", data=labels)
        handle.create_dataset("Z", data=snr)


def test_inspection_and_nt2_conversion(tmp_path):
    path = tmp_path / "radio.h5"
    _write_fixture(path)
    inventory = inspect_radioml_hdf5(path)
    assert inventory["X"][0] == (8, 16, 2)
    dataset = RadioML2018Dataset(
        path,
        RadioMLSchema("X", "Y", "Z"),
        classes=["class_b"],
        class_names=["class_a", "class_b"],
        snr_min=0,
    )
    sample, label, metadata = dataset[0]
    assert sample.shape == (2, 16)
    assert sample.dtype == torch.float32
    assert label == 0
    assert metadata["source_label_id"] == 1
    assert metadata["class_name"] == "class_b"
    assert metadata["snr_db"] >= 0


def test_subsampling_is_reproducible(tmp_path):
    path = tmp_path / "radio.h5"
    _write_fixture(path)
    schema = RadioMLSchema("X", "Y", "Z")
    first = RadioML2018Dataset(path, schema, max_examples_per_class=2, seed=7)
    second = RadioML2018Dataset(path, schema, max_examples_per_class=2, seed=7)
    np.testing.assert_array_equal(first.indices, second.indices)


def test_dataset_reuses_and_can_close_hdf5_handle(tmp_path):
    path = tmp_path / "radio.h5"
    _write_fixture(path)
    dataset = RadioML2018Dataset(path, RadioMLSchema("X", "Y", "Z"))
    dataset[0]
    first_handle = dataset._handle
    dataset[1]
    assert dataset._handle is first_handle
    dataset.close()
    assert dataset._handle is None


def test_snr_filter_requires_metadata_key(tmp_path):
    path = tmp_path / "radio.h5"
    _write_fixture(path)
    with pytest.raises(ValueError, match="SNR filtering"):
        RadioML2018Dataset(path, RadioMLSchema("X", "Y"), snr_min=0)


def test_incompatible_shape_is_rejected(tmp_path):
    path = tmp_path / "bad.h5"
    _write_fixture(path, shape=(8, 3, 16))
    with pytest.raises(ValueError, match="I/Q axis"):
        RadioML2018Dataset(path, RadioMLSchema("X", "Y", "Z"))
