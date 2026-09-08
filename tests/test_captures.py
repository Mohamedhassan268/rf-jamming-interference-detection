import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.captures import MappedCapturedRFDataset, load_capture_session


def _write_session(path, labels=("clean", "interference")):
    path.mkdir()
    with (path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"receiver": None, "unrecognized_field": "preserved"}, handle)
    np.save(path / "samples.npy", np.zeros((len(labels), 16, 2), dtype=np.float32))
    pd.DataFrame({"label": labels, "snr_db": [None] * len(labels)}).to_csv(
        path / "labels.csv", index=False
    )


def test_capture_loader_transposes_and_preserves_open_metadata(tmp_path):
    session_path = tmp_path / "session"
    _write_session(session_path)
    session = load_capture_session(session_path)
    assert session.samples.shape == (2, 2, 16)
    assert session.metadata["receiver"] is None
    assert session.metadata["unrecognized_field"] == "preserved"


def test_capture_labels_map_to_checkpoint_order(tmp_path):
    session_path = tmp_path / "session"
    _write_session(session_path)
    dataset = MappedCapturedRFDataset(session_path, ["interference", "clean"])
    sample, label, metadata = dataset[0]
    assert sample.shape == (2, 16)
    assert sample.dtype == torch.float32
    assert label == 1
    assert metadata["source_class_name"] == "clean"


def test_unknown_capture_label_is_rejected(tmp_path):
    session_path = tmp_path / "session"
    _write_session(session_path, labels=("unknown",))
    with pytest.raises(ValueError, match="never guessed"):
        MappedCapturedRFDataset(session_path, ["clean", "interference"])
