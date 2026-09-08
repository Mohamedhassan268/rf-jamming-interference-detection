import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation.calibration import expected_calibration_error, maximum_softmax_confidence
from src.evaluation.jamming import binary_jamming_metrics
from src.evaluation.metrics import classification_metrics
from src.evaluation.per_snr import metrics_per_snr


def test_classification_metrics_match_known_example():
    truth = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, 1, 2, 0])
    result = classification_metrics(truth, predicted, labels=[0, 1, 2])
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_f1"] == pytest.approx(f1_score(truth, predicted, average="macro"))
    assert np.asarray(result["confusion_matrix"]).shape == (3, 3)


def test_per_snr_groups_samples():
    result = metrics_per_snr([0, 0, 1, 1], [0, 1, 1, 1], [-10, -10, 0, None])
    assert set(result) == {-10.0, 0.0}
    assert result[-10.0]["accuracy"] == pytest.approx(0.5)


def test_confidence_and_ece_are_bounded():
    logits = np.array([[4.0, 0.0], [0.0, 3.0], [1.0, 2.0]])
    truth = np.array([0, 1, 0])
    confidence = maximum_softmax_confidence(logits)
    ece = expected_calibration_error(logits, truth, n_bins=4)
    assert np.all((confidence >= 0) & (confidence <= 1))
    assert 0 <= ece <= 1


def test_binary_jamming_metrics_report_false_alarms_and_condition_rates():
    truth = [0, 1, 0, 1, 0, 1]
    predicted = [0, 1, 1, 0, 0, 1]
    metadata = [
        {"jammer_type": None, "jsr_db_requested": None},
        {"jammer_type": "tone", "jsr_db_requested": -10.0},
        {"jammer_type": None, "jsr_db_requested": None},
        {"jammer_type": "tone", "jsr_db_requested": 0.0},
        {"jammer_type": None, "jsr_db_requested": None},
        {"jammer_type": "barrage", "jsr_db_requested": 0.0},
    ]
    result = binary_jamming_metrics(truth, predicted, metadata)
    assert result["clean_false_positive_rate"] == {"rate": pytest.approx(1 / 3), "support": 3}
    assert result["jammed_detection_rate"] == {"rate": pytest.approx(2 / 3), "support": 3}
    assert result["detection_rate_by_jammer"]["tone"]["rate"] == pytest.approx(0.5)
    assert result["detection_rate_by_jsr_db"]["0"]["rate"] == pytest.approx(0.5)
