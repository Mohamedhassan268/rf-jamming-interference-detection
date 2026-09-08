import pytest
import torch

from src.models import CompactRFNet, ProvisionalCompactRFNet, count_trainable_parameters


def test_compact_model_input_output_dimensions():
    model = CompactRFNet(num_classes=5, input_length=256)
    assert isinstance(model, ProvisionalCompactRFNet)
    output = model(torch.randn(7, 2, 256))
    assert output.shape == (7, 5)
    assert count_trainable_parameters(model) > 0


def test_compact_model_rejects_wrong_length():
    model = CompactRFNet(num_classes=3, input_length=128)
    with pytest.raises(ValueError, match="input length"):
        model(torch.randn(2, 2, 64))
