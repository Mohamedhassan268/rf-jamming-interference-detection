import pytest

from src.utils.config import ConfigError, require_resolved


def test_unresolved_task_has_specific_message():
    with pytest.raises(ConfigError, match="Task label definition has not yet been reconstructed"):
        require_resolved({"dataset": {"task_type": None}}, ["dataset.task_type"])


def test_resolved_value_passes():
    require_resolved({"dataset": {"task_type": "verified-test-task"}}, ["dataset.task_type"])
