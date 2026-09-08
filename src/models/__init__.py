"""RF model definitions."""

from .compact_cnn import CompactRFNet, ProvisionalCompactRFNet, count_trainable_parameters

__all__ = ["CompactRFNet", "ProvisionalCompactRFNet", "count_trainable_parameters"]
