"""
HOTAConfig.validate() error paths and boundary values, plus invariants:
- HOTAConfig is a frozen dataclass (no post-construction mutation)
- iou_thresholds array is locked read-only
- Defaults are constructed per-instance (no shared mutable state across evaluators)
"""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from reid_hota import HOTAConfig, HOTAReIDEvaluator
from reid_hota.hota_errors import (
    EmptyIOUThresholdsError,
    InvalidIDAlignmentMethodError,
    InvalidIOUThresholdsRangeError,
    InvalidSimilarityMetricError,
)


def test_invalid_id_alignment_method():
    cfg = HOTAConfig(id_alignment_method='bogus')
    with pytest.raises(InvalidIDAlignmentMethodError):
        cfg.validate()


def test_invalid_similarity_metric():
    cfg = HOTAConfig(similarity_metric='l2')
    with pytest.raises(InvalidSimilarityMetricError):
        cfg.validate()


def test_empty_iou_thresholds():
    cfg = HOTAConfig(iou_thresholds=np.array([]))
    with pytest.raises(EmptyIOUThresholdsError):
        cfg.validate()


def test_iou_thresholds_above_one():
    cfg = HOTAConfig(iou_thresholds=np.array([0.5, 1.5]))
    with pytest.raises(InvalidIOUThresholdsRangeError):
        cfg.validate()


def test_iou_thresholds_below_zero():
    cfg = HOTAConfig(iou_thresholds=np.array([-0.1, 0.5]))
    with pytest.raises(InvalidIOUThresholdsRangeError):
        cfg.validate()


def test_iou_thresholds_nan():
    # nan <= 1 is False → range check fires
    cfg = HOTAConfig(iou_thresholds=np.array([0.5, np.nan]))
    with pytest.raises(InvalidIOUThresholdsRangeError):
        cfg.validate()


def test_iou_thresholds_inf():
    cfg = HOTAConfig(iou_thresholds=np.array([np.inf]))
    with pytest.raises(InvalidIOUThresholdsRangeError):
        cfg.validate()


def test_iou_thresholds_zero_and_one_allowed():
    """Boundary values 0.0 and 1.0 are valid."""
    cfg = HOTAConfig(iou_thresholds=np.array([0.0, 1.0]))
    cfg.validate()  # no exception


def test_iou_thresholds_as_python_list_raises():
    """
    validate() does `self.iou_thresholds >= 0` which raises TypeError on Python lists.
    Pin that the caller must pass a numpy array.
    """
    cfg = HOTAConfig(iou_thresholds=[0.5, 0.7])
    with pytest.raises(TypeError):
        cfg.validate()


def test_valid_config_does_not_raise():
    HOTAConfig().validate()


# ---------------------------------------------------------------------------
# Immutability and per-instance defaults
# ---------------------------------------------------------------------------

def test_evaluator_default_config_not_shared():
    """Each HOTAReIDEvaluator gets its own config instance, not a shared one."""
    e1 = HOTAReIDEvaluator()
    e2 = HOTAReIDEvaluator()
    assert e1.config is not e2.config


def test_hota_config_is_frozen():
    """Attribute assignment on a frozen dataclass raises FrozenInstanceError."""
    cfg = HOTAConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.class_ids = [1, 2]


def test_hota_config_iou_thresholds_array_is_readonly():
    """In-place mutation of the iou_thresholds array is rejected by numpy."""
    cfg = HOTAConfig(iou_thresholds=np.array([0.3, 0.5, 0.7]))
    with pytest.raises(ValueError, match="read-only"):
        cfg.iou_thresholds[0] = 99.0
