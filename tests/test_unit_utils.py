"""
Unit tests for utility functions in reid_hota.hota_utils and reid_hota.cost_matrix.

These tests exercise the math primitives in isolation — independent of
the full HOTA pipeline.
"""
import copy

import numpy as np
import pytest

from reid_hota import HOTAConfig, HOTAData
from reid_hota.constants import BOX_FORMAT
from reid_hota.cost_matrix import CostMatrixDataFrame
from reid_hota.hota_errors import MissingVideoIDError, UnsupportedBoxFormatError
from reid_hota.hota_utils import (
    calculate_box_ious,
    calculate_latlon_l2,
    calculate_latlonalt_l2,
    merge_hota_data,
    normalize_cost_matrix,
    ECEF_L2_DECAY_METERS,
)


# ---------------------------------------------------------------------------
# Sanity: box format constant
# ---------------------------------------------------------------------------

def test_box_format_constant_is_xyxy():
    """All fixtures assume xyxy. If this flips, every box fixture must be re-examined."""
    assert BOX_FORMAT == 'xyxy'


# ---------------------------------------------------------------------------
# §6.1 — calculate_box_ious
# ---------------------------------------------------------------------------

def test_box_ious_identical_boxes():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    iou = calculate_box_ious(boxes, boxes, box_format='xyxy')
    assert iou.shape == (2, 2)
    assert np.allclose(np.diag(iou), 1.0)


def test_box_ious_non_overlapping():
    a = np.array([[0, 0, 1, 1]], dtype=float)
    b = np.array([[10, 10, 11, 11]], dtype=float)
    iou = calculate_box_ious(a, b, box_format='xyxy')
    assert iou[0, 0] == 0.0


def test_box_ious_partial_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 5, 15, 15]], dtype=float)
    iou = calculate_box_ious(a, b, box_format='xyxy')
    # Overlap area = 5*5 = 25; union = 100+100-25 = 175
    assert np.isclose(iou[0, 0], 25 / 175)


def test_box_ious_xywh_format():
    a_xyxy = np.array([[0, 0, 10, 10]], dtype=float)
    b_xyxy = np.array([[5, 5, 15, 15]], dtype=float)
    a_xywh = np.array([[0, 0, 10, 10]], dtype=float)
    b_xywh = np.array([[5, 5, 10, 10]], dtype=float)
    iou_xyxy = calculate_box_ious(a_xyxy, b_xyxy, box_format='xyxy')
    iou_xywh = calculate_box_ious(a_xywh, b_xywh, box_format='xywh')
    assert np.allclose(iou_xyxy, iou_xywh)


def test_box_ious_unsupported_format_raises():
    a = np.array([[0, 0, 1, 1]], dtype=float)
    with pytest.raises(UnsupportedBoxFormatError):
        calculate_box_ious(a, a, box_format='tlbr')


def test_box_ious_batch_shape():
    a = np.random.rand(3, 4)
    a[:, 2:] += a[:, :2]
    b = np.random.rand(5, 4)
    b[:, 2:] += b[:, :2]
    iou = calculate_box_ious(a, b, box_format='xyxy')
    assert iou.shape == (3, 5)


def test_box_ious_empty_inputs():
    a = np.zeros((0, 4))
    b = np.array([[0, 0, 1, 1]], dtype=float)
    assert calculate_box_ious(a, b, box_format='xyxy').shape == (0, 1)
    assert calculate_box_ious(b, a, box_format='xyxy').shape == (1, 0)


def test_box_ious_alias_x0y0x1y1():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    iou_alias = calculate_box_ious(a, a, box_format='x0y0x1y1')
    iou_xyxy = calculate_box_ious(a, a, box_format='xyxy')
    assert np.allclose(iou_alias, iou_xyxy)


def test_box_ious_degenerate_box_documents_behavior():
    """x2 < x1 → negative width clipped to 0 by np.maximum(0, right-left) → IoU 0."""
    a = np.array([[10, 0, 0, 10]], dtype=float)  # x2 < x1
    b = np.array([[0, 0, 5, 5]], dtype=float)
    iou = calculate_box_ious(a, b, box_format='xyxy')
    # The function returns 0 because the intersection clamps to 0. Lock that.
    assert iou[0, 0] == 0.0


# ---------------------------------------------------------------------------
# §6.2 — calculate_latlonalt_l2 / calculate_latlon_l2
# ---------------------------------------------------------------------------

def test_latlonalt_same_point_similarity_one():
    p = np.array([[40.0, -80.0, 100.0]])
    sim = calculate_latlonalt_l2(p, p)
    assert np.isclose(sim[0, 0], 1.0)


def test_latlonalt_known_distance_about_10m_alt():
    """Two points differing in altitude by 10 m → similarity ≈ exp(-1)."""
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 10.0]])
    sim = calculate_latlonalt_l2(a, b)
    assert np.isclose(sim[0, 0], np.exp(-10 / ECEF_L2_DECAY_METERS), atol=1e-3)


def test_latlonalt_output_shape():
    a = np.random.rand(3, 3)
    b = np.random.rand(5, 3)
    assert calculate_latlonalt_l2(a, b).shape == (3, 5)


def test_latlon_ignores_altitude():
    """Same lat/lon → distance 0 → similarity 1, regardless of any extra columns."""
    a = np.array([[40.0, -80.0]])
    b = np.array([[40.0, -80.0]])
    sim = calculate_latlon_l2(a, b)
    assert np.isclose(sim[0, 0], 1.0)


def test_latlon_empty_inputs():
    a = np.zeros((0, 2))
    b = np.array([[40.0, -80.0]])
    assert calculate_latlon_l2(a, b).shape == (0, 1)


# ---------------------------------------------------------------------------
# §6.3 — normalize_cost_matrix
# ---------------------------------------------------------------------------

def test_normalize_all_ones_3x3():
    """All-ones 3x3: each output element = 1/(3+3-1) = 0.2"""
    m = np.ones((3, 3))
    out = normalize_cost_matrix(m)
    assert np.allclose(out, 0.2)


def test_normalize_all_ones_rectangular():
    """All-ones MxN: each output element = 1/(M+N-1)"""
    m = np.ones((2, 4))
    out = normalize_cost_matrix(m)
    assert np.allclose(out, 1 / (2 + 4 - 1))


def test_normalize_all_zeros():
    """All-zero input: denom=0 → division-where guard keeps output at 0."""
    m = np.zeros((3, 3))
    out = normalize_cost_matrix(m)
    assert np.all(out == 0)


def test_normalize_output_range_for_nonnegative():
    np.random.seed(0)
    m = np.random.rand(4, 5)
    out = normalize_cost_matrix(m)
    assert (out >= 0).all() and (out <= 1).all()


def test_normalize_preserves_zero_entries():
    m = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = normalize_cost_matrix(m)
    assert out[0, 1] == 0.0 and out[1, 0] == 0.0


# ---------------------------------------------------------------------------
# §6.4 — merge_hota_data
# ---------------------------------------------------------------------------

def _populated_hota(video_id="vid", frame=0, tp_extra=0):
    """Build a populated HOTAData with TP=1+tp_extra (just for arithmetic checks)."""
    i_ids = np.array([1], dtype=np.int64)
    j_ids = np.array([10], dtype=np.int64)
    cost = np.array([[1.0]])
    cm = CostMatrixDataFrame(
        i_ids=i_ids, j_ids=j_ids, i_hashes=None, j_hashes=None,
        cost_matrix=cost, video_id=video_id, frame=frame,
    )
    cfg = HOTAConfig(
        iou_thresholds=np.array([0.5]),
        reference_contains_dense_annotations=True,
    )
    h = HOTAData(sim_cost_matrix=cm, gt_to_tracker_id_map={1: 10}, config=cfg)
    if tp_extra:
        h.metrics.tp += tp_extra
    return h


def test_merge_single_item_finalized():
    h = _populated_hota(video_id="vid_a", frame=0)
    out = merge_hota_data([h])
    # frame is wiped to None by merge
    assert out.frame is None
    # tp/fp/fn carried through
    assert out.metrics.tp[0] == h.metrics.tp[0]


def test_merge_video_id_propagated_from_first_item():
    """Locks the *current* (buggy) leak — see test_known_bugs.test_merge_resets_video_id."""
    h1 = _populated_hota(video_id="vid_a")
    h2 = _populated_hota(video_id="vid_b")
    out = merge_hota_data([h1, h2])
    assert out.video_id == "vid_a"  # leaks; reid_hota.py wipes it after


def test_merge_none_video_id_raises():
    h1 = _populated_hota(video_id="vid_a")
    h2 = _populated_hota(video_id="vid_b")
    h2.video_id = None
    with pytest.raises(MissingVideoIDError):
        merge_hota_data([h1, h2])


def test_merge_tp_counts_add():
    h1 = _populated_hota(video_id="vid_a")
    h2 = _populated_hota(video_id="vid_b")
    expected_tp = h1.metrics.tp[0] + h2.metrics.tp[0]
    out = merge_hota_data([h1, h2])
    assert out.metrics.tp[0] == expected_tp


def test_merge_empty_list_no_config_uses_defaults():
    """No config passed → HOTAData built with default HOTAConfig (9 thresholds)."""
    out = merge_hota_data([])
    assert len(out.iou_thresholds) == 9
    assert np.all(out.metrics.tp == 0)


def test_merge_empty_list_respects_caller_config():
    """Empty placeholder must honor the caller's iou_thresholds, not silently pick its own.

    Regression for the bug previously documented as B2 in test_known_bugs.py: when
    a pipeline configured with N thresholds receives no per-video data, the merged
    global object must still have N-shaped metric arrays so downstream aggregation
    (e.g. __iadd__) stays shape-consistent.
    """
    cfg = HOTAConfig(iou_thresholds=np.array([0.3, 0.7]))
    out = merge_hota_data([], config=cfg)
    assert np.array_equal(out.iou_thresholds, np.array([0.3, 0.7]))
    assert out.metrics.tp.shape == (2,)


# ---------------------------------------------------------------------------
# §6.5 — HOTAData.is_equal
# ---------------------------------------------------------------------------

def test_is_equal_self():
    h = _populated_hota()
    assert h.is_equal(h)


def test_is_equal_after_deepcopy():
    h = _populated_hota()
    assert copy.deepcopy(h).is_equal(h)


def test_not_equal_different_tp():
    a = _populated_hota()
    b = _populated_hota()
    b.metrics.tp = b.metrics.tp + 1
    assert not a.is_equal(b)


def test_not_equal_different_video_id():
    a = _populated_hota(video_id="x")
    b = _populated_hota(video_id="y")
    assert not a.is_equal(b)


def test_not_equal_different_frame():
    a = _populated_hota(frame=0)
    b = _populated_hota(frame=1)
    assert not a.is_equal(b)
