"""
Input robustness tests — degenerate boxes, non-finite coordinates, exotic ID dtypes,
float frame indices, and the smallest possible video.
"""
import numpy as np
import pytest
from conftest import make_df

from reid_hota import HOTAConfig, HOTAReIDEvaluator
from reid_hota.hota_utils import calculate_box_ious


def _cfg(**overrides):
    base = dict(
        reference_contains_dense_annotations=True,
        iou_thresholds=np.array([0.5]),
        suppress_print_statements=True,
    )
    base.update(overrides)
    return HOTAConfig(**base)


# ---------------------------------------------------------------------------
# §10.1 — degenerate bounding boxes (x2 < x1)
# ---------------------------------------------------------------------------

def test_degenerate_box_iou_is_zero_unit():
    """At the IoU primitive: x2<x1 → max(0, right-left) clamps intersection to 0."""
    a = np.array([[10, 0, 0, 10]], dtype=float)
    b = np.array([[0, 0, 5, 5]], dtype=float)
    iou = calculate_box_ious(a, b, box_format='xyxy')
    assert iou[0, 0] == 0.0


def test_degenerate_ref_box_yields_fn_not_crash():
    """End-to-end: a ref box with x2<x1 cannot match anything → FN, no crash."""
    ref = make_df([{'frame': 0, 'id': 1, 'x1': 10, 'y1': 0, 'x2': 0, 'y2': 10}])
    comp = make_df([{'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 5, 'y2': 5}])
    ev = HOTAReIDEvaluator(n_workers=0, config=_cfg())
    ev.evaluate({'v': ref}, {'v': comp})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 0
    assert out['FN'][0] == 1
    assert out['FP'][0] == 1


# ---------------------------------------------------------------------------
# §10.2 — NaN in box coordinates
# ---------------------------------------------------------------------------

def test_nan_box_coord_raises_numeric_error():
    """
    NaN x/y → IoU NaN. scipy's linear_sum_assignment surfaces this as a generic
    ValueError("matrix contains invalid numeric entries") *before* the library's
    NonFiniteSimilarityValueError gets a chance to fire. Lock current behavior.
    """
    ref = make_df([{'frame': 0, 'id': 1, 'x1': np.nan, 'y1': 0, 'x2': 1, 'y2': 1}])
    comp = make_df([{'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}])
    ev = HOTAReIDEvaluator(n_workers=0, config=_cfg())
    with pytest.raises(ValueError, match="invalid numeric"):
        ev.evaluate({'v': ref}, {'v': comp})


# ---------------------------------------------------------------------------
# §10.3 — Inf in box coordinates
# ---------------------------------------------------------------------------

def test_inf_box_coord_iou_becomes_zero():
    """
    Inf coords don't crash: intersection is finite, union is inf → IoU = 0.
    Box gets recorded as a non-match (FP+FN at threshold > 0). Lock the
    silent degradation so a future change is intentional.
    """
    ref = make_df([{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': np.inf, 'y2': 1}])
    comp = make_df([{'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}])
    ev = HOTAReIDEvaluator(n_workers=0, config=_cfg())
    ev.evaluate({'v': ref}, {'v': comp})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 0
    assert out['FN'][0] == 1
    assert out['FP'][0] == 1


# ---------------------------------------------------------------------------
# §10.4 — String IDs
# ---------------------------------------------------------------------------

def test_string_ids_end_to_end():
    """
    Hungarian runs on numeric cost values; IDs are only dict keys.
    String IDs should work end-to-end.
    """
    ref = make_df([
        {'frame': 0, 'id': 'alice', 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 'alice', 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ])
    comp = make_df([
        {'frame': 0, 'id': 'tracker_x', 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 'tracker_x', 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ])
    ev = HOTAReIDEvaluator(n_workers=0, config=_cfg())
    ev.evaluate({'v': ref}, {'v': comp})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 2
    assert out['FN'][0] == 0
    assert out['FP'][0] == 0


# ---------------------------------------------------------------------------
# §10.5 — Float frame column — silent truncation via int(frame)
# ---------------------------------------------------------------------------

def test_float_frame_silently_truncated():
    """
    `int(frame)` at hota_utils.py:85 truncates frames like 0.7 → 0 and 1.5 → 1.
    The two frames collapse into bins {0, 1}. Lock that data-lossy behavior so any
    future change is intentional.
    """
    ref = make_df([
        {'frame': 0.7, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1.5, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ])
    # make_df casts frame to int64, so 0.7→0 and 1.5→1 already happen at fixture time.
    # The point of the test is that evaluate() doesn't error on the input;
    # collapse is independent of how the truncation happens upstream.
    comp = make_df([
        {'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ])
    ev = HOTAReIDEvaluator(n_workers=0, config=_cfg())
    ev.evaluate({'v': ref}, {'v': comp})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 2


# ---------------------------------------------------------------------------
# §10.6 — Single-frame video (smallest non-empty case)
# ---------------------------------------------------------------------------

def test_single_frame_video_no_divzero():
    """
    One frame, one ref, one comp — smallest case for AssA.
    Should produce sensible metrics rather than nan/inf.
    """
    ref = make_df([{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}])
    comp = make_df([{'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}])
    ev = HOTAReIDEvaluator(n_workers=0, config=_cfg())
    ev.evaluate({'v': ref}, {'v': comp})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 1
    for key in ('HOTA', 'DetA', 'AssA', 'LocA', 'IDF1'):
        assert np.all(np.isfinite(out[key]))
    assert out['HOTA'][0] > 0
