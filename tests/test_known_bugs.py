"""
Regression tests for latent bugs uncovered during plan review.

Each xfail(strict=True) test documents a known bug at a specific source location.
When the bug is fixed, remove the xfail marker — the test then guards the fix.
"""
import numpy as np
import pytest

from reid_hota import HOTAConfig, HOTAData, HOTAReIDEvaluator
from reid_hota.cost_matrix import CostMatrixDataFrame
from reid_hota.hota_utils import merge_hota_data, normalize_cost_matrix

# Note: B1 (mutable-default config) was fixed; its tests now live in
# test_config_validation.py as positive invariants.


def _make_populated_hota(video_id="vid", frame=0, iou_thresholds=None):
    """Build a HOTAData with a single matched ref/comp pair (similarity 1.0)."""
    if iou_thresholds is None:
        iou_thresholds = np.array([0.5])
    i_ids = np.array([1], dtype=np.int64)
    j_ids = np.array([10], dtype=np.int64)
    cost_matrix = np.array([[1.0]])
    cm = CostMatrixDataFrame(
        i_ids=i_ids, j_ids=j_ids,
        i_hashes=None, j_hashes=None,
        cost_matrix=cost_matrix, video_id=video_id, frame=frame,
    )
    cfg = HOTAConfig(
        iou_thresholds=iou_thresholds,
        reference_contains_dense_annotations=True,
    )
    return HOTAData(sim_cost_matrix=cm, gt_to_tracker_id_map={1: 10}, config=cfg)


# Note: B2 (merge_hota_data([]) ignored caller config) was fixed by adding an
# optional `config` parameter to merge_hota_data. The positive invariant test
# now lives in test_unit_utils.py alongside the rest of the merge_hota_data
# coverage (§6.4).


# ---------------------------------------------------------------------------
# B3: merge_hota_data leaks first item's video_id
# hota_utils.py:40-41 — only `frame` is reset, `video_id` carries through deepcopy
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="merge leaks first item's video_id; caller (reid_hota.py:205) wipes it manually")
def test_merge_resets_video_id():
    h1 = _make_populated_hota(video_id="X", frame=0)
    h2 = _make_populated_hota(video_id="Y", frame=1)
    merged = merge_hota_data([h1, h2])
    assert merged.video_id is None, f"merged video_id leaked: {merged.video_id!r}"


# ---------------------------------------------------------------------------
# B4: HOTAData.is_equal silently skips non-ndarray fields
# hota_data.py:180-208 — only ndarray attributes of metrics are compared
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="is_equal does not compare unmatched_fp (a float)")
def test_is_equal_compares_unmatched_fp():
    a = _make_populated_hota()
    b = _make_populated_hota()
    b.metrics.unmatched_fp = a.metrics.unmatched_fp + 999
    assert not a.is_equal(b)


@pytest.mark.xfail(strict=True, reason="is_equal does not compare sparse_data (ref_id_counts, etc.)")
def test_is_equal_compares_sparse_data():
    a = _make_populated_hota()
    b = _make_populated_hota()
    b.sparse_data['ref_id_counts'].add_at(9999, 42)
    assert not a.is_equal(b)


@pytest.mark.xfail(strict=True, reason="is_equal does not compare iou_thresholds")
def test_is_equal_compares_iou_thresholds():
    a = _make_populated_hota(iou_thresholds=np.array([0.5]))
    b = _make_populated_hota(iou_thresholds=np.array([0.5]))
    b.iou_thresholds = np.array([0.5, 0.7])
    assert not a.is_equal(b)


# ---------------------------------------------------------------------------
# B5: normalize_cost_matrix short-circuits size-1 input unchanged
# hota_utils.py:499-501
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="1x1 input returned unchanged, including values > 1")
def test_normalize_size_one_is_normalized():
    out = normalize_cost_matrix(np.array([[5.0]]))
    assert 0.0 <= out[0, 0] <= 1.0, f"1x1 normalization produced out-of-range value: {out[0, 0]}"


# ---------------------------------------------------------------------------
# B6: unmatched_fp is a per-frame total, not a unique-ID count.
# Lock the *current* per-frame-total semantics so an accidental change is caught.
# ---------------------------------------------------------------------------

def test_unmatched_fp_is_per_frame_total():
    """Same comp ID appearing across N frames with no matching ref counts N times."""
    from conftest import make_df, empty_df
    # comp has the same id (10) in 3 frames; ref is empty
    comp = make_df([
        {'frame': 0, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 2, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ])
    ref = empty_df()
    cfg = HOTAConfig(
        reference_contains_dense_annotations=False,  # default
        iou_thresholds=np.array([0.5]),
        suppress_print_statements=True,
    )
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': ref}, {'v': comp})
    out = ev.get_global_hota_data()
    # Per-frame total: one comp id, three frames → 3 (not 1).
    assert out['UnmatchedFP'] == 3, (
        f"unmatched_fp should be a per-frame total (3), not a unique count; got {out['UnmatchedFP']}"
    )


# ---------------------------------------------------------------------------
# B7: length-1 merge_hota_data — loop body never runs
# hota_utils.py:24-51
# ---------------------------------------------------------------------------

def test_merge_length_one_matches_input():
    """merge_hota_data([single]) should produce something equivalent to the input."""
    inp = _make_populated_hota(video_id="vid", frame=0)
    out = merge_hota_data([inp])
    # frame is explicitly wiped to None by merge; align manually for the compare
    inp_copy_frame_none = _make_populated_hota(video_id="vid", frame=None)
    # is_equal compares video_id and frame, so reuse a constructed counterpart
    # with frame=None. Note: we build with frame=None to match the merge's wipe.
    assert out.is_equal(inp_copy_frame_none)


# ---------------------------------------------------------------------------
# B8: Mixed-dtype ID columns (see also test_empty_and_missing_data.py §1.10)
# ---------------------------------------------------------------------------

def test_mixed_dtype_ids_match():
    # Locks current behavior — verified that mixed int64/float64 ids match via
    # numpy/Python dict equality semantics. If this ever breaks (e.g. due to a
    # numpy 2.x dtype change), it would silently produce TP=0 in real pipelines.
    from conftest import make_df
    # ref has int64 ids; comp has float64 ids with the same numeric values
    ref = make_df([{'frame': 0, 'id': np.int64(1), 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}])
    comp = make_df([{'frame': 0, 'id': np.float64(1.0), 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}])
    cfg = HOTAConfig(
        reference_contains_dense_annotations=True,
        iou_thresholds=np.array([0.5]),
        suppress_print_statements=True,
    )
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': ref}, {'v': comp})
    out = ev.get_global_hota_data()
    # Expect a match: same numeric value, even if dtypes differ
    assert out['TP'][0] == 1, f"mixed-dtype IDs failed to match; TP={out['TP']}"
