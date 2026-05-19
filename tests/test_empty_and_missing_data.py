"""
Empty / missing data edge cases.

These guard against crashes from videos that have no data on one or both sides,
missing keys in the comparison/reference dicts, and duplicate IDs within a frame.
"""
import numpy as np
import pytest
from conftest import empty_df, make_df

from reid_hota import HOTAConfig, HOTAReIDEvaluator
from reid_hota.hota_errors import DuplicateIDError


def _basic_cfg(**overrides):
    base = dict(
        reference_contains_dense_annotations=True,
        iou_thresholds=np.array([0.5]),
        suppress_print_statements=True,
    )
    base.update(overrides)
    return HOTAConfig(**base)


# ---------------------------------------------------------------------------
# §1.1 — both ref and comp empty for one video
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alignment", ['global', 'per_video', 'per_frame'])
def test_empty_video_both_sides_no_crash(alignment):
    """Video B is completely empty on both sides; should not crash."""
    rows_a = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    ref_dfs = {'vid_a': make_df(rows_a), 'vid_b': empty_df()}
    comp_dfs = {'vid_a': make_df(rows_a), 'vid_b': empty_df()}

    cfg = _basic_cfg(id_alignment_method=alignment)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)

    per_video = ev.get_per_video_hota_data()
    assert 'vid_b' in per_video
    v_b = per_video['vid_b']
    assert np.array_equal(v_b['TP'], np.zeros(1, dtype=int))
    assert np.array_equal(v_b['FP'], np.zeros(1, dtype=int))
    assert np.array_equal(v_b['FN'], np.zeros(1, dtype=int))
    assert len(v_b['TP']) == len(cfg.iou_thresholds)


# ---------------------------------------------------------------------------
# §1.2 — ref empty, comp has detections
# ---------------------------------------------------------------------------

def test_empty_ref_comp_has_detections_dense():
    comp_rows = [
        {'frame': 0, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 11, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        {'frame': 1, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ]
    ref_dfs = {'v': empty_df()}
    comp_dfs = {'v': make_df(comp_rows)}

    cfg = _basic_cfg(reference_contains_dense_annotations=True)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 0
    assert out['FN'][0] == 0
    assert out['FP'][0] == 3


def test_empty_ref_comp_has_detections_sparse():
    comp_rows = [
        {'frame': 0, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 11, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        {'frame': 1, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ]
    ref_dfs = {'v': empty_df()}
    comp_dfs = {'v': make_df(comp_rows)}

    cfg = _basic_cfg(reference_contains_dense_annotations=False)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 0
    assert out['FP'][0] == 0
    # Per-frame total of filtered comp rows: 3 comp detections across 2 frames
    assert out['UnmatchedFP'] == 3


def test_unmatched_fp_is_per_frame_total():
    """Same comp ID appearing across N frames with no matching ref counts N times.

    Locks per-frame-total semantics: UnmatchedFP counts per-frame detections, not
    unique IDs. One comp id across 3 frames → 3, not 1.
    """
    comp = make_df([
        {'frame': 0, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 2, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ])
    ev = HOTAReIDEvaluator(n_workers=0, config=_basic_cfg(reference_contains_dense_annotations=False))
    ev.evaluate({'v': empty_df()}, {'v': comp})
    out = ev.get_global_hota_data()
    assert out['UnmatchedFP'] == 3


# ---------------------------------------------------------------------------
# §1.3 — comp empty, ref has annotations
# ---------------------------------------------------------------------------

def test_comp_empty_ref_has_annotations():
    ref_rows = [
        {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        {'frame': 1, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ]
    ref_dfs = {'v': make_df(ref_rows)}
    comp_dfs = {'v': empty_df()}

    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 0
    assert out['FP'][0] == 0
    assert out['FN'][0] == 3


# ---------------------------------------------------------------------------
# §1.4 / §1.5 — video present in one dict but absent in the other
# ---------------------------------------------------------------------------

def test_video_missing_from_comp_dict():
    rows = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    ref_dfs = {'vid_a': make_df(rows), 'vid_b': make_df(rows)}
    comp_dfs = {'vid_a': make_df(rows)}  # vid_b absent

    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)
    per_video = ev.get_per_video_hota_data()
    assert {'vid_a', 'vid_b'} == set(per_video.keys())


def test_video_missing_from_ref_dict():
    rows = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    ref_dfs = {'vid_a': make_df(rows)}  # vid_b absent
    comp_dfs = {'vid_a': make_df(rows), 'vid_b': make_df(rows)}

    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)
    per_video = ev.get_per_video_hota_data()
    assert {'vid_a', 'vid_b'} == set(per_video.keys())


# ---------------------------------------------------------------------------
# §1.6 — all videos empty
# ---------------------------------------------------------------------------

def test_all_videos_empty():
    ref_dfs = {'v': empty_df()}
    comp_dfs = {'v': empty_df()}

    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 0
    assert out['FP'][0] == 0
    assert out['FN'][0] == 0
    assert out['HOTA'][0] == 0


# ---------------------------------------------------------------------------
# §1.7 — empty input dict
# ---------------------------------------------------------------------------

def test_empty_input_dict_raises():
    """Currently raises ValueError from jaccard_cost_matrices. Lock behavior."""
    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    # TODO: arguably should produce empty results instead. Document with a follow-up.
    with pytest.raises(ValueError, match="dict.*is empty"):
        ev.evaluate({}, {})


# ---------------------------------------------------------------------------
# §1.8 — duplicate ref ID in frame → DuplicateIDError
# ---------------------------------------------------------------------------

def test_duplicate_ref_id_in_frame_raises():
    ref_rows = [
        {'frame': 0, 'id': 5, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 5, 'x1': 2, 'y1': 2, 'x2': 3, 'y2': 3},  # duplicate
    ]
    comp_rows = [{'frame': 0, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    ref_dfs = {'v': make_df(ref_rows)}
    comp_dfs = {'v': make_df(comp_rows)}

    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    with pytest.raises(DuplicateIDError):
        ev.evaluate(ref_dfs, comp_dfs)


# ---------------------------------------------------------------------------
# §1.9 — duplicate comp ID in frame → Jaccard dedup, no crash
# ---------------------------------------------------------------------------

def test_duplicate_comp_id_in_frame_no_crash():
    """Duplicate comp id with *different* boxes — Jaccard dedup should handle it."""
    ref_rows = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    comp_rows = [
        {'frame': 0, 'id': 5, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 5, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},  # same id, different box
    ]
    ref_dfs = {'v': make_df(ref_rows)}
    comp_dfs = {'v': make_df(comp_rows)}

    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)  # should not raise
    out = ev.get_global_hota_data()
    assert np.isfinite(out['TP']).all()
    assert (out['TP'] >= 0).all()
    assert (out['FP'] >= 0).all()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Degenerate Jaccard: when duplicate comp ids have IDENTICAL boxes that perfectly "
        "overlap a ref box, per-frame dedup (hota_utils.py:193) produces sim>1, which the "
        "size-1 normalize_cost_matrix short-circuit (hota_utils.py:499-501) lets escape, "
        "then global jaccard divides by zero → ValueError in linear_sum_assignment."
    ),
)
def test_duplicate_comp_id_identical_boxes_no_crash():
    ref_rows = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    comp_rows = [
        {'frame': 0, 'id': 5, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 5, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ]
    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': make_df(ref_rows)}, {'v': make_df(comp_rows)})


# ---------------------------------------------------------------------------
# §1.10 — mixed-dtype id columns end-to-end
# ---------------------------------------------------------------------------

def test_mixed_dtype_id_columns_end_to_end():
    """Ref ids as int64; comp ids as float64. Numeric values match → expect TP=1."""
    ref = make_df([{'frame': 0, 'id': np.int64(1), 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}])
    comp = make_df([{'frame': 0, 'id': np.float64(1.0), 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}])

    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': ref}, {'v': comp})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 1


# ---------------------------------------------------------------------------
# §1.11 — minimal single-video / single-frame / single-object smoke
# ---------------------------------------------------------------------------

def test_single_video_single_frame_single_object():
    row = {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}
    ref_dfs = {'v': make_df([row])}
    comp_dfs = {'v': make_df([{**row, 'id': 100}])}  # different tracker id, same box

    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)
    out = ev.get_global_hota_data()
    assert out['HOTA'][0] > 0
    assert out['TP'][0] == 1
