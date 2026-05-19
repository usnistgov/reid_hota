"""
Tests for HOTAConfig options: dense/sparse, class_ids, gids, box-hash tracking, IoU threshold cardinality.
"""
import numpy as np
import pytest
from conftest import make_df

from reid_hota import HOTAConfig, HOTAReIDEvaluator


def _cfg(**overrides):
    base = dict(
        iou_thresholds=np.array([0.5]),
        suppress_print_statements=True,
    )
    base.update(overrides)
    return HOTAConfig(**base)


# ---------------------------------------------------------------------------
# §4.1 — reference_contains_dense_annotations
# ---------------------------------------------------------------------------

def _build_one_ref_n_confuser_data(n_confusers: int, n_frames: int):
    """Ref: one tracked object. Comp: that object + N confusers in each frame."""
    ref_rows = []
    comp_rows = []
    for f in range(n_frames):
        ref_rows.append({'frame': f, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1})
        comp_rows.append({'frame': f, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1})
        for c in range(n_confusers):
            # well-separated confusers
            x = 100 + c * 10
            comp_rows.append({
                'frame': f, 'id': 1000 + c,
                'x1': x, 'y1': 0, 'x2': x + 1, 'y2': 1,
            })
    return ref_rows, comp_rows


def test_dense_vs_sparse_confuser_handling():
    n_confusers, n_frames = 2, 3
    ref_rows, comp_rows = _build_one_ref_n_confuser_data(n_confusers, n_frames)
    ref_dfs = {'v': make_df(ref_rows)}
    comp_dfs = {'v': make_df(comp_rows)}

    dense = HOTAReIDEvaluator(n_workers=0, config=_cfg(reference_contains_dense_annotations=True))
    dense.evaluate(ref_dfs, comp_dfs)
    dense_out = dense.get_global_hota_data()

    sparse = HOTAReIDEvaluator(n_workers=0, config=_cfg(reference_contains_dense_annotations=False))
    sparse.evaluate(ref_dfs, comp_dfs)
    sparse_out = sparse.get_global_hota_data()

    # Dense: every confuser inflates FP. Sparse: confusers land in UnmatchedFP.
    assert dense_out['FP'][0] > sparse_out['FP'][0]
    assert sparse_out['UnmatchedFP'] > 0
    # Per-frame-total semantics (see test_known_bugs B6): n_confusers * n_frames
    assert sparse_out['UnmatchedFP'] == n_confusers * n_frames


# ---------------------------------------------------------------------------
# §4.2 — class_ids filtering
# ---------------------------------------------------------------------------

def _two_class_data():
    ref_rows = [
        {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'object_type': 1},
        {'frame': 0, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6, 'object_type': 2},
    ]
    comp_rows = [
        {'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'object_type': 1},
        {'frame': 0, 'id': 200, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6, 'object_type': 2},
    ]
    return ref_rows, comp_rows


@pytest.mark.parametrize("class_ids,expected_tp", [
    (None, 2),
    ([1], 1),
    ([2], 1),
    ([1, 2], 2),
    ([99], 0),
    ([], 0),
])
def test_class_ids_filtering(class_ids, expected_tp):
    ref_rows, comp_rows = _two_class_data()
    ev = HOTAReIDEvaluator(n_workers=0, config=_cfg(class_ids=class_ids))
    ev.evaluate(
        {'v': make_df(ref_rows)},
        {'v': make_df(comp_rows)},
    )
    out = ev.get_global_hota_data()
    assert out['TP'][0] == expected_tp


def test_class_ids_none_with_missing_object_type_column():
    """When class_ids=None, the object_type column is no longer required."""
    rows = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    df = make_df(rows)
    df = df.drop(columns=['object_type'])  # column genuinely absent

    cfg = _cfg(class_ids=None, reference_contains_dense_annotations=True)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': df}, {'v': df.copy()})  # no crash
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 1


# ---------------------------------------------------------------------------
# §4.3 — gids filtering
# ---------------------------------------------------------------------------

def test_gids_filtering_keeps_only_specified_ids():
    """Ref has 3 ids tracked perfectly. gids=[1,2] → only those count as TP."""
    ref_rows, comp_rows = [], []
    for f in range(2):
        ref_rows.append({'frame': f, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1})
        ref_rows.append({'frame': f, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6})
        ref_rows.append({'frame': f, 'id': 3, 'x1': 10, 'y1': 10, 'x2': 11, 'y2': 11})
        comp_rows.append({'frame': f, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1})
        comp_rows.append({'frame': f, 'id': 200, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6})
        comp_rows.append({'frame': f, 'id': 300, 'x1': 10, 'y1': 10, 'x2': 11, 'y2': 11})

    cfg = _cfg(gids=[1, 2], reference_contains_dense_annotations=True)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': make_df(ref_rows)}, {'v': make_df(comp_rows)})
    out = ev.get_global_hota_data()
    # Two frames × two surviving ids = 4 TP
    assert out['TP'][0] == 4


@pytest.mark.parametrize("gids", [None, []])
def test_gids_none_and_empty_equivalent(gids):
    """In _populate, `gids is not None and len(gids) > 0` — empty list ≡ None."""
    rows = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    df = make_df(rows)
    cfg = _cfg(gids=gids, reference_contains_dense_annotations=True)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': df}, {'v': df.copy()})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 1  # no filtering applied


# ---------------------------------------------------------------------------
# §4.4 — track_fp_fn_tp_box_hashes
# ---------------------------------------------------------------------------

def test_box_hash_tracking_populates_keys():
    rows_ref = [
        {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ]
    rows_comp = [
        {'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ]
    cfg = _cfg(track_fp_fn_tp_box_hashes=True, reference_contains_dense_annotations=True)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(
        {'v': make_df(rows_ref, include_box_hash=True)},
        {'v': make_df(rows_comp, include_box_hash=True)},
    )
    per_video = ev.get_per_video_hota_data()
    v = per_video['v']
    assert 'TP_hashes' in v
    assert 'FP_hashes' in v
    assert 'FN_hashes' in v
    # TP_hashes is a list (one per threshold) of sets; check at least one is non-empty
    assert any(len(s) > 0 for s in v['TP_hashes'])
    # Hash entries are strings (per fixture-computed _compute_box_hash)
    flat = {h for s in v['TP_hashes'] for h in s}
    assert all(isinstance(h, str) for h in flat)


def test_box_hash_tracking_requires_box_hash_column():
    """track_fp_fn_tp_box_hashes=True without a box_hash column → ValueError."""
    rows = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    df = make_df(rows)  # no box_hash column
    cfg = _cfg(track_fp_fn_tp_box_hashes=True)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    with pytest.raises(ValueError, match="box_hash"):
        ev.evaluate({'v': df}, {'v': df.copy()})


# ---------------------------------------------------------------------------
# §4.5 / §4.6 — IoU threshold cardinality and monotonicity
# ---------------------------------------------------------------------------

def test_single_iou_threshold_metric_length():
    rows = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    df = make_df(rows)
    cfg = _cfg(iou_thresholds=np.array([0.5]), reference_contains_dense_annotations=True)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': df}, {'v': df.copy()})
    out = ev.get_global_hota_data()
    assert len(out['HOTA']) == 1
    assert len(out['TP']) == 1
    assert len(out['FP']) == 1


def test_many_iou_thresholds_hota_monotonically_non_increasing():
    """At successively stricter thresholds, fewer matches → HOTA non-increasing."""
    # 4 boxes with varied IoU
    ref_rows = []
    comp_rows = []
    for f, shift in enumerate([0, 1, 3, 5]):
        ref_rows.append({'frame': f, 'id': f + 1, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10})
        comp_rows.append({
            'frame': f, 'id': (f + 1) * 100,
            'x1': shift, 'y1': 0, 'x2': shift + 10, 'y2': 10,
        })
    cfg = _cfg(
        iou_thresholds=np.linspace(0.1, 0.9, 20),
        reference_contains_dense_annotations=True,
    )
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': make_df(ref_rows)}, {'v': make_df(comp_rows)})
    out = ev.get_global_hota_data()
    assert len(out['HOTA']) == 20
    # TP is non-increasing as threshold rises
    tp = out['TP']
    assert all(tp[i] >= tp[i + 1] for i in range(len(tp) - 1))


def test_iou_threshold_zero_matches_everything():
    rows_ref = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}]
    # comp box far away — IoU = 0
    rows_comp = [{'frame': 0, 'id': 100, 'x1': 100, 'y1': 100, 'x2': 101, 'y2': 101}]
    cfg = _cfg(
        iou_thresholds=np.array([0.0]),
        reference_contains_dense_annotations=True,
    )
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': make_df(rows_ref)}, {'v': make_df(rows_comp)})
    out = ev.get_global_hota_data()
    # threshold = 0 → even IoU=0 satisfies sim >= 0 - eps
    assert out['TP'][0] == 1


def test_iou_threshold_one_requires_exact_match():
    """At threshold 1.0 — only sim >= 1 - eps counts. Identical boxes pass; near-misses don't."""
    # Identical boxes → IoU=1
    rows_ref_exact = [{'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}]
    rows_comp_exact = [{'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}]
    cfg = _cfg(iou_thresholds=np.array([1.0]), reference_contains_dense_annotations=True)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': make_df(rows_ref_exact)}, {'v': make_df(rows_comp_exact)})
    assert ev.get_global_hota_data()['TP'][0] == 1

    # Near-miss — slightly shifted box
    rows_comp_near = [{'frame': 0, 'id': 100, 'x1': 1, 'y1': 0, 'x2': 11, 'y2': 10}]
    ev2 = HOTAReIDEvaluator(n_workers=0, config=_cfg(iou_thresholds=np.array([1.0]),
                                                     reference_contains_dense_annotations=True))
    ev2.evaluate({'v': make_df(rows_ref_exact)}, {'v': make_df(rows_comp_near)})
    assert ev2.get_global_hota_data()['TP'][0] == 0
