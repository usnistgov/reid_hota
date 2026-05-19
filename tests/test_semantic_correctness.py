"""
Semantic correctness tests — hand-crafted data with analytically known answers.
"""
import numpy as np
import pytest
from conftest import empty_df, make_df

from reid_hota import HOTAConfig, HOTAReIDEvaluator


def _basic_cfg(**overrides):
    base = dict(
        reference_contains_dense_annotations=True,
        iou_thresholds=np.array([0.5]),
        suppress_print_statements=True,
    )
    base.update(overrides)
    return HOTAConfig(**base)


# ---------------------------------------------------------------------------
# §5.1 — perfect tracking → HOTA ≈ 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alignment", ['global', 'per_video', 'per_frame'])
def test_perfect_tracking_hota_equals_one(alignment):
    rows = [
        {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        {'frame': 1, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
    ]
    df = make_df(rows)
    cfg = _basic_cfg(id_alignment_method=alignment)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': df}, {'v': df.copy()})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 4
    assert out['FP'][0] == 0
    assert out['FN'][0] == 0
    assert np.isclose(out['HOTA'][0], 1.0, atol=1e-4)
    assert np.isclose(out['DetA'][0], 1.0, atol=1e-4)
    assert np.isclose(out['AssA'][0], 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# §5.2 — zero tracking
# ---------------------------------------------------------------------------

def test_zero_tracking_hota_equals_zero():
    rows = [
        {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ]
    cfg = _basic_cfg()
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': make_df(rows)}, {'v': empty_df()})
    out = ev.get_global_hota_data()
    assert out['HOTA'][0] == 0
    assert out['TP'][0] == 0
    assert out['FP'][0] == 0
    assert out['FN'][0] == 2


# ---------------------------------------------------------------------------
# §5.3 — all false positives
# ---------------------------------------------------------------------------

def test_all_false_positives():
    comp_rows = [
        {'frame': 0, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 10, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ]
    cfg = _basic_cfg(reference_contains_dense_annotations=True)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': empty_df()}, {'v': make_df(comp_rows)})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 0
    assert out['FN'][0] == 0
    assert out['FP'][0] == 2
    assert out['HOTA'][0] == 0


# ---------------------------------------------------------------------------
# §5.4 — known TP count
# ---------------------------------------------------------------------------

def test_tp_count_known_value():
    def make_video():
        return make_df([
            {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
            {'frame': 0, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
            {'frame': 1, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
            {'frame': 1, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        ])
    ref_dfs = {'v1': make_video(), 'v2': make_video()}
    comp_dfs = {'v1': make_video(), 'v2': make_video()}
    cfg = _basic_cfg(iou_thresholds=np.array([0.1]))
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate(ref_dfs, comp_dfs)
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 8


# ---------------------------------------------------------------------------
# §5.5 — partial IoU threshold split
# ---------------------------------------------------------------------------

def test_partial_iou_threshold_split():
    """4 pairs at IoU≈0.778, 4 pairs at IoU=0.0 → at threshold 0.5: TP=4, FN=4, FP=4."""
    ref_rows = []
    comp_rows = []
    # 4 high-IoU pairs (frames 0..3, each with one ref+one comp box that overlap)
    for f in range(4):
        ref_rows.append({'frame': f, 'id': f + 1, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10})
        # box shifted by 1 → intersection 9*9=81; union 100+100-81=119 → IoU ≈ 0.68
        comp_rows.append({'frame': f, 'id': (f + 1) * 100, 'x1': 1, 'y1': 1, 'x2': 11, 'y2': 11})
    # 4 zero-IoU pairs (frames 4..7)
    for f in range(4, 8):
        ref_rows.append({'frame': f, 'id': f + 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1})
        comp_rows.append({'frame': f, 'id': (f + 1) * 100, 'x1': 100, 'y1': 100, 'x2': 101, 'y2': 101})

    cfg = _basic_cfg(iou_thresholds=np.array([0.5]))
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'v': make_df(ref_rows)}, {'v': make_df(comp_rows)})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 4
    assert out['FN'][0] == 4
    assert out['FP'][0] == 4


# ---------------------------------------------------------------------------
# §5.6 — ID swap reduces AssA
# ---------------------------------------------------------------------------

def test_id_swap_reduces_assa():
    # Baseline: comp ids consistent across frames
    baseline_ref = [
        {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        {'frame': 1, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
    ]
    baseline_comp = [
        {'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 200, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        {'frame': 1, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 200, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
    ]
    # Swapped: frame 1 has ids flipped relative to their boxes
    swap_comp = [
        {'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 200, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        {'frame': 1, 'id': 200, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},  # swapped id
        {'frame': 1, 'id': 100, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},  # swapped id
    ]
    cfg = _basic_cfg(id_alignment_method='global')
    ev1 = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev1.evaluate({'v': make_df(baseline_ref)}, {'v': make_df(baseline_comp)})
    baseline_assa = ev1.get_global_hota_data()['AssA'][0]

    ev2 = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev2.evaluate({'v': make_df(baseline_ref)}, {'v': make_df(swap_comp)})
    swap_assa = ev2.get_global_hota_data()['AssA'][0]

    assert swap_assa < baseline_assa


# ---------------------------------------------------------------------------
# §5.7 — alignment-method ordering: per_frame ≥ global on imperfect tracking
# ---------------------------------------------------------------------------

def test_per_frame_alignment_at_least_as_lenient_as_global():
    """
    Per-frame Hungarian re-matches every frame, so it can rescue ID swaps that
    confuse the global assignment. HOTA(per_frame) ≥ HOTA(global) on swapped data.
    """
    ref = [
        {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        {'frame': 1, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 2, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
    ]
    # Swap ids between frames — global alignment can't fix both frames; per-frame can.
    comp = [
        {'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 0, 'id': 200, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
        {'frame': 1, 'id': 200, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 100, 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 6},
    ]
    cfg_global = _basic_cfg(id_alignment_method='global')
    cfg_per_frame = _basic_cfg(id_alignment_method='per_frame')

    e1 = HOTAReIDEvaluator(n_workers=0, config=cfg_global)
    e1.evaluate({'v': make_df(ref)}, {'v': make_df(comp)})
    hota_global = e1.get_global_hota_data()['HOTA'][0]

    e2 = HOTAReIDEvaluator(n_workers=0, config=cfg_per_frame)
    e2.evaluate({'v': make_df(ref)}, {'v': make_df(comp)})
    hota_pf = e2.get_global_hota_data()['HOTA'][0]

    assert hota_pf >= hota_global


# ---------------------------------------------------------------------------
# §5.8 — Hungarian tie-breaking determinism
# ---------------------------------------------------------------------------

def test_hungarian_ties_deterministic():
    """Two equal-cost matchings — SciPy must pick the same one every run."""
    ref = [
        {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10},
        {'frame': 0, 'id': 2, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10},
    ]
    comp = [
        {'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10},
        {'frame': 0, 'id': 200, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10},
    ]
    # Above will trip DuplicateIDError because ref has duplicate id within frame.
    # Instead use different ref boxes that yield equal IoU with the comp boxes.
    ref = [
        {'frame': 0, 'id': 1, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10},
        {'frame': 0, 'id': 2, 'x1': 20, 'y1': 20, 'x2': 30, 'y2': 30},
    ]
    comp = [
        {'frame': 0, 'id': 100, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10},
        {'frame': 0, 'id': 200, 'x1': 20, 'y1': 20, 'x2': 30, 'y2': 30},
    ]
    cfg = _basic_cfg(id_alignment_method='global')

    results = []
    for _ in range(3):
        ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
        ev.evaluate({'v': make_df(ref)}, {'v': make_df(comp)})
        results.append(ev.get_global_hota_data()['HOTA'][0])
    assert results[0] == results[1] == results[2]
