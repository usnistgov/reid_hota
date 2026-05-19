"""
Export / output API surface tests.

Locks the current shape of the parquet outputs (one row per video / frame /
global, each metric cell holding an object-typed numpy array) so any change
to the export layout is intentional.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from reid_hota import HOTAConfig, HOTAReIDEvaluator
from test_utils import load_hota_results, save_hota_results

from conftest import make_df


def _basic_cfg(**overrides):
    base = dict(
        reference_contains_dense_annotations=True,
        iou_thresholds=np.array([0.5]),
        suppress_print_statements=True,
    )
    base.update(overrides)
    return HOTAConfig(**base)


def _eval_simple():
    """Tiny two-video, two-frame fixture — enough to populate every output dict."""
    rows = lambda offset: [
        {'frame': 0, 'id': 1 + offset, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
        {'frame': 1, 'id': 1 + offset, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1},
    ]
    ref_dfs = {'va': make_df(rows(0)), 'vb': make_df(rows(10))}
    comp_dfs = {'va': make_df(rows(0)), 'vb': make_df(rows(10))}
    ev = HOTAReIDEvaluator(n_workers=0, config=_basic_cfg())
    ev.evaluate(ref_dfs, comp_dfs)
    return ev


# ---------------------------------------------------------------------------
# §8.1 — export_to_file writes the expected files
# ---------------------------------------------------------------------------

def test_export_writes_all_files(tmp_path):
    ev = _eval_simple()
    ev.export_to_file(str(tmp_path))

    for fname in ('hota.parquet', 'hota.csv', 'hota_per_video.parquet', 'hota_per_frame.parquet'):
        assert (tmp_path / fname).exists(), f"missing {fname}"

    # global parquet round-trips and has the metric columns
    df_global = pd.read_parquet(tmp_path / 'hota.parquet')
    for key in ('HOTA', 'DetA', 'AssA', 'TP', 'FP', 'FN'):
        assert key in df_global.columns

    # per-video parquet has one row per video
    df_pv = pd.read_parquet(tmp_path / 'hota_per_video.parquet')
    assert len(df_pv) == 2
    assert set(df_pv['video_id']) == {'va', 'vb'}


# ---------------------------------------------------------------------------
# §8.2 / §8.3 — save_per_frame / save_per_video flags
# ---------------------------------------------------------------------------

def test_export_save_per_frame_false_omits_frame_file(tmp_path):
    ev = _eval_simple()
    ev.export_to_file(str(tmp_path), save_per_frame=False)
    assert not (tmp_path / 'hota_per_frame.parquet').exists()
    assert (tmp_path / 'hota_per_video.parquet').exists()
    assert (tmp_path / 'hota.parquet').exists()


def test_export_save_per_video_false_omits_video_file(tmp_path):
    ev = _eval_simple()
    ev.export_to_file(str(tmp_path), save_per_video=False)
    assert not (tmp_path / 'hota_per_video.parquet').exists()
    assert (tmp_path / 'hota_per_frame.parquet').exists()
    assert (tmp_path / 'hota.parquet').exists()


# ---------------------------------------------------------------------------
# §8.4 — export before evaluate() warns and returns
# ---------------------------------------------------------------------------

def test_export_before_evaluate_warns(tmp_path):
    ev = HOTAReIDEvaluator(n_workers=0, config=_basic_cfg())
    # ev.global_hota_data is None until evaluate() runs
    with pytest.warns(UserWarning):
        ev.export_to_file(str(tmp_path))
    # nothing should have been written
    assert not (tmp_path / 'hota.parquet').exists()


# ---------------------------------------------------------------------------
# §8.5 — get_results() consistency with the three individual getters
# ---------------------------------------------------------------------------

def test_get_results_consistency():
    ev = _eval_simple()
    g, pv, pf = ev.get_results()

    g2 = ev.get_global_hota_data()
    for key in g:
        if isinstance(g[key], np.ndarray):
            assert np.array_equal(g[key], g2[key])
        else:
            assert g[key] == g2[key]

    assert set(pv.keys()) == set(ev.get_per_video_hota_data().keys())
    assert set(pf.keys()) == set(ev.get_per_frame_hota_data().keys())


# ---------------------------------------------------------------------------
# §8.6 — per_frame_hota_data is dict[video_id, dict[frame_int, dict]]
# ---------------------------------------------------------------------------

def test_per_frame_keys_match_input_frames():
    ev = _eval_simple()
    pf = ev.get_per_frame_hota_data()
    for vid in ('va', 'vb'):
        assert vid in pf
        assert set(pf[vid].keys()) == {0, 1}, f"video {vid} frame keys: {pf[vid].keys()}"


# ---------------------------------------------------------------------------
# §8.7 — JSON round-trip via save_hota_results / load_hota_results
# ---------------------------------------------------------------------------

def test_json_round_trip_preserves_numeric_values(tmp_path):
    ev = _eval_simple()
    fp = str(tmp_path / 'global.json')
    save_hota_results(ev.global_hota_data, fp)
    loaded = load_hota_results(fp)

    g = ev.get_global_hota_data()
    for key in ('HOTA', 'DetA', 'AssA', 'TP', 'FP', 'FN'):
        original = g[key]
        if isinstance(original, np.ndarray):
            assert np.allclose(np.asarray(loaded[key], dtype=float), original, atol=1e-6)
        else:
            assert loaded[key] == original


def test_json_round_trip_drops_hash_keys(tmp_path):
    ev = _eval_simple()
    fp = str(tmp_path / 'global.json')
    save_hota_results(ev.global_hota_data, fp)
    with open(fp) as f:
        on_disk = json.load(f)
    # save_hota_results explicitly strips 'hashes' / 'counts' keys
    assert not any('hashes' in k for k in on_disk)
    assert not any('counts' in k for k in on_disk)
