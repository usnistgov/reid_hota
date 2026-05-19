"""
Worker-count determinism — sequential and pool paths must produce identical results.

Note: `hota_utils.py` uses `if n_workers > 1` to gate Pool creation, so
`n_workers ∈ {0, 1}` are *both* sequential. Real parallelism starts at 2.
"""
import numpy as np
import pytest

from reid_hota import HOTAConfig, HOTAReIDEvaluator


def _cfg(alignment, **overrides):
    base = dict(
        id_alignment_method=alignment,
        similarity_metric='iou',
        reference_contains_dense_annotations=True,
        suppress_print_statements=True,
    )
    base.update(overrides)
    return HOTAConfig(**base)


def _assert_results_close(a, b, atol=1e-8):
    """Compare two HOTA result dicts of numpy arrays / scalars."""
    assert set(a.keys()) == set(b.keys())
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, np.ndarray):
            assert np.allclose(va, vb, atol=atol), f"mismatch on {key}: {va} vs {vb}"
        else:
            assert va == vb, f"mismatch on {key}: {va} vs {vb}"


# ---------------------------------------------------------------------------
# §7.1 — sequential (n_workers=0) vs true-parallel (n_workers=4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alignment", ['global', 'per_video', 'per_frame'])
def test_workers_0_vs_4_global_metrics(alignment, base_tracking_data):
    """Same data, same config — n_workers=0 and n_workers=4 must agree globally."""
    ref_dfs, comp_dfs = base_tracking_data

    cfg = _cfg(alignment)
    ev_seq = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev_seq.evaluate(ref_dfs, comp_dfs)
    seq_global = ev_seq.get_global_hota_data()

    ev_par = HOTAReIDEvaluator(n_workers=4, config=cfg)
    ev_par.evaluate(ref_dfs, comp_dfs)
    par_global = ev_par.get_global_hota_data()

    _assert_results_close(seq_global, par_global)


@pytest.mark.parametrize("alignment", ['global', 'per_video', 'per_frame'])
def test_workers_0_vs_4_per_video_metrics(alignment, base_tracking_data):
    """
    Per-video agreement — catches an `imap_unordered`-style regression that
    a global-only check would miss.
    """
    ref_dfs, comp_dfs = base_tracking_data

    cfg = _cfg(alignment)
    ev_seq = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev_seq.evaluate(ref_dfs, comp_dfs)
    seq_pv = ev_seq.get_per_video_hota_data()

    ev_par = HOTAReIDEvaluator(n_workers=4, config=cfg)
    ev_par.evaluate(ref_dfs, comp_dfs)
    par_pv = ev_par.get_per_video_hota_data()

    assert set(seq_pv.keys()) == set(par_pv.keys())
    for vid in seq_pv:
        _assert_results_close(seq_pv[vid], par_pv[vid])


# ---------------------------------------------------------------------------
# §7.2 — n_workers ∈ {0, 1} are both sequential (no Pool)
# ---------------------------------------------------------------------------

def test_workers_0_and_1_are_both_sequential(base_tracking_data):
    """
    Documents that the gate is strictly `n_workers > 1`. If someone changes the
    threshold to `>= 1`, this test catches it (because n_workers=1 would then
    spawn a one-worker pool — measurable timing/behavior change, but here we
    just lock that the numeric results are identical, which is the contract).
    """
    ref_dfs, comp_dfs = base_tracking_data

    cfg = _cfg('global')
    ev_0 = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev_0.evaluate(ref_dfs, comp_dfs)

    ev_1 = HOTAReIDEvaluator(n_workers=1, config=cfg)
    ev_1.evaluate(ref_dfs, comp_dfs)

    _assert_results_close(ev_0.get_global_hota_data(), ev_1.get_global_hota_data())
