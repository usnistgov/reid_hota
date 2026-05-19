"""
End-to-end tests for the geographic similarity metrics — 'latlonalt' and 'latlon'.

Distances are kept in the 0–30 m range so the decay curve (exp(-d/10)) is exercised,
not its asymptotic tail.
"""
import numpy as np
import pytest
from conftest import make_df

from reid_hota import HOTAConfig, HOTAReIDEvaluator
from reid_hota.hota_utils import (
    ECEF_L2_DECAY_METERS,
    calculate_latlon_l2,
    calculate_latlonalt_l2,
)


def _cfg(similarity_metric, **overrides):
    base = dict(
        similarity_metric=similarity_metric,
        iou_thresholds=np.array([0.5]),
        reference_contains_dense_annotations=True,
        suppress_print_statements=True,
    )
    base.update(overrides)
    return HOTAConfig(**base)


def _geo_row(frame, oid, lat, lon, alt=0.0):
    """One row carrying only frame/id/lat/lon/alt — no bounding box."""
    return {'frame': frame, 'id': oid, 'lat': lat, 'lon': lon, 'alt': alt}


# ---------------------------------------------------------------------------
# §3.1 / §3.2 — end-to-end smoke for latlonalt and latlon across alignments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ['latlonalt', 'latlon'])
@pytest.mark.parametrize("alignment", ['global', 'per_video', 'per_frame'])
def test_geographic_metric_end_to_end(metric, alignment):
    """2 videos, a few frames, ref ≈ comp positions within ~1 m → HOTA > 0."""
    np.random.seed(42)

    def build(prefix):
        rows_ref, rows_comp = [], []
        for f in range(3):
            for oid in (1, 2):
                base_lat = 40.0 + oid * 1e-5
                base_lon = -80.0 + oid * 1e-5
                rows_ref.append(_geo_row(f, oid, base_lat, base_lon, alt=oid * 2.0))
                # ~1 m jitter (1e-5° ≈ 1.11 m)
                jitter = 1e-6 * np.random.randn()
                rows_comp.append(_geo_row(f, oid * 100,
                                          base_lat + jitter, base_lon + jitter,
                                          alt=oid * 2.0 + 0.1 * np.random.randn()))
        return make_df(rows_ref, include_box=False, include_latlonalt=True), \
               make_df(rows_comp, include_box=False, include_latlonalt=True)

    ref_a, comp_a = build('a')
    ref_b, comp_b = build('b')

    cfg = _cfg(metric, id_alignment_method=alignment)
    ev = HOTAReIDEvaluator(n_workers=0, config=cfg)
    ev.evaluate({'va': ref_a, 'vb': ref_b}, {'va': comp_a, 'vb': comp_b})
    out = ev.get_global_hota_data()
    assert out['HOTA'][0] > 0
    assert out['TP'][0] > 0


# ---------------------------------------------------------------------------
# §3.3 — identical positions → similarity = 1, TP = 1
# ---------------------------------------------------------------------------

def test_identical_position_yields_tp():
    ref = make_df(
        [_geo_row(0, 1, 40.0, -80.0, alt=10.0)],
        include_box=False, include_latlonalt=True,
    )
    comp = make_df(
        [_geo_row(0, 100, 40.0, -80.0, alt=10.0)],
        include_box=False, include_latlonalt=True,
    )
    ev = HOTAReIDEvaluator(n_workers=0, config=_cfg('latlonalt'))
    ev.evaluate({'v': ref}, {'v': comp})
    out = ev.get_global_hota_data()
    assert out['TP'][0] == 1
    assert out['FP'][0] == 0
    assert out['FN'][0] == 0


# ---------------------------------------------------------------------------
# §3.4 — decay curve sanity at known meter offsets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_m", [0.0, 5.0, 10.0, 20.0, 30.0])
def test_decay_curve_alt_axis(dist_m):
    """Altitude separation in meters → similarity = exp(-d/decay) within 1%."""
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, dist_m]])
    sim = calculate_latlonalt_l2(a, b)[0, 0]
    expected = np.exp(-dist_m / ECEF_L2_DECAY_METERS)
    if expected > 0:
        assert abs(sim - expected) / expected < 0.01
    else:
        assert sim == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# §3.5 — latlonalt vs latlon — altitude makes the difference
# ---------------------------------------------------------------------------

def test_altitude_difference_blocks_latlonalt_match_only():
    """
    Same lat/lon, 15 m altitude gap.
    latlonalt: sim ≈ exp(-1.5) ≈ 0.223 → below 0.5 threshold → 0 TP
    latlon:    altitude ignored → sim = 1.0 → 1 TP
    """
    ref = make_df(
        [_geo_row(0, 1, 40.0, -80.0, alt=0.0)],
        include_box=False, include_latlonalt=True,
    )
    comp = make_df(
        [_geo_row(0, 100, 40.0, -80.0, alt=15.0)],
        include_box=False, include_latlonalt=True,
    )

    ev_3d = HOTAReIDEvaluator(n_workers=0, config=_cfg('latlonalt'))
    ev_3d.evaluate({'v': ref}, {'v': comp})
    assert ev_3d.get_global_hota_data()['TP'][0] == 0

    ev_2d = HOTAReIDEvaluator(n_workers=0, config=_cfg('latlon'))
    ev_2d.evaluate({'v': ref}, {'v': comp})
    assert ev_2d.get_global_hota_data()['TP'][0] == 1


# ---------------------------------------------------------------------------
# §3.6 — NaN in lat/lon propagates to a numeric failure
# ---------------------------------------------------------------------------

def test_nan_latlon_raises_numeric_error():
    """
    NaN lat/lon → ECEF transform NaN → cost matrix NaN.
    Currently scipy's linear_sum_assignment catches it first
    (cost_matrix.py:77) and raises ValueError("matrix contains invalid
    numeric entries"), *not* NonFiniteSimilarityValueError. Lock that.
    """
    ref = make_df(
        [_geo_row(0, 1, np.nan, -80.0, alt=0.0)],
        include_box=False, include_latlonalt=True,
    )
    comp = make_df(
        [_geo_row(0, 100, 40.0, -80.0, alt=0.0)],
        include_box=False, include_latlonalt=True,
    )
    ev = HOTAReIDEvaluator(n_workers=0, config=_cfg('latlonalt'))
    with pytest.raises(ValueError, match="invalid numeric"):
        ev.evaluate({'v': ref}, {'v': comp})


# ---------------------------------------------------------------------------
# Direct cross-checks against the utility functions (lightweight)
# ---------------------------------------------------------------------------

def test_latlon_matches_latlonalt_when_alt_zero():
    """With altitude=0 in both inputs, latlonalt should agree with latlon."""
    p2 = np.array([[40.0, -80.0], [40.001, -80.001]])
    p3 = np.column_stack([p2, np.zeros(len(p2))])
    sim_2d = calculate_latlon_l2(p2, p2)
    sim_3d = calculate_latlonalt_l2(p3, p3)
    assert np.allclose(sim_2d, sim_3d, atol=1e-6)
