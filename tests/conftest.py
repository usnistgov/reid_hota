"""
Shared fixtures and helpers for the reid_hota test suite.

Two main exports:
- ``make_df``: synthetic DataFrame factory used by the new test modules
- ``base_tracking_data``: session-scoped loader for the meva_rid_short CSV data
"""
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import pytest
from test_utils import compute_box_hash

from reid_hota.constants import AnnotationColumn

# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _setup_multiprocessing():
    """Force the 'spawn' start method so worker behavior matches the library default."""
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # already set in this interpreter — fine
        pass


# ---------------------------------------------------------------------------
# Synthetic DataFrame factory
# ---------------------------------------------------------------------------

_DEFAULTS = {
    AnnotationColumn.FRAME: 0,
    AnnotationColumn.OBJECT_ID: 0,
    AnnotationColumn.X1: 0.0,
    AnnotationColumn.Y1: 0.0,
    AnnotationColumn.X2: 1.0,
    AnnotationColumn.Y2: 1.0,
    AnnotationColumn.CLASS_ID: 1,
    AnnotationColumn.LAT: 0.0,
    AnnotationColumn.LON: 0.0,
    AnnotationColumn.ALT: 0.0,
}

_BOX_COLS = (AnnotationColumn.X1, AnnotationColumn.Y1, AnnotationColumn.X2, AnnotationColumn.Y2)
_LATLONALT_COLS = (AnnotationColumn.LAT, AnnotationColumn.LON, AnnotationColumn.ALT)


def make_df(
    rows,
    *,
    include_box=True,
    include_latlonalt=False,
    include_box_hash=False,
):
    """
    Build a DataFrame from a list of row dicts.

    Only columns relevant to the current test's similarity metric are included.
    Missing per-row keys are filled from `_DEFAULTS`.

    Parameters
    ----------
    rows : list[dict]
        Each dict may contain any subset of: frame, id, x1, y1, x2, y2,
        object_type, lat, lon, alt.
    include_box : bool
        Include x1/y1/x2/y2 columns (default True).
    include_latlonalt : bool
        Include lat/lon/alt columns (default False).
    include_box_hash : bool
        Compute and include a box_hash column (default False).
    """
    cols = [AnnotationColumn.FRAME, AnnotationColumn.OBJECT_ID, AnnotationColumn.CLASS_ID]
    if include_box:
        cols.extend(_BOX_COLS)
    if include_latlonalt:
        cols.extend(_LATLONALT_COLS)

    data = {col: [] for col in cols}
    hashes = []
    for row in rows:
        for col in cols:
            data[col].append(row.get(col, _DEFAULTS[col]))
        if include_box_hash:
            hashes.append(compute_box_hash(row, defaults=_DEFAULTS))

    df = pd.DataFrame(data)

    # cast numeric columns to the dtypes the pipeline expects
    if AnnotationColumn.FRAME in df.columns:
        df[AnnotationColumn.FRAME] = df[AnnotationColumn.FRAME].astype(np.int64)
    if AnnotationColumn.OBJECT_ID in df.columns:
        # leave as-is so callers can pass strings or floats for type-robustness tests
        pass
    for col in _BOX_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)
    for col in _LATLONALT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)

    if include_box_hash:
        df[AnnotationColumn.BOX_HASH] = hashes
    return df


def empty_df(*, include_box=True, include_latlonalt=False, include_box_hash=False):
    """Empty DataFrame with the columns the active config requires."""
    return make_df([], include_box=include_box, include_latlonalt=include_latlonalt,
                   include_box_hash=include_box_hash)


# ---------------------------------------------------------------------------
# meva_rid_short fixture (shared with the legacy regression suite)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_tracking_data():
    """Load meva_rid_short ref/comp CSVs. Adds seeded random lat/lon/alt and box_hash."""
    np.random.seed(42)
    gt_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'ref')
    pred_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'comp')

    fns = [fn for fn in os.listdir(gt_fp) if fn.endswith('.csv')]
    ref_dfs, comp_dfs = {}, {}
    for fn in fns:
        gt_df = pd.read_csv(os.path.join(gt_fp, fn))
        pred_df = pd.read_csv(os.path.join(pred_fp, fn))

        for df in (gt_df, pred_df):
            df[AnnotationColumn.LAT] = np.random.rand(len(df)) * 10
            df[AnnotationColumn.LON] = np.random.rand(len(df)) * 10
            df[AnnotationColumn.ALT] = np.random.rand(len(df)) * 10

        for df in (gt_df, pred_df):
            df[AnnotationColumn.BOX_HASH] = df.apply(
                lambda row: compute_box_hash(row.to_dict()), axis=1
            )

        key = fn.replace('.csv', '')
        ref_dfs[key] = gt_df
        comp_dfs[key] = pred_df

    return ref_dfs, comp_dfs
