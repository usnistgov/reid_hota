# CLAUDE.md

Your highest and best use is to challenge my thinking and force me to clarify my intent.
Plans go to `./plan.md`; all reasoning (tradeoffs, rejected alternatives, constraints) must precede implementation steps so DESIGN.md readers can reconstruct the *why* — can be one Background section or multiple purpose-named sections

## Project Overview

**ReID-HOTA** (`reid_hota` on PyPI) is a Python library for evaluating multi-object tracking and re-identification (ReID) performance. It computes both HOTA (Higher Order Tracking Accuracy) and IDF1 concurrently, with modifications to handle identity switches and re-appearances typical in ReID benchmarks.

### Architecture

Input is two `dict[str, pd.DataFrame]` (reference and comparison), keyed by video name. The pipeline:
1. **Cost matrices** — per-frame pairwise similarity between reference and comparison IDs (`hota_utils.py:compute_cost_per_video_per_frame`)
2. **Jaccard merging** — per-video or global ID alignment via Jaccard aggregation of per-frame costs (`hota_utils.py:jaccard_cost_matrices`)
3. **Hungarian assignment** — optimal GT→tracker ID mapping from the merged matrix (`cost_matrix.py:construct_assignment`)
4. **HOTAData objects** — metric computation per frame using the ID mapping (`hota_data.py`)
5. **Aggregation** — per-frame results merged into per-video and global metrics via `__iadd__`

Parallelism via `multiprocessing.Pool`; `n_workers` controls pool size. Multiprocessing start method must be `'spawn'` (set in tests; important for macOS/Linux consistency).

### Key Modules (`src/reid_hota/`)

| File | Purpose |
|---|---|
| `reid_hota.py` | `HOTAReIDEvaluator` — public API, orchestrates the pipeline |
| `config.py` | `HOTAConfig` dataclass — all tunable parameters |
| `hota_data.py` | `HOTAData` — metric storage and computation; `__iadd__` enables aggregation |
| `hota_utils.py` | Cost matrix computation, Jaccard merging, HOTA object construction |
| `cost_matrix.py` | `CostMatrixData` sparse cost matrix; wraps Hungarian solver |
| `sparse_matrix.py` | `Sparse2DMatrix` / `Sparse1DMatrix` — dict-backed sparse arrays used internally |
| `hota_errors.py` | Custom exception hierarchy (`HOTAConfigError`, `HOTARuntimeError`, subtypes) |
| `constants.py` | Column name string constants for DataFrames |

### Similarity Metrics

- `'iou'` (default) — Intersection over Union on xyxy bounding boxes
- `'latlon'` — L2 in ECEF meters (WGS84 reprojection via `pyproj`), alt=0 assumed
- `'latlonalt'` — L2 in ECEF meters including altitude

Lat/lon similarity uses `exp(-distance / ECEF_L2_DECAY_METERS)` (default decay = 10.0 m).

### ID Alignment Methods

`HOTAConfig(id_alignment_method=...)`:
- `'global'` — single GT→tracker assignment across all videos (default; best for true ReID)
- `'per_video'` — independent assignment per video
- `'per_frame'` — independent assignment per frame (equivalent to standard MOT HOTA)

### Non-Dense Annotation Handling

`reference_contains_dense_annotations=False` (default): comparison IDs with no GT match are removed from FP counts and counted separately as `UnmatchedFP`. Useful when GT only annotates a subset of objects in the scene.

### Required DataFrame Columns

Minimum: `['frame', 'id', 'x1', 'y1', 'x2', 'y2', 'object_type']`
Geographic: add `['lat', 'lon']` (latlon) or `['lat', 'lon', 'alt']` (latlonalt)
Hash tracking: add `['box_hash']` when `track_fp_fn_tp_box_hashes=True`

### Output

All metric fields (`HOTA`, `AssA`, `DetA`, `IDF1`, `LocA`, etc.) are numpy arrays of length `len(iou_thresholds)`. Results available via `get_global_hota_data()`, `get_per_video_hota_data()`, `get_per_frame_hota_data()`, or exported to parquet/CSV via `export_to_file()`.

### Testing

```bash
uv run pytest tests/
```

Tests in `tests/test_meva_reid_short.py` validate against ground-truth JSON baselines (tolerance 1e-4). Test data lives in `tests/data/`. Always set `multiprocessing.set_start_method('spawn')` before tests.

### Publishing

```bash
uv build
uv publish
```

Version is set in `src/reid_hota/__init__.py`; `pyproject.toml` reads it dynamically.

## Development Setup

See [README.md](README.md) for detailed installation and configuration instructions.


## Coding Conventions

- Use `uv` for dependency management, not pip directly; with a local `.venv`
- Include code comments as needed to improve understandability for future readers (human or AI). Module docstrings for every file explaining purpose and key abstractions. Section-separator comments for logical blocks within longer files. Inline comments for non-obvious decisions, hidden constraints, or connections to other modules. Do not over-comment obvious code — this is Python, not assembly.

## Documentation Conventions

- **Per-folder READMEs:** Directories with 3+ files or non-trivial complexity have a `README.md` explaining purpose, key files, and how they connect.
- **Code-adjacent DESIGN.md:** Retrospective design docs live next to the code (e.g., `pipeline/DESIGN.md`), not in `docs/`. They capture the *why* behind architectural decisions and rejected alternatives.
- **Breadcrumbs:** When information is hard to find, add a brief summary with a link to the full details in the most logical location. See `memory/feedback_breadcrumbs.md`.
