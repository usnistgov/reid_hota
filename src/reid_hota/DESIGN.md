# reid_hota — Pipeline Design Notes

This document captures the *why* behind the architecture in `hota_utils.py`,
`reid_hota.py`, and `hota_data.py`. Mechanics are in the code; this file is
for decisions, trade-offs, and contracts that aren't obvious from reading.

## The four-stage pipeline

`HOTAReIDEvaluator.evaluate()` (`reid_hota.py:73`) runs four stages, each
implemented in `hota_utils.py`:

1. **Per-frame similarity matrices** — `compute_cost_per_video_per_frame`
   builds, for every (video, frame), a dense `CostMatrixDataFrame` whose
   entries are pairwise similarity (IoU, ECEF-meter L2) between ref and
   comp detections in that frame. Each matrix carries its own
   `(video_id, frame)` for later traceability.

2. **Jaccard merge across frames** — `jaccard_cost_matrices` aggregates the
   per-frame matrices into either one matrix per video (`per_video`) or one
   matrix across the whole dataset (`global`). The aggregation formula
   `S(i,j) / (count(i) + count(j) - S(i,j))` is the structural Jaccard, not
   a simple average: it rewards id pairs that co-occur in many frames *and*
   are similar when they do, while a naive average would let two ids that
   coincided in a single frame with high IoU outrank a pair that matched
   reliably in hundreds of frames.

3. **Hungarian assignment** — `CostMatrixData.construct_assignment` calls
   `scipy.optimize.linear_sum_assignment` on the merged matrix to lock in
   a single ref→comp id mapping. For `per_frame`, this stage is skipped
   and matching is rebuilt fresh inside `HOTAData._populate` for each frame.

4. **Per-frame HOTA + aggregation** — `build_HOTA_objects` constructs one
   `HOTAData` per (video, frame), then `merge_hota_data` walks the per-frame
   list to fold each video into a `per_video_hota_data`, and the per-video
   results into `global_hota_data`. See contract below.

## ID alignment methods

The merge stage exists to make `id_alignment_method` work:

| method | what merges in stage 2 | when to use |
|---|---|---|
| `per_frame` | nothing — stages 2/3 are no-ops | standard MOT HOTA; each frame solves its own assignment |
| `per_video` | one matrix per video → one ref→comp map per video | tracker that's consistent within a video but renames ids across videos |
| `global` | one matrix across all videos → one ref→comp map | true ReID — same identity should hold the same id everywhere |

Default is `global` because the original motivation for this fork was ReID
evaluation, where per-frame and per-video both let id-switches go unpunished.

## The `__iadd__` / `_finalize()` contract

`HOTAData` is built incrementally. To keep aggregation O(N) instead of
re-running the metric math per frame, only the *additive* state survives a
merge:

- `__iadd__` (`hota_data.py:154`) sums **only** TP, FN, FP, UnmatchedFP,
  LocA_unnorm, matches_counts, ref_id_counts, comp_id_counts.
- AssA/AssRe/AssPr/DetA/DetRe/DetPr/HOTA/OWTA/IDF1/LocA are **stale** after
  any `+=`. They are valid only between `_populate()` (which calls
  `_finalize` at the end) and the next mutation.
- `merge_hota_data` (`hota_utils.py:25`) deepcopies the first element, runs
  `+=` for the rest, and calls `_finalize()` once at the end. Callers should
  not read the derived fields off a merge result until that finalize runs.

If you add a new metric, decide which bucket it lives in — additive (touch
both `__iadd__` and the data class init) or derived (compute it in
`_finalize` from the additive state). Mixing the two will silently produce
wrong global numbers.

## Duplicate-id handling (and a known degenerate case)

Reference data is required to have unique ids per frame —
`compute_id_alignment_similarity` raises `DuplicateIDError` on violation.

Comparison data may legitimately have duplicate ids in a frame (some
trackers emit detections per camera or per crop). The function dedupes
duplicate comp columns *inside* the frame using the same Jaccard formula
as stage 2 (count of ref appearances = 1, count of comp appearances =
mask size). This keeps stage-2 input clean.

**Known bug** — when duplicate comp ids have *identical* boxes that also
perfectly overlap a ref box, the in-frame Jaccard produces sim > 1, the
size-1 short-circuit in `normalize_cost_matrix` (`hota_utils.py:513`) lets
it through, and the global Jaccard later divides by zero. Locked as
`xfail(strict=True)` in `tests/test_empty_and_missing_data.py::test_duplicate_comp_id_identical_boxes_no_crash`.
Fix would require either rejecting identical-box duplicates upstream or
clamping post-Jaccard similarity to [0, 1] before the global merge —
both have semantic implications, so the bug is documented rather than
patched.

## The 1x1 short-circuit in `normalize_cost_matrix`

`normalize_cost_matrix` returns the input unchanged when it has a single
element (`hota_utils.py:513`). This is hit thousands of times per
evaluation (any frame with one ref and one comp), and matters because:

- For non-IoU metrics (lat/lon), values are not bounded to [0, 1], so the
  raw magnitude carries information.
- Normalizing a 1x1 matrix would collapse any non-zero value to 1.0,
  destroying the ability to compare independent 1x1 frames against each
  other in stage 2.

The trade-off is the degenerate-duplicate bug above. The short-circuit
is the cheaper side of that trade.

## Empty-frame handling

`compute_id_alignment_similarity` returns a degenerate zero-cost
`CostMatrixDataFrame` (no rows, no cols, or both) when either side has
no detections in a frame. This keeps downstream code from branching on
emptiness — the cost matrix is always present, just possibly shaped
`(0, N)`, `(N, 0)`, or `(0, 0)`. Stage-2 aggregation and stage-3
assignment handle zero-dimension matrices natively.

`build_HOTA_objects` similarly returns an `_empty_hota_data` placeholder
for any video with no frames, so per-video output keys always match the
union of `ref_dfs` and `comp_dfs` keys.

## Multiprocessing

Stages 1, 2, and 4 each use their own `multiprocessing.Pool` with
`n_workers > 1`. The library does not enforce a start method — callers
must set `'spawn'` before invoking the evaluator on Linux/macOS, or fork
semantics will leak state. The test suite does this in `conftest.py`.

`build_HOTA_objects_worker` takes the full `gt_to_tracker_id_map` per
chunk — small enough to pickle, large enough that the per-video
`construct_id2idx_lookup` precomputation in stage 3 saves real time by
avoiding redundant dict construction in every worker.
