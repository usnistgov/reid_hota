"""
Pipeline orchestration utilities for the HOTA evaluator.

This module owns the glue between stages 1, 3, and 4 of the pipeline
(per-frame similarity computation, in-frame duplicate-comp dedup, HOTA
object construction, and per-frame → per-video merging). The pure math
lives in sibling modules:

- ``similarity``  — IoU and ECEF L2 primitives
- ``jaccard``     — stage-2 Jaccard aggregation across frames

The symbols moved to those modules are re-exported below so existing
imports from ``reid_hota.hota_utils`` keep working.
"""
import copy
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .config import HOTAConfig
from .constants import BOX_FORMAT, AnnotationColumn
from .cost_matrix import CostMatrixData, CostMatrixDataFrame
from .hota_data import FrameExtractionInputData, HOTAData, VideoFrameData
from .hota_errors import DuplicateIDError, InvalidSimilarityMetricError
from .jaccard import (
    jaccard_cost_matrices,
    normalize_cost_matrix,
    process_jaccard_cost_matrix_chunk,
)
from .similarity import (
    ECEF_L2_DECAY_METERS,
    calculate_box_ious,
    calculate_latlon_l2,
    calculate_latlonalt_l2,
)

# Re-exports for backward compatibility — existing test code and downstream
# users import these names from this module.
__all__ = [
    "ECEF_L2_DECAY_METERS",
    "build_HOTA_objects",
    "build_HOTA_objects_worker",
    "calculate_box_ious",
    "calculate_latlon_l2",
    "calculate_latlonalt_l2",
    "compute_cost_per_video_per_frame",
    "compute_id_alignment_similarity",
    "compute_id_alignment_similarity_from_df",
    "jaccard_cost_matrices",
    "merge_hota_data",
    "normalize_cost_matrix",
    "process_jaccard_cost_matrix_chunk",
]


# ---------------------------------------------------------------------------
# Stage 4 helpers — HOTAData merging
# ---------------------------------------------------------------------------

def merge_hota_data(hota_data_list: list[HOTAData], config: HOTAConfig | None = None) -> HOTAData:
    """
    Merge a list of HOTAData objects into a single aggregated object.

    The merged object's video_id is set to the shared video_id when all inputs
    agree (per-frame → per-video merge), or None when they differ (per-video →
    global merge).

    Args:
        hota_data_list: List of HOTAData objects to merge
        config: Optional HOTAConfig used only to shape the empty-list placeholder
            (so its iou_thresholds match the caller's). Ignored when the list is
            non-empty — in that case the merged object inherits its shape from
            hota_data_list[0]. Default-None preserves prior behavior for callers
            that haven't been updated.

    Raises:
        MissingVideoIDError: If any item at index 1+ has video_id=None
    """
    if len(hota_data_list) == 0:
        # Empty placeholder must honor caller's iou_thresholds shape, otherwise
        # downstream aggregation (which assumes consistent shapes) breaks silently.
        return HOTAData(config=config)

    global_hota_data = copy.deepcopy(hota_data_list[0])
    global_hota_data.frame = None

    first_video_id = hota_data_list[0].video_id
    all_same_video_id = True

    for dat in hota_data_list[1:]:
        if dat.video_id is None:
            from .hota_errors import MissingVideoIDError
            raise MissingVideoIDError()
        if dat.video_id != first_video_id:
            all_same_video_id = False
        global_hota_data += dat

    global_hota_data.video_id = first_video_id if all_same_video_id else None
    global_hota_data._finalize()
    return global_hota_data


# ---------------------------------------------------------------------------
# Stage 1 — per-frame similarity computation + in-frame dedup
# ---------------------------------------------------------------------------

def compute_id_alignment_similarity_from_df(
    input_dat: FrameExtractionInputData,
    similarity_metric: str = 'iou',
) -> tuple[str, list[CostMatrixDataFrame]]:
    """Group ref/comp DataFrames by frame and emit one cost matrix per frame."""
    ref_df = input_dat.ref_df
    comp_df = input_dat.comp_df

    cols = ref_df.columns.tolist()

    ref_frames_df = ref_df.groupby(AnnotationColumn.FRAME)
    comp_frames_df = comp_df.groupby(AnnotationColumn.FRAME)

    shared_unique_frames = sorted(set(ref_frames_df.groups) | set(comp_frames_df.groups))

    cm_list = []
    for frame in shared_unique_frames:
        ref_frame_df = ref_frames_df.get_group(frame) if frame in ref_frames_df.groups else pd.DataFrame(columns=cols)
        comp_frame_df = comp_frames_df.get_group(frame) if frame in comp_frames_df.groups else pd.DataFrame(columns=cols)

        dat = VideoFrameData(ref_frame_df.values, comp_frame_df.values, input_dat.video_id, int(frame), cols)
        cm_list.append(compute_id_alignment_similarity(dat, similarity_metric))
    return input_dat.video_id, cm_list


def compute_id_alignment_similarity(
    dat: VideoFrameData,
    similarity_metric: str = 'iou',
) -> CostMatrixDataFrame:
    """
    Compute one frame's similarity matrix and dedupe duplicate comp ids via Jaccard.

    The dedupe step here mirrors the cross-frame Jaccard in ``jaccard.py``: when
    a comp id appears N times in the frame, the N columns collapse into one
    using ``sum / (1 + N - sum)`` so duplicate detections don't artificially
    inflate per-frame similarity.
    """
    f_idx = dat.col_names.index(AnnotationColumn.FRAME)
    id_idx = dat.col_names.index(AnnotationColumn.OBJECT_ID)
    hash_idx = dat.col_names.index(AnnotationColumn.BOX_HASH) if AnnotationColumn.BOX_HASH in dat.col_names else None

    ref_frames = np.unique(dat.ref_np[:, f_idx])
    comp_frames = np.unique(dat.comp_np[:, f_idx])
    # If either side has no detections this frame, return a degenerate zero matrix
    # so downstream code doesn't branch on emptiness.
    if len(comp_frames) == 0 or len(ref_frames) == 0:
        ref_ids = dat.ref_np[:, id_idx]
        comp_ids = dat.comp_np[:, id_idx]
        if hash_idx is not None:
            ref_hashes = dat.ref_np[:, hash_idx]
            comp_hashes = dat.comp_np[:, hash_idx]
        else:
            ref_hashes = None
            comp_hashes = None
        cost_matrix = np.zeros((len(ref_ids), len(comp_ids)))
        return CostMatrixDataFrame(
            i_ids=ref_ids, j_ids=comp_ids, i_hashes=ref_hashes, j_hashes=comp_hashes,
            cost_matrix=cost_matrix, video_id=dat.video_id, frame=dat.frame,
        )
    assert len(ref_frames) == 1 and len(comp_frames) == 1
    assert ref_frames[0] == comp_frames[0]

    # Reference must have unique ids per frame — caller bug if not.
    ref_ids_t = dat.ref_np[:, id_idx]
    unique_ref_ids, ref_counts = np.unique(ref_ids_t, return_counts=True)
    if np.max(ref_counts) > 1:
        duplicate_ids = unique_ref_ids[ref_counts > 1]
        raise DuplicateIDError(True, dat.video_id, dat.frame, duplicate_ids)

    # Comp may have duplicate ids legitimately (e.g. multi-camera trackers); deduped below.
    comp_ids_t = dat.comp_np[:, id_idx]
    unique_comp_ids, comp_counts = np.unique(comp_ids_t, return_counts=True)

    ref_ids = dat.ref_np[:, id_idx]
    comp_ids = dat.comp_np[:, id_idx]
    if hash_idx is not None:
        ref_hashes = dat.ref_np[:, hash_idx]
        comp_hashes = dat.comp_np[:, hash_idx]
    else:
        ref_hashes = None
        comp_hashes = None

    if similarity_metric == 'iou':
        box_idx = [
            dat.col_names.index(col)
            for col in [AnnotationColumn.X1, AnnotationColumn.Y1, AnnotationColumn.X2, AnnotationColumn.Y2]
        ]
        bb1 = dat.ref_np[:, box_idx].astype(float)
        bb2 = dat.comp_np[:, box_idx].astype(float)
        cost_matrix = calculate_box_ious(bb1, bb2, box_format=BOX_FORMAT)
    elif similarity_metric == 'latlonalt':
        box_idx = [
            dat.col_names.index(col)
            for col in [AnnotationColumn.LAT, AnnotationColumn.LON, AnnotationColumn.ALT]
        ]
        bb1 = dat.ref_np[:, box_idx].astype(float)
        bb2 = dat.comp_np[:, box_idx].astype(float)
        cost_matrix = calculate_latlonalt_l2(bb1, bb2)
    elif similarity_metric == 'latlon':
        box_idx = [dat.col_names.index(col) for col in [AnnotationColumn.LAT, AnnotationColumn.LON]]
        bb1 = dat.ref_np[:, box_idx].astype(float)
        bb2 = dat.comp_np[:, box_idx].astype(float)
        cost_matrix = calculate_latlon_l2(bb1, bb2)
    else:
        raise InvalidSimilarityMetricError(similarity_metric)

    # In-frame Jaccard dedup for duplicate comp ids.
    duplicate_comp_ids = unique_comp_ids[comp_counts > 1]
    cols_to_delete = []
    for c_id in duplicate_comp_ids:
        mask = np.where(comp_ids_t == c_id)[0]
        sub_cost_matrix = cost_matrix[:, mask]

        cost_sum = np.sum(sub_cost_matrix, axis=1)
        ref_count = 1
        comp_count = len(mask)

        jaccard_values = cost_sum / (ref_count + comp_count - cost_sum)
        jaccard_values = np.where(np.isfinite(jaccard_values), jaccard_values, 0.0)

        cost_matrix[:, mask[0]] = jaccard_values
        cols_to_delete.extend(mask[1:])

    if cols_to_delete:
        cost_matrix = np.delete(cost_matrix, cols_to_delete, axis=1)
        comp_ids = np.delete(comp_ids, cols_to_delete)
        if comp_hashes is not None:
            comp_hashes = np.delete(comp_hashes, cols_to_delete)

    return CostMatrixDataFrame(
        i_ids=ref_ids, j_ids=comp_ids, i_hashes=ref_hashes, j_hashes=comp_hashes,
        cost_matrix=cost_matrix, video_id=dat.video_id, frame=dat.frame,
    )


def compute_cost_per_video_per_frame(
    ref_dfs: dict[str, pd.DataFrame],
    comp_dfs: dict[str, pd.DataFrame],
    n_workers: int = 0,
    similarity_metric: str = 'iou',
) -> dict[str, list[CostMatrixDataFrame]]:
    """Stage 1: build ``{video_id: [per-frame CostMatrixDataFrame, ...]}``."""
    frame_extraction_work_queue = [
        FrameExtractionInputData(ref_dfs[video_id], comp_dfs[video_id], video_id)
        for video_id in ref_dfs.keys()
    ]
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.starmap(
                compute_id_alignment_similarity_from_df,
                [(dat, similarity_metric) for dat in frame_extraction_work_queue],
            )
    else:
        results = [compute_id_alignment_similarity_from_df(dat, similarity_metric) for dat in frame_extraction_work_queue]

    return {video_id: result for video_id, result in results}


# ---------------------------------------------------------------------------
# Stage 4 — HOTA object construction
# ---------------------------------------------------------------------------

def build_HOTA_objects_worker(
    sim_cost_matrix_list: list[CostMatrixData],
    gt_to_tracker_id_map: dict[int, int] | None,
    config: HOTAConfig,
) -> list[HOTAData]:
    """Build per-frame HOTAData for one video. None map ⇒ per-frame id alignment."""
    return [HOTAData(sim_cost_matrix, gt_to_tracker_id_map, config) for sim_cost_matrix in sim_cost_matrix_list]


def _id_map_for(
    video_id: str,
    config: HOTAConfig,
    per_video_cost_matrices: dict[str, CostMatrixData] | None,
    global_cost_matrix: CostMatrixData | None,
) -> dict[int, int] | None:
    """Pick the GT→tracker id map to hand to ``build_HOTA_objects_worker``."""
    if config.id_alignment_method == 'per_video':
        return per_video_cost_matrices[video_id].ref2comp_id_map
    if config.id_alignment_method == 'per_frame':
        return None
    return global_cost_matrix.ref2comp_id_map


def build_HOTA_objects(
    id_similarity_per_video: dict[str, list[CostMatrixDataFrame]],
    config: HOTAConfig,
    per_video_cost_matrices: dict[str, CostMatrixData] | None,
    global_cost_matrix: CostMatrixData | None,
    n_workers: int = 1,
):
    """Stage 4: emit per-video and per-frame HOTAData using the chosen id map."""
    video_ids = list(id_similarity_per_video.keys())
    video_chunks = list(id_similarity_per_video.values())

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            video_results = pool.starmap(
                build_HOTA_objects_worker,
                [
                    (chunk, _id_map_for(vid, config, per_video_cost_matrices, global_cost_matrix), config)
                    for vid, chunk in zip(video_ids, video_chunks, strict=False)
                ],
            )
    else:
        video_results = [
            build_HOTA_objects_worker(
                chunk,
                _id_map_for(vid, config, per_video_cost_matrices, global_cost_matrix),
                config,
            )
            for vid, chunk in zip(video_ids, video_chunks, strict=False)
        ]

    per_frame_hota_data = {vid: res for vid, res in zip(video_ids, video_results, strict=False)}

    def _empty_hota_data(vid: str, cfg: HOTAConfig) -> HOTAData:
        d = HOTAData(config=cfg)
        d.video_id = vid  # preserve key so empty videos still appear in the output dict
        return d

    per_video_hota_data = {
        vid: merge_hota_data(res) if res else _empty_hota_data(vid, config)
        for vid, res in zip(video_ids, video_results, strict=False)
    }
    return per_video_hota_data, per_frame_hota_data
