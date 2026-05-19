"""
Jaccard aggregation of per-frame cost matrices.

Stage 2 of the pipeline (see ``DESIGN.md``): given the dict of per-video
per-frame similarity matrices, combine them into either one matrix per
video (``return_per_key=True``) or one matrix across all videos
(``return_per_key=False``).

The merge uses the structural Jaccard
``S(i,j) / (count(i) + count(j) - S(i,j))``
rather than an average so id pairs that *consistently* match across many
frames outrank one-off high-similarity coincidences.

``normalize_cost_matrix`` lives here too because it is only called from the
chunk worker — it pre-conditions each per-frame matrix so columns/rows with
high local fan-out don't dominate the aggregation.
"""
from multiprocessing import Pool

import numpy as np

from .cost_matrix import CostMatrixData


def normalize_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    epsilon = 1e-8

    if np.size(cost_matrix) == 1:
        # Design decision: skip normalization for 1x1 matrices. Normalizing would
        # collapse any non-zero value to 1.0, destroying magnitude information needed
        # to compare independent 1x1 frames against each other. This path is hit
        # frequently (thousands of times per evaluation) and similarity metrics like
        # lat/lon are not bounded to [0, 1], so the raw value must be preserved.
        return cost_matrix

    row_sums = np.sum(cost_matrix, axis=1, keepdims=True)
    col_sums = np.sum(cost_matrix, axis=0, keepdims=True)

    denom = row_sums + col_sums - cost_matrix
    # Where denom is ~0 the entry stays 0 — see DESIGN.md for the degenerate
    # identical-box duplicate case that can still escape this guard.
    return np.divide(cost_matrix, denom, where=denom > epsilon, out=np.zeros_like(cost_matrix))


def process_jaccard_cost_matrix_chunk(video_id: str, matrices_chunk: list[CostMatrixData]) -> tuple:
    """Aggregate one video's per-frame matrices into (i_ids, j_ids, i_counts, j_counts, cost_sum).

    Each per-frame matrix contributes:
      - +1 to ``i_counts[i]`` for every reference id present that frame
      - +1 to ``j_counts[j]`` for every comparison id present that frame
      - the normalized similarity to ``cost_sum[i, j]``

    The Jaccard formula is applied in ``jaccard_cost_matrices`` after chunks
    are combined, so this function only returns the additive intermediates.
    """
    if not matrices_chunk:
        return video_id, np.array([]), np.array([]), np.array([]), np.array([]), np.zeros((0, 0), dtype=np.float64)

    chunk_i_ids = np.unique(np.concatenate([data.i_ids for data in matrices_chunk]))
    chunk_j_ids = np.unique(np.concatenate([data.j_ids for data in matrices_chunk]))

    chunk_i_lookup = {id_val: idx for idx, id_val in enumerate(chunk_i_ids)}
    chunk_j_lookup = {id_val: idx for idx, id_val in enumerate(chunk_j_ids)}

    shape = (len(chunk_i_ids), len(chunk_j_ids))
    chunk_i_counts = np.zeros(shape[0])
    chunk_j_counts = np.zeros(shape[1])
    chunk_cost_sum = np.zeros(shape, dtype=np.float64)

    for data in matrices_chunk:
        i_idx = np.fromiter((chunk_i_lookup[id_] for id_ in data.i_ids), dtype=int)
        j_idx = np.fromiter((chunk_j_lookup[id_] for id_ in data.j_ids), dtype=int)

        chunk_i_counts[i_idx] += 1
        chunk_j_counts[j_idx] += 1

        if len(i_idx) > 0 and len(j_idx) > 0:
            cm = normalize_cost_matrix(data.cost_matrix)
            chunk_cost_sum[i_idx[:, np.newaxis], j_idx[np.newaxis, :]] += cm

    return video_id, chunk_i_ids, chunk_j_ids, chunk_i_counts, chunk_j_counts, chunk_cost_sum


def jaccard_cost_matrices(
    matrices_dict: dict[str, list[CostMatrixData]],
    return_per_key: bool = False,
    n_workers: int = 1,
) -> dict[str, CostMatrixData]:
    """Combine per-frame cost matrices via structural Jaccard aggregation.

    Args:
        matrices_dict: ``{video_id: [per-frame CostMatrixData, ...]}``.
        return_per_key: When True, return one ``CostMatrixData`` per video
            (``per_video`` alignment). When False, return ``{'global': ...}``
            with one matrix spanning all videos (``global`` alignment).
        n_workers: Process pool size for chunk aggregation; serial when ≤1.
    """
    if not matrices_dict:
        raise ValueError("dict[str, list[CostMatrixData]] is empty")

    if n_workers <= 1:
        results = [process_jaccard_cost_matrix_chunk(video_id, chunk) for video_id, chunk in matrices_dict.items()]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.starmap(process_jaccard_cost_matrix_chunk, matrices_dict.items())

    if return_per_key:
        cost_matricies = dict()
        for video_id, i_ids, j_ids, i_counts, j_counts, cost_sum in results:
            cost_matrix = cost_sum / (i_counts[:, np.newaxis] + j_counts[np.newaxis, :] - cost_sum)
            cost_matricies[video_id] = CostMatrixData(
                i_ids=i_ids, j_ids=j_ids, cost_matrix=cost_matrix, video_id=None, frame=None,
            )
        return cost_matricies

    # Global aggregation: union all ids across chunks, then rebuild counts and cost_sum.
    all_i_ids = np.unique(np.concatenate([res[1] for res in results]))
    all_j_ids = np.unique(np.concatenate([res[2] for res in results]))

    ref_lookup = {id_val: idx for idx, id_val in enumerate(all_i_ids)}
    comp_lookup = {id_val: idx for idx, id_val in enumerate(all_j_ids)}

    shape = (len(all_i_ids), len(all_j_ids))
    i_counts = np.zeros(shape[0])
    j_counts = np.zeros(shape[1])
    cost_sum = np.zeros(shape, dtype=np.float64)

    for _, chunk_i_ids, chunk_j_ids, chunk_i_counts, chunk_j_counts, chunk_cost_sum in results:
        i_global_idx = np.fromiter((ref_lookup[id_] for id_ in chunk_i_ids), dtype=int)
        j_global_idx = np.fromiter((comp_lookup[id_] for id_ in chunk_j_ids), dtype=int)

        for local_idx, global_idx in enumerate(i_global_idx):
            i_counts[global_idx] += chunk_i_counts[local_idx]
        for local_idx, global_idx in enumerate(j_global_idx):
            j_counts[global_idx] += chunk_j_counts[local_idx]

        for i_local, i_global in enumerate(i_global_idx):
            for j_local, j_global in enumerate(j_global_idx):
                cost_sum[i_global, j_global] += chunk_cost_sum[i_local, j_local]

    cost_matrix = cost_sum / (i_counts[:, np.newaxis] + j_counts[np.newaxis, :] - cost_sum)
    return {'global': CostMatrixData(i_ids=all_i_ids, j_ids=all_j_ids, cost_matrix=cost_matrix, video_id=None, frame=None)}
