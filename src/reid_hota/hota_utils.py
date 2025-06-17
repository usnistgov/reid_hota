import numpy as np
import pandas as pd
import copy
from multiprocessing import Pool
from typing import List


# Suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None

from .cost_matrix import CostMatrixData, CostMatrixDataFrame
from .hota_data import VideoFrameData, FrameExtractionInputData, HOTAData
from .config import HOTAConfig


def merge_hota_data(hota_data_list: List[HOTAData]) -> HOTAData:
    """
    Merge a list of HOTA data objects into a single aggregated object.
    
    Args:
        hota_data_list: List of HOTAData objects to merge
        
    Returns:
        Single merged HOTAData object
        
    Raises:
        ValueError: If any video_id is None
    """
    if len(hota_data_list) == 0:
        return HOTAData() ## create empty placeholder
    # composite together the HOTA_DATAs into a single HOTA_DATA
    global_hota_data = copy.deepcopy(hota_data_list[0])
    global_hota_data.frame = None

    # iterate through the list of HOTA_DATAs and add them together, starting at 1
    # we already have the first HOTA_DATA in global_hota_data
    for dat in hota_data_list[1:]:
        if dat.video_id is None:
            raise ValueError("video_id is None")
        global_hota_data += dat
        
    global_hota_data._finalize()
    return global_hota_data


def compute_id_alignment_similarity_from_df(input_dat: FrameExtractionInputData, similarity_metric: str = 'iou') -> tuple[str, list[CostMatrixDataFrame]]:
    """
    Compute alignment costs between reference and comparison frames.
    """
    ref_df = input_dat.ref_df
    comp_df = input_dat.comp_df

    cols = ref_df.columns.tolist()
    
    # Group by frame
    # return pd.api.typing.DataFrameGroupBy
    ref_frames_df = ref_df.groupby('frame')
    comp_frames_df = comp_df.groupby('frame')

    k1 = set(ref_frames_df.groups.keys())
    k2 = set(comp_frames_df.groups.keys())
    shared_unique_frames = list(k1 | k2)  # union of keys
    shared_unique_frames.sort()

    cm_list = list()
    for frame in shared_unique_frames:
        if frame in ref_frames_df.groups:
            ref_frame_df = ref_frames_df.get_group(frame)
        else:
            ref_frame_df = pd.DataFrame(columns=cols)
        if frame in comp_frames_df.groups:
            comp_frame_df = comp_frames_df.get_group(frame)
        else:
            comp_frame_df = pd.DataFrame(columns=cols)

        # package into dataclass, adding class information
        dat = VideoFrameData(ref_frame_df.values, comp_frame_df.values, input_dat.video_id, int(frame), cols)
        cm = compute_id_alignment_similarity(dat, similarity_metric)
        cm_list.append(cm)
    return input_dat.video_id, cm_list


def compute_id_alignment_similarity(dat: VideoFrameData, similarity_metric: str = 'iou') -> CostMatrixDataFrame:
    """
    Compute alignment costs between reference and comparison frames.
    """

    f_idx = dat.col_names.index('frame')
    id_idx = dat.col_names.index('object_id')
    hash_idx = dat.col_names.index('box_hash') if 'box_hash' in dat.col_names else None
    # Quick validation using values access
    ref_frames = np.unique(dat.ref_np[:, f_idx])
    comp_frames = np.unique(dat.comp_np[:, f_idx])
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
        return CostMatrixDataFrame(i_ids=ref_ids, j_ids=comp_ids, i_hashes=ref_hashes, j_hashes=comp_hashes, cost_matrix=cost_matrix, video_id=dat.video_id, frame=dat.frame)
    assert len(ref_frames) == 1 and len(comp_frames) == 1
    assert ref_frames[0] == comp_frames[0]

    # This is reference data, it should never happen, but ... you never know
    # Check for duplicate IDs in reference data
    ref_ids_t = dat.ref_np[:, id_idx]
    unique_ref_ids, ref_counts = np.unique(ref_ids_t, return_counts=True)
    if np.max(ref_counts) > 1:
        duplicate_ids = unique_ref_ids[ref_counts > 1]
        raise ValueError(f'Ground-truth has duplicate IDs in frame {dat.frame}: {duplicate_ids}')

    # TODO how do we want to handle duplicate IDs? and reporting back to performers
    # Check for duplicate IDs in comparison data
    comp_ids_t = dat.comp_np[:, id_idx]
    unique_comp_ids, comp_counts = np.unique(comp_ids_t, return_counts=True)
    if np.max(comp_counts) > 1:
        duplicate_ids = unique_comp_ids[comp_counts > 1]
        raise ValueError(f'Tracker predictions have duplicate IDs in frame {dat.frame}: {duplicate_ids}')

    # Get unique IDs once
    ref_ids = dat.ref_np[:, id_idx]
    comp_ids = dat.comp_np[:, id_idx]
    if hash_idx is not None:
        ref_hashes = dat.ref_np[:, hash_idx]
        comp_hashes = dat.comp_np[:, hash_idx]
    else:
        ref_hashes = None
        comp_hashes = None

    if similarity_metric == 'iou':
        # Direct numpy array creation for bounding boxes
        box_idx = [dat.col_names.index(col) for col in ['x', 'y', 'w', 'h']]
        bb1 = dat.ref_np[:, box_idx].astype(float)
        bb2 = dat.comp_np[:, box_idx].astype(float)

        # Create cost matrix and compute IOUs
        cost_matrix = calculate_box_ious(bb1, bb2, box_format='xywh')
    elif similarity_metric == 'latlonalt':
        # Create cost matrix and compute lat/lon distance
        box_idx = [dat.col_names.index(col) for col in ['lat', 'lon', 'alt']]
        bb1 = dat.ref_np[:, box_idx].astype(float)
        bb2 = dat.comp_np[:, box_idx].astype(float)
        cost_matrix = calculate_latlonalt_l2(bb1, bb2)
    elif similarity_metric == 'latlon':
        # Create cost matrix and compute lat/lon distance
        box_idx = [dat.col_names.index(col) for col in ['lat', 'lon']]
        bb1 = dat.ref_np[:, box_idx].astype(float)
        bb2 = dat.comp_np[:, box_idx].astype(float)
        cost_matrix = calculate_latlon_l2(bb1, bb2)
    else:
        raise ValueError(f'Unsupported similarity metric: {similarity_metric}')

    return CostMatrixDataFrame(i_ids=ref_ids, j_ids=comp_ids, i_hashes=ref_hashes, j_hashes=comp_hashes, cost_matrix=cost_matrix, video_id=dat.video_id, frame=dat.frame)


def build_HOTA_objects_worker(sim_cost_matrix_list: list[CostMatrixData], gt_to_tracker_id_map: dict[int, int], config: HOTAConfig) -> list[HOTAData]:
    # if gt_to_tracker_id_map is None, then we use per-frame id alignment
    dat_list = [HOTAData(sim_cost_matrix, gt_to_tracker_id_map, config) for sim_cost_matrix in sim_cost_matrix_list]
    return dat_list


def build_HOTA_objects(id_similarity_per_video, config: HOTAConfig, per_video_cost_matrices: list[CostMatrixData], global_cost_matrix: CostMatrixData, n_workers: int = 1):
    # Create a list of (video_id, frames) tuples to process
    video_chunks = list(id_similarity_per_video.values())

    if n_workers > 1:
        # Process all videos in parallel - one chunk per video
        with Pool(processes=n_workers) as pool:

            # Process each video in parallel
            if config.id_alignment_method == 'per_video':
                # use the per-video id alignment cost matrix instead of the global one
                video_results = pool.starmap(build_HOTA_objects_worker, [(chunk, per_video_cost_matrices[chunk[0].video_id].ref2comp_id_map, config) for chunk in video_chunks])
            elif config.id_alignment_method == 'per_frame':
                video_results = pool.starmap(build_HOTA_objects_worker, [(chunk, None, config) for chunk in video_chunks])
            else:
                video_results = pool.starmap(build_HOTA_objects_worker, [(chunk, global_cost_matrix.ref2comp_id_map, config) for chunk in video_chunks])

    else:
        video_results = []
        for video_id, cm_values in id_similarity_per_video.items():
            # Process frames for this video sequentially
            if config.id_alignment_method == 'per_video':
                frame_dat = build_HOTA_objects_worker(cm_values, per_video_cost_matrices[video_id].ref2comp_id_map, config)
            elif config.id_alignment_method == 'per_frame':
                frame_dat = build_HOTA_objects_worker(cm_values, None, config)
            else:
                frame_dat = build_HOTA_objects_worker(cm_values, global_cost_matrix.ref2comp_id_map, config)
            video_results.append(frame_dat)

    # video_results is a list[HOTA_DATA]
    
    # Organize results into per-video structure
    per_frame_hota_data = {res[0].video_id: res for res in video_results}
    per_video_hota_data = {res[0].video_id: merge_hota_data(res) for res in video_results}
    return per_video_hota_data, per_frame_hota_data








def compute_cost_per_video_per_frame(ref_dfs: dict[str, pd.DataFrame], comp_dfs: dict[str, pd.DataFrame], n_workers:int=0, similarity_metric: str = 'iou') -> dict[str, list[CostMatrixDataFrame]]:

    # ************************************
    # Convert the list[pd.DataFrame] into a list[CostMatrixData]
    # ************************************
    frame_extraction_work_queue = [FrameExtractionInputData(ref_dfs[video_id], comp_dfs[video_id], video_id) for video_id in ref_dfs.keys()]
    id_similarity_per_video = dict()
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.starmap(compute_id_alignment_similarity_from_df, [(dat, similarity_metric) for dat in frame_extraction_work_queue])
    else:
        results = [compute_id_alignment_similarity_from_df(dat, similarity_metric) for dat in frame_extraction_work_queue]

    id_similarity_per_video = {}
    for video_id, result in results:
        id_similarity_per_video[video_id] = result

    return id_similarity_per_video


def process_jaccard_cost_matrix_chunk(video_id: str, matrices_chunk: list[CostMatrixData]) -> tuple:
    """Process a chunk of cost matrices and return intermediate calculations."""
    if not matrices_chunk:
        return video_id, np.array([]), np.array([]), np.array([]), np.array([]), np.zeros((0, 0), dtype=np.float64)
        

    # Get unique IDs in this chunk
    chunk_i_ids = np.unique(np.concatenate([data.i_ids for data in matrices_chunk]))
    chunk_j_ids = np.unique(np.concatenate([data.j_ids for data in matrices_chunk]))
    
    # Create lookups within this chunk
    chunk_i_lookup = {id_val: idx for idx, id_val in enumerate(chunk_i_ids)}
    chunk_j_lookup = {id_val: idx for idx, id_val in enumerate(chunk_j_ids)}
    
    # Initialize matrices for this chunk
    shape = (len(chunk_i_ids), len(chunk_j_ids))
    chunk_i_counts = np.zeros(shape[0])
    chunk_j_counts = np.zeros(shape[1])
    chunk_cost_sum = np.zeros(shape, dtype=np.float64)
    
    # Process each matrix in the chunk
    for data in matrices_chunk:
        i_idx = np.fromiter((chunk_i_lookup[id_] for id_ in data.i_ids), dtype=int)
        j_idx = np.fromiter((chunk_j_lookup[id_] for id_ in data.j_ids), dtype=int)

        chunk_i_counts[i_idx] += 1
        chunk_j_counts[j_idx] += 1

        if len(i_idx) > 0 and len(j_idx) > 0:
            cm = normalize_cost_matrix(data.cost_matrix)
            chunk_cost_sum[i_idx[:, np.newaxis], j_idx[np.newaxis, :]] += cm
    
    return video_id, chunk_i_ids, chunk_j_ids, chunk_i_counts, chunk_j_counts, chunk_cost_sum


def jaccard_cost_matrices(matrices_dict: dict[str, list[CostMatrixData]], return_per_key:bool = False, n_workers: int = 1) -> CostMatrixData:
    if not matrices_dict:
        raise ValueError("dict[str, list[CostMatrixData]] is empty")
    
    # Single chunk case - use original implementation
    if n_workers <= 1:
        # raise ValueError("n_workers must be greater than 1")
        results = [process_jaccard_cost_matrix_chunk(video_id, chunk) for video_id, chunk in matrices_dict.items()]
    else:
        # Process chunks in parallel
        with Pool(processes=n_workers) as pool:
            results = pool.starmap(process_jaccard_cost_matrix_chunk, matrices_dict.items())
        
    if return_per_key:
        cost_matricies = dict()
        for video_id, i_ids, j_ids, i_counts, j_counts, cost_sum in results:
            # Apply Jaccard formula
            cost_matrix = cost_sum / (i_counts[:, np.newaxis] + j_counts[np.newaxis, :] - cost_sum)
            cost_matrix = CostMatrixData(i_ids=i_ids, j_ids=j_ids, cost_matrix=cost_matrix, video_id=None, frame=None)
            cost_matricies[video_id] = cost_matrix
        return cost_matricies

    else:
        # Collect all unique IDs
        # video_id, chunk_i_ids, chunk_j_ids, chunk_i_counts, chunk_j_counts, chunk_cost_sum
        all_i_ids = np.unique(np.concatenate([res[1] for res in results]))
        all_j_ids = np.unique(np.concatenate([res[2] for res in results]))
        
        # Create global lookups
        ref_lookup = {id_val: idx for idx, id_val in enumerate(all_i_ids)}
        comp_lookup = {id_val: idx for idx, id_val in enumerate(all_j_ids)}
        
        # Initialize global matrices
        shape = (len(all_i_ids), len(all_j_ids))
        i_counts = np.zeros(shape[0])
        j_counts = np.zeros(shape[1])
        cost_sum = np.zeros(shape, dtype=np.float64)
        
        # Combine chunk results
        for _, chunk_i_ids, chunk_j_ids, chunk_i_counts, chunk_j_counts, chunk_cost_sum in results:

            # Map chunk indices to global indices
            i_global_idx = np.fromiter((ref_lookup[id_] for id_ in chunk_i_ids), dtype=int)
            j_global_idx = np.fromiter((comp_lookup[id_] for id_ in chunk_j_ids), dtype=int)
            
            # Update global counts and sum
            for local_idx, global_idx in enumerate(i_global_idx):
                i_counts[global_idx] += chunk_i_counts[local_idx]
                
            for local_idx, global_idx in enumerate(j_global_idx):
                j_counts[global_idx] += chunk_j_counts[local_idx]
            
            # Add cost sums from chunk to global matrix
            for i_local, i_global in enumerate(i_global_idx):
                for j_local, j_global in enumerate(j_global_idx):
                    cost_sum[i_global, j_global] += chunk_cost_sum[i_local, j_local]
        
        # Apply Jaccard formula
        cost_matrix = cost_sum / (i_counts[:, np.newaxis] + j_counts[np.newaxis, :] - cost_sum)
        
        return {'global': CostMatrixData(i_ids=all_i_ids, j_ids=all_j_ids, cost_matrix=cost_matrix, video_id=None, frame=None)}
        


def extract_per_frame_data(input_dat: FrameExtractionInputData, class_id: np.dtype[np.object_] = None) -> list[VideoFrameData]:
    ref_df = input_dat.ref_df
    comp_df = input_dat.comp_df

    cols = ref_df.columns.tolist()

    if class_id is not None:
        # only keep the relevant class
        ref_df = ref_df[ref_df['class_id'] == class_id]
        comp_df = comp_df[comp_df['class_id'] == class_id]
    
    # Group by frame
    # return pd.api.typing.DataFrameGroupBy
    ref_frames_df = ref_df.groupby('frame')
    comp_frames_df = comp_df.groupby('frame')

    k1 = set(ref_frames_df.groups.keys())
    k2 = set(comp_frames_df.groups.keys())
    shared_unique_frames = list(k1 | k2)  # union of keys
    shared_unique_frames.sort()

    dat_list = list()
    for frame in shared_unique_frames:
        if frame in ref_frames_df.groups:
            ref_frame_df = ref_frames_df.get_group(frame)
        else:
            ref_frame_df = pd.DataFrame(columns=cols)
        if frame in comp_frames_df.groups:
            comp_frame_df = comp_frames_df.get_group(frame)
        else:
            comp_frame_df = pd.DataFrame(columns=cols)

        # package into dataclass, adding class information
        dat = VideoFrameData(ref_frame_df.values, comp_frame_df.values, input_dat.video_id, int(frame), class_id, cols)
        dat_list.append(dat)
    return dat_list


def calculate_box_ious(bboxes1: np.ndarray, bboxes2: np.ndarray, box_format='xywh'):
    """
    Calculates the IOU (intersection over union) between two arrays of boxes using vectorized operations.

    Args:
        bboxes1: Array of shape (N, 4) containing first set of bounding boxes
        bboxes2: Array of shape (M, 4) containing second set of bounding boxes
        box_format: Format of input boxes - either 'xywh' (x, y, width, height) or
                   'x0y0x1y1' (x_min, y_min, x_max, y_max) (alias 'xyxy')
        
    Returns:
        Array of shape (N, M) containing pairwise IOU values
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))
    
    # Convert to x0y0x1y1 format if needed - avoid unnecessary operations
    if box_format == 'xywh':
        boxes1 = np.column_stack([
            bboxes1[:, 0], bboxes1[:, 1], 
            bboxes1[:, 0] + bboxes1[:, 2], bboxes1[:, 1] + bboxes1[:, 3]
        ])
        boxes2 = np.column_stack([
            bboxes2[:, 0], bboxes2[:, 1], 
            bboxes2[:, 0] + bboxes2[:, 2], bboxes2[:, 1] + bboxes2[:, 3]
        ])
    elif box_format in ('x0y0x1y1', 'xyxy'):
        # Use direct references instead of copying
        boxes1, boxes2 = bboxes1, bboxes2
    else:
        raise ValueError(f'Unsupported box format: {box_format}')

    # Pre-compute box areas once
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection coordinates efficiently
    # Use min/max operations on specific axes for better performance
    left = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    top = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    right = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    bottom = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    # Calculate intersection area - avoid creating temporary arrays
    width = np.maximum(0, right - left)
    height = np.maximum(0, bottom - top)
    intersection = width * height

    # Calculate union directly
    union = boxes1_area[:, None] + boxes2_area[None, :] - intersection

    # Constant for numerical stability - single definition
    epsilon = 1e-8

    # Compute IOUs
    ious = np.divide(intersection, np.maximum(union, epsilon))

    return ious


def calculate_latlonalt_l2(latlonalt1: np.ndarray, latlonalt2: np.ndarray):
    """
    Calculates the L2 Euclidean distance between points in 3D space (lat, lon, alt).

    Args:
        latlonalt1: Array of shape (N, 3) containing lat long alt
        latlonalt2: Array of shape (M, 3) containing lat long alt

    Returns:
        Array of shape (N, M) containing pairwise L2 distances
    """
    # Reshape to allow broadcasting: (N,3) -> (N,1,3) and (M,3) -> (1,M,3)
    points1 = latlonalt1[:, np.newaxis, :]
    points2 = latlonalt2[np.newaxis, :, :]
    
    # Calculate squared differences for all pairs
    squared_diff = np.sum((points1 - points2) ** 2, axis=2)
    # Take square root to get Euclidean distance
    distances = np.sqrt(squared_diff)
    similarities = np.exp(-distances / 10)
    
    return similarities


def calculate_latlon_l2(latlon1: np.ndarray, latlon2: np.ndarray):
    """
    Calculates the L2 Euclidean distance between points in 2D space (lat, lont).

    Args:
        latlon1: Array of shape (N, 2) containing lat long
        latlon2: Array of shape (M, 2) containing lat long

    Returns:
        Array of shape (N, M) containing pairwise L2 distances
    """
    # Reshape to allow broadcasting: (N,2) -> (N,1,2) and (M,2) -> (1,M,2)
    points1 = latlon1[:, np.newaxis, :]
    points2 = latlon2[np.newaxis, :, :]
    
    # Calculate squared differences for all pairs
    squared_diff = np.sum((points1 - points2) ** 2, axis=2)
    
    # Take square root to get Euclidean distance
    distances = np.sqrt(squared_diff)
    # use exp(-d/10) to get a [0,1] value that decays from 1 to 0 as distances increase
    # the d/10 calibrates the distance to similarity to better utilize the [0,1] range for normal inter-human distances
    similarities = np.exp(-distances / 10)
    
    return similarities



def normalize_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    epsilon = 1e-8

    if np.size(cost_matrix) == 1:
        # don't normalize single values to 1.0
        return cost_matrix

    # Compute row and column sums once
    row_sums = np.sum(cost_matrix, axis=1, keepdims=True)
    col_sums = np.sum(cost_matrix, axis=0, keepdims=True)

    # Calculate denominator and normalize in one step where possible
    denom = row_sums + col_sums - cost_matrix
    # Avoid division by zero by using np.divide with where
    cost_matrix = np.divide(cost_matrix, denom, where=denom > epsilon, out=np.zeros_like(cost_matrix))

    return cost_matrix



