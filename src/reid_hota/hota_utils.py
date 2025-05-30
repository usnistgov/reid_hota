import numpy as np
import pandas as pd
import copy
from multiprocessing import Pool


# Suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None

from .cost_matrix import CostMatrixData, CostMatrixDataFrame
from .hota_data import VideoFrameData, FrameExtractionInputData, HOTAData
from .config import HOTAConfig


def merge_hota_data(hota_data_list: list[HOTAData]) -> HOTAData:
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


def plot_hota(hota_data: HOTAData, output_dir: str):
    import matplotlib.pyplot as plt
    import os
    
    # Get unique metrics
    dict_data = hota_data.get_dict()
    del dict_data['frame']
    del dict_data['video_id']
    metrics = list(dict_data.keys())
    x_vals = hota_data.iou_thresholds
    
    # Plot each metric in its own subplot
    for metric_name in metrics:
        # Filter rows for this metric
        metric_data = dict_data[metric_name]
        
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, metric_data, '-')
        plt.xlabel('Threshold Alpha')
        plt.ylabel(f'{metric_name}')
        plt.title(f'{metric_name}')
        plt.grid(True, alpha=0.3)
        
        metric_plot_path = os.path.join(output_dir, f'{metric_name}.png')
        plt.savefig(metric_plot_path)
        plt.close()

def plot_per_frame_hota_data(per_frame_hota_data: pd.DataFrame, output_dir: str):
    import matplotlib.pyplot as plt
    import os
    
    # Verify we have a single video_id
    video_ids = per_frame_hota_data['video_id'].unique()
    assert len(video_ids) == 1
    video_id = video_ids[0]
    
    # Get unique metrics
    metrics = per_frame_hota_data['metric'].unique()
    
    # Get all alpha columns (those starting with 'alpha_')
    alpha_cols = [col for col in per_frame_hota_data.columns if col.startswith('alpha_')]
    
    # Plot each metric in its own subplot
    for metric_name in metrics:
        # Filter rows for this metric
        metric_data = per_frame_hota_data[per_frame_hota_data['metric'] == metric_name]
        
        # Calculate metrics for all frames at once using groupby
        grouped_metrics = metric_data.groupby('frame')[alpha_cols].mean().mean(axis=1).reset_index()
        
        # Sort by frame number
        grouped_metrics = grouped_metrics.sort_values('frame')
        
        plt.figure(figsize=(10, 6))
        plt.plot(grouped_metrics['frame'], grouped_metrics[0], '-')
        plt.xlabel('Video Frame')
        plt.ylabel(f'{metric_name}')
        plt.title(f'{metric_name} - Video {video_id}')
        plt.grid(True, alpha=0.3)
        
        metric_plot_path = os.path.join(output_dir, f'{video_id}_{metric_name}.png')
        plt.savefig(metric_plot_path)
        plt.close()
        

def compute_id_alignment_similarity_from_df(input_dat: FrameExtractionInputData, similarity_metric: str = 'iou') -> tuple[str, list[CostMatrixDataFrame]]:
    """
    Compute alignment costs between reference and comparison frames.
    """
    ref_df = input_dat.ref_df
    comp_df = input_dat.comp_df

    cols = ref_df.columns.tolist()
    
    # Group by frame
    # return pd.api.typing.DataFrameGroupBy
    ref_frames_df = ref_df.groupby('frame_id')
    comp_frames_df = comp_df.groupby('frame_id')

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

    f_idx = dat.col_names.index('frame_id')
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
        cost_matrix = calculate_latlogalt_l2(bb1, bb2)
    else:
        raise ValueError(f'Unsupported similarity metric: {similarity_metric}')

    return CostMatrixDataFrame(i_ids=ref_ids, j_ids=comp_ids, i_hashes=ref_hashes, j_hashes=comp_hashes, cost_matrix=cost_matrix, video_id=dat.video_id, frame=dat.frame)


def build_HOTA_objects(sim_cost_matrix_list: list[CostMatrixData], gt_to_tracker_id_map: dict[int, int], config: HOTAConfig) -> list[HOTAData]:
    # if gt_to_tracker_id_map is None, then we use per-frame id alignment
    dat_list = [HOTAData(sim_cost_matrix, gt_to_tracker_id_map, config) for sim_cost_matrix in sim_cost_matrix_list]
    return dat_list




def compute_cost_per_video_per_frame(ref_dfs: dict[str, pd.DataFrame], comp_dfs: dict[str, pd.DataFrame], n_workers:int=0, similarity_metric: str = 'iou') -> dict[str, list[CostMatrixDataFrame]]:

    # ************************************
    # Convert the list[pd.DataFrame] into a list[CostMatrixData]
    # ************************************
    frame_extraction_work_queue = _linearize_FrameExtractionInputData(ref_dfs, comp_dfs)
    id_similarity_per_video = dict()
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.starmap(compute_id_alignment_similarity_from_df, [(dat, similarity_metric) for dat in frame_extraction_work_queue])
    else:
        results = [compute_id_alignment_similarity_from_df(dat, similarity_metric) for dat in frame_extraction_work_queue]

    id_similarity_per_video = {}
    for vid, res in results:
        id_similarity_per_video[vid] = res

    return id_similarity_per_video
    

def jaccard_cost_matrices(matrices_list: list[CostMatrixData]) -> CostMatrixData:
    if not matrices_list:
        raise ValueError("list[CostMatrixData] is empty")
    
    # Get unique IDs across all matrices
    ref_ids = np.unique(np.concatenate([
        data.i_ids for data in matrices_list
    ]))
    comp_ids = np.unique(np.concatenate([
        data.j_ids for data in matrices_list
    ]))

    ref_lookup = {id_val: idx for idx, id_val in enumerate(ref_ids)}
    comp_lookup = {id_val: idx for idx, id_val in enumerate(comp_ids)}

    # Initialize output matrices with zeros
    shape = (len(ref_ids), len(comp_ids))
    i_counts = np.zeros(shape[0])
    j_counts = np.zeros(shape[1])
    cost_sum = np.zeros(shape, dtype=np.float64)

    # Process each CostMatrixData
    for data in matrices_list:
        ref_idx = np.fromiter((ref_lookup[id_] for id_ in data.i_ids), dtype=int)
        comp_idx = np.fromiter((comp_lookup[id_] for id_ in data.j_ids), dtype=int)

        i_counts[ref_idx] += 1
        j_counts[comp_idx] += 1

        # get a copy of the matrix, normalize the cost values and add to the sum
        cm = normalize_cost_matrix(data.cost_matrix.copy())
        cost_sum[ref_idx[:, np.newaxis], comp_idx[np.newaxis, :]] += cm

    cost_matrix = cost_sum / (i_counts[:, np.newaxis] + j_counts[np.newaxis, :] - cost_sum)

    return CostMatrixData(i_ids=ref_ids, j_ids=comp_ids, cost_matrix=cost_matrix, video_id=None, frame=None)


def process_jaccard_cost_matrix_chunk(matrices_chunk: list[CostMatrixData]) -> tuple:
    """Process a chunk of cost matrices and return intermediate calculations."""
    if not matrices_chunk:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.zeros((0, 0), dtype=np.float64)
        

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
        
        cm = normalize_cost_matrix(data.cost_matrix.copy())
        chunk_cost_sum[i_idx[:, np.newaxis], j_idx[np.newaxis, :]] += cm
    
    return chunk_i_ids, chunk_j_ids, chunk_i_counts, chunk_j_counts, chunk_cost_sum


def jaccard_cost_matrices_parallel(matrices_dict: dict[str, list[CostMatrixData]], n_workers: int = 1) -> CostMatrixData:
    if not matrices_dict:
        raise ValueError("dict[str, list[CostMatrixData]] is empty")
    
    
    # Single chunk case - use original implementation
    if n_workers <= 1:
        raise ValueError("n_workers must be greater than 1")
        
    else:
        # Split into chunks
        chunks = list(matrices_dict.values())

        # Process chunks in parallel
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_jaccard_cost_matrix_chunk, chunks)
        
    # Collect all unique IDs
    all_i_ids = np.unique(np.concatenate([res[0] for res in results]))
    all_j_ids = np.unique(np.concatenate([res[1] for res in results]))
    
    # Create global lookups
    ref_lookup = {id_val: idx for idx, id_val in enumerate(all_i_ids)}
    comp_lookup = {id_val: idx for idx, id_val in enumerate(all_j_ids)}
    
    # Initialize global matrices
    shape = (len(all_i_ids), len(all_j_ids))
    i_counts = np.zeros(shape[0])
    j_counts = np.zeros(shape[1])
    cost_sum = np.zeros(shape, dtype=np.float64)
    
    # Combine chunk results
    for chunk_i_ids, chunk_j_ids, chunk_i_counts, chunk_j_counts, chunk_cost_sum in results:
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
    
    return CostMatrixData(i_ids=all_i_ids, j_ids=all_j_ids, cost_matrix=cost_matrix, video_id=None, frame=None)


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
    ref_frames_df = ref_df.groupby('frame_id')
    comp_frames_df = comp_df.groupby('frame_id')

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
    # Convert to x0y0x1y1 format if needed - avoid unnecessary operations
    if box_format == 'xywh':
        # Create views instead of copies where possible
        boxes1 = np.empty_like(bboxes1)
        boxes2 = np.empty_like(bboxes2)

        # Compute coordinates directly
        np.copyto(boxes1[:, :2], bboxes1[:, :2])
        np.copyto(boxes2[:, :2], bboxes2[:, :2])
        boxes1[:, 2:] = bboxes1[:, :2] + bboxes1[:, 2:]
        boxes2[:, 2:] = bboxes2[:, :2] + bboxes2[:, 2:]
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


def calculate_latlogalt_l2(latlonalt1: np.ndarray, latlonalt2: np.ndarray):
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



def normalize_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    epsilon = 1e-8

    # Compute row and column sums once
    row_sums = np.sum(cost_matrix, axis=1, keepdims=True)
    col_sums = np.sum(cost_matrix, axis=0, keepdims=True)

    # Calculate denominator and normalize in one step where possible
    denom = row_sums + col_sums - cost_matrix
    # Pre-compute the maximum denominator with epsilon
    np.maximum(denom, epsilon, out=denom)
    # Perform division in-place
    np.divide(cost_matrix, denom, out=cost_matrix)

    return cost_matrix



def _linearize_FrameExtractionInputData(ref_dfs, comp_dfs) -> list[FrameExtractionInputData]:
    # ref_dfs and comp_dfs must have the same keys
    assert set(ref_dfs.keys()) == set(comp_dfs.keys())

    # Create the result list with a single list comprehension
    return [FrameExtractionInputData(ref_dfs[video_id], comp_dfs[video_id], video_id) for video_id in ref_dfs.keys()]



