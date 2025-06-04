import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .cost_matrix import CostMatrixData, CostMatrixDataFrame
from .sparse_matrix import Sparse2DMatrix, Sparse1DMatrix
from .config import HOTAConfig


@dataclass
class FrameExtractionInputData:
    ref_df: pd.DataFrame
    comp_df: pd.DataFrame
    video_id: str  # the video id


@dataclass
class VideoFrameData:
    ref_np: np.ndarray  # reference video frame box numpy array with column names defined in col_names
    comp_np: np.ndarray  # comparison video frame box numpy array with column names defined in col_names
    video_id: str  # the video id
    frame: int  # the video frame number
    col_names: list[str]

    

@dataclass
class HOTAMetrics:
    """Container for HOTA metric data"""
    # Core counts
    tp: np.ndarray
    fn: np.ndarray  
    fp: np.ndarray
    unmatched_fp: float  # FP that are not matched to any ground truth id
    
    # Location accuracy
    loc_a_unnorm: np.ndarray
    loc_a: np.ndarray  # The average similarity score for matching detections
    
    # Association metrics
    ass_a: np.ndarray
    ass_re: np.ndarray
    ass_pr: np.ndarray
    
    # Detection metrics  
    det_a: np.ndarray
    det_re: np.ndarray
    det_pr: np.ndarray  # equivalent to precision used in mAP
    
    # Final metrics
    hota: np.ndarray
    owta: np.ndarray
    idf1: np.ndarray


class HOTAData:
    """
    Class for managing HOTA (Higher Order Tracking Accuracy) metric calculation and data.
    
    This class handles the computation of HOTA metrics including detection accuracy,
    association accuracy, and localization accuracy across multiple IoU thresholds.
    """

    def __init__(self, 
                 sim_cost_matrix: Optional[CostMatrixData] = None, 
                 gt_to_tracker_id_map: Optional[dict[np.dtype[np.object_], np.dtype[np.object_]]] = None, 
                 config: HOTAConfig = HOTAConfig()):
        """
        Initialize HOTAData instance.
        
        Args:
            sim_cost_matrix: The cost matrix for this frame.
            gt_to_tracker_id_map: A map from ground truth ids to tracker ids.
            iou_thresholds: Set of IoU thresholds to compute metrics for.
            purge_non_matched_comp_ids: Whether to purge non-matched comparison ids to reduce FP counts from video data that does not have full dense annotations of all objects
            gids: The ground truth ids to use for the HOTA metric. If provided, all other ids are ignored.
        """

        self.purge_non_matched_comp_ids = config.purge_non_matched_comp_ids
        self.iou_thresholds = np.asarray(config.iou_thresholds)
        self.gids = config.gids

        # sim_cost_matrix is the cost matrix just for this single frame.
        # if gt_to_tracker_id_map is None, then we use per-frame id alignment
        # Initialize data dictionary to store all metrics
        self.metrics = HOTAMetrics(
            tp=np.zeros(len(self.iou_thresholds), dtype=int),
            fn=np.zeros(len(self.iou_thresholds), dtype=int),
            fp=np.zeros(len(self.iou_thresholds), dtype=int),
            unmatched_fp=0,
            loc_a_unnorm=np.zeros(len(self.iou_thresholds), dtype=float),
            loc_a=np.zeros(len(self.iou_thresholds), dtype=float),
            ass_a=np.zeros(len(self.iou_thresholds), dtype=float),
            ass_re=np.zeros(len(self.iou_thresholds), dtype=float),
            ass_pr=np.zeros(len(self.iou_thresholds), dtype=float),
            det_a=np.zeros(len(self.iou_thresholds), dtype=float),
            det_re=np.zeros(len(self.iou_thresholds), dtype=float),
            det_pr=np.zeros(len(self.iou_thresholds), dtype=float),
            hota=np.zeros(len(self.iou_thresholds), dtype=float),
            owta=np.zeros(len(self.iou_thresholds), dtype=float),
            idf1=np.zeros(len(self.iou_thresholds), dtype=float)
        )
        self.sparse_data = {
            'matches_counts': [Sparse2DMatrix() for _ in range(len(self.iou_thresholds))],
            'ref_id_counts': Sparse1DMatrix(),
            'comp_id_counts': Sparse1DMatrix(),
            # 'TP_hashes': [set() for _ in range(len(self.iou_thresholds))],
            # 'FN_hashes': [set() for _ in range(len(self.iou_thresholds))],
            # 'FP_hashes': [set() for _ in range(len(self.iou_thresholds))]
        }

        # Frame metadata
        self.video_id: Optional[str] = None
        self.frame: Optional[int] = None
        if sim_cost_matrix is not None:
            self.video_id = sim_cost_matrix.video_id
            self.frame = sim_cost_matrix.frame
            self._populate(sim_cost_matrix, gt_to_tracker_id_map)
        else:
            raise ValueError("sim_cost_matrix is required")

    def get_dict(self) -> dict:
        """Get dictionary representation of HOTA data."""
        ret = {
            'IOU Thresholds': self.iou_thresholds,
            'video_id': self.video_id,
            'frame': self.frame,
            'TP': self.metrics.tp,
            'FN': self.metrics.fn,
            'FP': self.metrics.fp,
            'UnmatchedFP': self.metrics.unmatched_fp,
            'LocA': self.metrics.loc_a,
            'HOTA': self.metrics.hota,
            'AssA': self.metrics.ass_a,
            'AssRe': self.metrics.ass_re,
            'AssPr': self.metrics.ass_pr,
            'DetA': self.metrics.det_a,
            'DetRe': self.metrics.det_re,
            'DetPr': self.metrics.det_pr,
            'OWTA': self.metrics.owta,
            'IDF1': self.metrics.idf1}
        if 'FP_hashes' in self.sparse_data:
            ret['FP_hashes'] = list(self.sparse_data['FP_hashes'])
        if 'FN_hashes' in self.sparse_data:
            ret['FN_hashes'] = list(self.sparse_data['FN_hashes'])
        if 'TP_hashes' in self.sparse_data:
            ret['TP_hashes'] = list(self.sparse_data['TP_hashes'])
        return ret
        

    def __iadd__(self, other: 'HOTAData') -> 'HOTAData':
        """
        Add another HOTAData to this one in-place.
        
        Note: this only modifies TP, FN, FP, LocA, matches_counts, ref_id_counts, comp_id_counts.
        The other fields are not updated, and will need to be recomputed later with _finalize().
        
        Args:
            other: Another HOTAData to add to this one.
            
        Returns:
            Self, with values from other added.
        """
        # Add core metrics
        self.metrics.tp += other.metrics.tp
        self.metrics.fn += other.metrics.fn
        self.metrics.fp += other.metrics.fp
        self.metrics.unmatched_fp += other.metrics.unmatched_fp
        self.metrics.loc_a_unnorm += other.metrics.loc_a_unnorm
        
        # Add sparse data
        for a, alpha in enumerate(self.iou_thresholds):
            self.sparse_data['matches_counts'][a] += other.sparse_data['matches_counts'][a]
            
        self.sparse_data['ref_id_counts'] += other.sparse_data['ref_id_counts']
        self.sparse_data['comp_id_counts'] += other.sparse_data['comp_id_counts']
        
        return self

    def is_equal(self, other: 'HOTAData', tol: float = 1e-8) -> bool:
        """
        Check if this HOTAData object is equal to another one.
        
        Args:
            other: Another HOTAData object to compare with
            tol: Tolerance for floating point comparisons
            
        Returns:
            bool: True if the objects are equal, False otherwise
        """
        # Check basic attributes
        if self.video_id != other.video_id or self.frame != other.frame:
            return False
            
        # Check all numeric arrays in the metrics
        for key, value in self.metrics.__dict__.items():
            if isinstance(value, np.ndarray):
                other_value = getattr(other.metrics, key)
                if key in ['tp', 'fn', 'fp']:
                    # Integer arrays should be exactly equal
                    if not np.array_equal(value, other_value):
                        return False
                else:
                    # Float arrays should be close within tolerance
                    if not np.allclose(value, other_value, atol=tol):
                        return False
            
        return True
    
    def _add_TP_hashes(self, hashes: list[np.dtype[np.object_]], iou_threshold: float):
        key = 'TP_hashes'
        if key not in self.sparse_data:
            self.sparse_data[key] = [set() for _ in range(len(self.iou_thresholds))]
        self.sparse_data[key][iou_threshold].update(hashes)
    
    def _add_FN_hashes(self, hashes: list[np.dtype[np.object_]], iou_threshold: float):
        key = 'FN_hashes'
        if key not in self.sparse_data:
            self.sparse_data[key] = [set() for _ in range(len(self.iou_thresholds))]
        self.sparse_data[key][iou_threshold].update(hashes)
    
    def _add_FP_hashes(self, hashes: list[np.dtype[np.object_]], iou_threshold: float):
        key = 'FP_hashes'
        if key not in self.sparse_data:
            self.sparse_data[key] = [set() for _ in range(len(self.iou_thresholds))]
        self.sparse_data[key][iou_threshold].update(hashes)

    def _create_per_frame_id_mapping(self, sim_cost_matrix: CostMatrixData) -> dict:
        """Create ID mapping using Hungarian algorithm on the cost matrix."""
        frame_cost_matrix = sim_cost_matrix.copy()
        frame_cost_matrix.construct_assignment()
        frame_cost_matrix.construct_id2idx_lookup()
        return frame_cost_matrix.ref2comp_id_map
    
    def _extract_frame_matches(self, lcl_ref_ids: np.ndarray, lcl_comp_ids: np.ndarray, 
                          gt_to_tracker_id_map: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract matched ID pairs that exist in both the global mapping and current frame."""
        if len(lcl_ref_ids) == 0 or len(lcl_comp_ids) == 0:
            return np.array([], dtype=np.object_), np.array([], dtype=np.object_)
        
        comp_ids_set = set(lcl_comp_ids)
        frame_matches_id = []
        
        # Extract matches relevant to this frame
        for gt_id in lcl_ref_ids:
            if gt_id in gt_to_tracker_id_map:
                matched_tracker_id = gt_to_tracker_id_map[gt_id]
                if matched_tracker_id in comp_ids_set:
                    frame_matches_id.append((gt_id, matched_tracker_id))
        
        if frame_matches_id:
            match_ref_ids, match_comp_ids = zip(*frame_matches_id)
            return np.array(match_ref_ids, dtype=np.object_), np.array(match_comp_ids, dtype=np.object_)
        else:
            return np.array([], dtype=np.object_), np.array([], dtype=np.object_)
        
    def _filter_by_ground_truth_ids(self, lcl_ref_ids: np.ndarray, lcl_comp_ids: np.ndarray,
                                match_ref_ids: np.ndarray, match_comp_ids: np.ndarray, 
                                gids: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter all IDs to only include specified ground truth IDs and their matches."""
        # Filter matches to only include specified ground truth IDs
        mask = np.isin(match_ref_ids, gids)
        removed_ref_ids = match_ref_ids[np.invert(mask)]
        removed_comp_ids = match_comp_ids[np.invert(mask)]
        
        filtered_match_ref_ids = match_ref_ids[mask]
        filtered_match_comp_ids = match_comp_ids[mask]
        
        # Remove filtered IDs from local ID lists
        filtered_lcl_ref_ids = np.setdiff1d(lcl_ref_ids, removed_ref_ids)
        filtered_lcl_comp_ids = np.setdiff1d(lcl_comp_ids, removed_comp_ids)
        
        return filtered_lcl_ref_ids, filtered_lcl_comp_ids, filtered_match_ref_ids, filtered_match_comp_ids
    
    def _apply_comp_id_filtering(self, lcl_comp_ids: np.ndarray, match_comp_ids: np.ndarray,
                            gt_to_tracker_id_map: dict) -> tuple[np.ndarray, np.ndarray]:
        """Optionally filter comparison IDs to only include those with ground truth matches."""
        
        
        valid_comp_ids = list(gt_to_tracker_id_map.values())
        filtered_lcl_comp_ids = lcl_comp_ids[np.isin(lcl_comp_ids, valid_comp_ids)]
        filtered_match_comp_ids = match_comp_ids[np.isin(match_comp_ids, valid_comp_ids)]
        unmatched_fp = len(lcl_comp_ids) - len(filtered_lcl_comp_ids)
        return filtered_lcl_comp_ids, filtered_match_comp_ids, unmatched_fp
        
    
    def _get_matched_similarities(self, sim_cost_matrix: CostMatrixData, 
                             match_ref_ids: np.ndarray, match_comp_ids: np.ndarray) -> np.ndarray:
        """Extract similarity values for matched ID pairs and validate them."""
        if len(match_ref_ids) == 0:
            return np.array([])
        
        matched_similarity_vals = np.array([
            sim_cost_matrix.get_cost(i, j) for i, j in zip(match_ref_ids, match_comp_ids)
        ])
        
        # Validate similarity values are finite
        if np.any(~np.isfinite(matched_similarity_vals)):
            print(f"Non-finite value in matched_similarity_vals for video {sim_cost_matrix.video_id} frame {sim_cost_matrix.frame}")
            print(f"sim_cost_matrix.i_ids: {sim_cost_matrix.i_ids}")
            print(f"sim_cost_matrix.j_ids: {sim_cost_matrix.j_ids}")
            print(f"sim_cost_matrix.cost_matrix: {sim_cost_matrix.cost_matrix}")
            raise ValueError("Non-finite value in matched_similarity_vals")
        
        return matched_similarity_vals

    def _populate(self, sim_cost_matrix: CostMatrixData, 
                  gt_to_tracker_id_map: dict[np.dtype[np.object_], np.dtype[np.object_]]):
        
        # Step 1: Extract reference (ground truth) and comparison (tracker) IDs from this frame
        lcl_ref_ids = sim_cost_matrix.i_ids
        lcl_comp_ids = sim_cost_matrix.j_ids
        
        # Step 2: Establish ID mapping between ground truth and tracker
        if gt_to_tracker_id_map is None:
            # If no global mapping provided, create frame-level mapping using Hungarian algorithm
            gt_to_tracker_id_map = self._create_per_frame_id_mapping(sim_cost_matrix)

        # Step 3: Find matches between reference and comparison IDs for this frame
        match_ref_ids, match_comp_ids = self._extract_frame_matches(lcl_ref_ids, lcl_comp_ids, gt_to_tracker_id_map)

        # Step 4: Apply ground truth ID filtering if specified
        if self.gids is not None and len(self.gids) > 0:
            lcl_ref_ids, lcl_comp_ids, match_ref_ids, match_comp_ids = self._filter_by_ground_truth_ids(
                lcl_ref_ids, lcl_comp_ids, match_ref_ids, match_comp_ids, self.gids
            )

        # Step 5: Optional filtering of non-matched comparison IDs
        # This removes tracker IDs that don't have corresponding ground truth matches, reducing FP counts from video data that does not have full dense annotations of all objects
        if self.purge_non_matched_comp_ids:
            lcl_comp_ids, match_comp_ids, unmatched_fp = self._apply_comp_id_filtering(lcl_comp_ids, match_comp_ids, gt_to_tracker_id_map)
            self.metrics.unmatched_fp += unmatched_fp

        # Step 6: Extract similarity values for the matched pairs
        matched_similarity_vals = self._get_matched_similarities(sim_cost_matrix, match_ref_ids, match_comp_ids)

        # Step 7: Update detection counts for association metric computation
        # Calculate the total number of dets for each gt_id and tracker_id.
        for ref_id in lcl_ref_ids:
            self.sparse_data['ref_id_counts'].add_at(ref_id, 1)
        for comp_id in lcl_comp_ids:
            self.sparse_data['comp_id_counts'].add_at(comp_id, 1)
        
        # Step 8: Compute metrics across all IoU thresholds
        
        # Pre-compute common values outside the loop
        num_lcl_ref = len(lcl_ref_ids)
        num_lcl_comp = len(lcl_comp_ids)
        iou_thresholds_array = self.iou_thresholds
        eps = np.finfo('float').eps
        
        # Vectorized threshold comparison for all IoU thresholds at once
        if len(matched_similarity_vals) > 0:
            # Create a 2D mask: (num_matches, num_thresholds)
            threshold_masks = matched_similarity_vals[:, np.newaxis] >= (iou_thresholds_array - eps)[np.newaxis, :]
            
            # Count matches for each threshold using vectorized sum
            num_matches_per_threshold = np.sum(threshold_masks, axis=0)
            
            # Vectorized updates for TP, FN, FP
            self.metrics.tp += num_matches_per_threshold
            self.metrics.fn += num_lcl_ref - num_matches_per_threshold
            self.metrics.fp += num_lcl_comp - num_matches_per_threshold
            
            # Pre-compute masks for hash operations if needed
            if isinstance(sim_cost_matrix, CostMatrixDataFrame) and sim_cost_matrix.i_hashes is not None and sim_cost_matrix.j_hashes is not None:
                # Create index mappings for faster lookups
                ref_id_to_idx = {id_val: idx for idx, id_val in enumerate(sim_cost_matrix.i_ids)}
                comp_id_to_idx = {id_val: idx for idx, id_val in enumerate(sim_cost_matrix.j_ids)}
            
            # Process each threshold
            for a in range(len(self.iou_thresholds)):
                # Get matches for this threshold using pre-computed mask
                threshold_mask = threshold_masks[:, a]
                if np.any(threshold_mask):

                    alpha_match_ref_ids = match_ref_ids[threshold_mask]
                    alpha_match_comp_ids = match_comp_ids[threshold_mask]
                    sub_match_sim_vals = matched_similarity_vals[threshold_mask]
                    # Vectorized localization accuracy update
                    self.metrics.loc_a_unnorm[a] += float(np.sum(sub_match_sim_vals))
                    
                    # Batch update matches_counts - this is the main bottleneck we can't fully vectorize
                    # due to the sparse matrix structure, but we can optimize the loop
                    for ref_id, comp_id in zip(alpha_match_ref_ids, alpha_match_comp_ids):
                        self.sparse_data['matches_counts'][a].add_at(ref_id, comp_id, 1)
                
                    # Handle hash operations if needed
                    if isinstance(sim_cost_matrix, CostMatrixDataFrame) and sim_cost_matrix.i_hashes is not None and sim_cost_matrix.j_hashes is not None:
                    # Vectorized index lookup using pre-computed mappings
                    
                        matched_ref_indices = np.array([ref_id_to_idx[id_val] for id_val in alpha_match_ref_ids], dtype=int)
                        matched_comp_indices = np.array([comp_id_to_idx[id_val] for id_val in alpha_match_comp_ids], dtype=int)
                        
                        # Create boolean masks more efficiently
                        matched_ref_mask = np.zeros(len(sim_cost_matrix.i_ids), dtype=bool)
                        matched_comp_mask = np.zeros(len(sim_cost_matrix.j_ids), dtype=bool)
                        matched_ref_mask[matched_ref_indices] = True
                        matched_comp_mask[matched_comp_indices] = True
                        
                        # Extract hashes using masks
                        matched_ref_hashes = sim_cost_matrix.i_hashes[matched_ref_mask]
                        matched_comp_hashes = sim_cost_matrix.j_hashes[matched_comp_mask]
                        non_matched_ref_hashes = sim_cost_matrix.i_hashes[~matched_ref_mask]
                        non_matched_comp_hashes = sim_cost_matrix.j_hashes[~matched_comp_mask]
                        
                        self._add_TP_hashes(matched_ref_hashes, a)
                        self._add_TP_hashes(matched_comp_hashes, a)
                        self._add_FN_hashes(non_matched_ref_hashes, a)
                        self._add_FP_hashes(non_matched_comp_hashes, a)
                    
        else:
            # No matches case - vectorized update
            self.metrics.fn += num_lcl_ref
            self.metrics.fp += num_lcl_comp
            
            # Handle hashes for no-matches case
            if isinstance(sim_cost_matrix, CostMatrixDataFrame) and sim_cost_matrix.i_hashes is not None and sim_cost_matrix.j_hashes is not None:
                for a in range(len(self.iou_thresholds)):
                    self._add_FN_hashes(sim_cost_matrix.i_hashes, a)
                    self._add_FP_hashes(sim_cost_matrix.j_hashes, a)
                    
        self._finalize()
        
    def _finalize(self):
        for a, _ in enumerate(self.iou_thresholds):
            for k, v in self.sparse_data['matches_counts'][a].data_store.items():
                rid_count = self.sparse_data['ref_id_counts'].get(k[0])
                cid_count = self.sparse_data['comp_id_counts'].get(k[1])

                self.metrics.ass_a[a] += v * (v / max(1, (rid_count + cid_count - v)))
                self.metrics.ass_re[a] += v *(v / max(1, rid_count))
                self.metrics.ass_pr[a] += v *(v / max(1, cid_count))
            
        denom = np.maximum(1, self.metrics.tp)
        self.metrics.ass_a /= denom
        self.metrics.ass_re /= denom
        self.metrics.ass_pr /= denom
        self.metrics.loc_a = self.metrics.loc_a_unnorm / denom

        self.metrics.det_re = self.metrics.tp / np.maximum(1, self.metrics.tp + self.metrics.fn)
        self.metrics.det_pr = self.metrics.tp / np.maximum(1, self.metrics.tp + self.metrics.fp)
        # DetPr = TP / (TP + FP) which is equivalent to precision used in mAP.
        self.metrics.det_a = self.metrics.tp / np.maximum(1, self.metrics.tp + self.metrics.fn + self.metrics.fp)

        self.metrics.hota = np.sqrt(self.metrics.det_a * self.metrics.ass_a)
        self.metrics.owta = np.sqrt(self.metrics.det_re * self.metrics.ass_a)
        # ID-Recall = DetRe
        # ID-Precision = DetPr
        self.metrics.idf1 = self.metrics.tp / np.maximum(1, self.metrics.tp + (0.5 * self.metrics.fn) + (0.5 * self.metrics.fp))
        