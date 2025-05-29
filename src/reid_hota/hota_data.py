import numpy as np
import pandas as pd
from dataclasses import dataclass

from .cost_matrix import CostMatrixData
from .sparse_matrix import Sparse2DMatrix, Sparse1DMatrix


        
        


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

   




class HOTA_DATA:
    # Static class variables that can be accessed directly from the class
    array_labels = np.arange(0.1, 0.99, 0.1)  # TODO make this a parameter

    # res: dict[str, np.ndarray]  # on demand storage for HOTA fields
    video_id: str
    frame: int
    data: dict[str, np.ndarray | Sparse2DMatrix | Sparse1DMatrix]

    def __init__(self, sim_cost_matrix: CostMatrixData, gt_to_tracker_id_map: dict[np.dtype[np.object_], np.dtype[np.object_]], gids: list[np.dtype[np.object_]] = None):
        # sim_cost_matrix is the cost matrix just for this single frame.
        # if gt_to_tracker_id_map is None, then we use per-frame id alignment
        # Initialize data dictionary to store all metrics
        self.data = {
            # sparse storage required to build the HOTA metric later
            'matches_counts': [Sparse2DMatrix() for _ in HOTA_DATA.array_labels],
            'ref_id_counts': Sparse1DMatrix(),
            'comp_id_counts': Sparse1DMatrix(),
            'TP': np.zeros((len(HOTA_DATA.array_labels)), dtype=int),
            'FN': np.zeros((len(HOTA_DATA.array_labels)), dtype=int),
            'FP': np.zeros((len(HOTA_DATA.array_labels)), dtype=int),
            'TP_hashes': len(HOTA_DATA.array_labels) * [set()],
            'FN_hashes': len(HOTA_DATA.array_labels) * [set()],
            'FP_hashes': len(HOTA_DATA.array_labels) * [set()],
            'LocA_unnorm': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            'LocA': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            
            # Computed metrics
            'HOTA': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            'AssA': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            'AssRe': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            'AssPr': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            'DetA': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            'DetRe': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            'DetPr': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            'OWTA': np.zeros((len(HOTA_DATA.array_labels)), dtype=float),
            'IDF1': np.zeros((len(HOTA_DATA.array_labels)), dtype=float)
        }

        self.video_id = None
        self.frame = None
        if sim_cost_matrix is not None:
            self.video_id = sim_cost_matrix.video_id
            self.frame = sim_cost_matrix.frame

            self._populate(sim_cost_matrix, gt_to_tracker_id_map, gids)

    def get_dict(self) -> dict:
        return {
            'IOU Threshold': HOTA_DATA.array_labels,
            'video_id': self.video_id,
            'frame': self.frame,
            'TP': self.data['TP'],
            'FN': self.data['FN'],
            'FP': self.data['FP'],
            'LocA': self.data['LocA'],
            'HOTA': self.data['HOTA'],
            'AssA': self.data['AssA'],
            'AssRe': self.data['AssRe'],
            'AssPr': self.data['AssPr'],
            'DetA': self.data['DetA'],
            'DetRe': self.data['DetRe'],
            'DetPr': self.data['DetPr'],
            'OWTA': self.data['OWTA'],
            'IDF1': self.data['IDF1'],
            'TP_hashes': self.data['TP_hashes'] if 'TP_hashes' in self.data else None,
            'FN_hashes': self.data['FN_hashes'] if 'FN_hashes' in self.data else None,
            'FP_hashes': self.data['FP_hashes'] if 'FP_hashes' in self.data else None,
        }

    def __iadd__(self, other: 'HOTA_DATA') -> 'HOTA_DATA':
        """Adds another HOTA_DATA to this one in-place.
        Note: this only modifies TP, FN, FP, LocA, matches_counts, ref_id_counts, comp_id_counts
        The other fields are not updated, and will need to be recomputed later with _finalize()
        
        Args:
            other: Another HOTA_DATA to add to this one.
            
        Returns:
            Self, with values from other added.
        """
        self.data['TP'] += other.data['TP']
        self.data['FN'] += other.data['FN']
        self.data['FP'] += other.data['FP']
        self.data['LocA_unnorm'] += other.data['LocA_unnorm']
        
        for a in range(len(HOTA_DATA.array_labels)):
            self.data['matches_counts'][a] += other.data['matches_counts'][a]
            
        self.data['ref_id_counts'] += other.data['ref_id_counts']
        self.data['comp_id_counts'] += other.data['comp_id_counts']
        
        return self

    def is_equal(self, other: 'HOTA_DATA', tol: float = 1e-8) -> bool:
        """Check if this HOTA_DATA object is equal to another one.
        
        Args:
            other: Another HOTA_DATA object to compare with
            
        Returns:
            bool: True if the objects are equal, False otherwise
        """
        # Check basic attributes
        if self.video_id != other.video_id or self.frame != other.frame:
            return False
            
        # Check all numeric arrays in the data dictionary
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                if key in ['TP', 'FN', 'FP']:
                    # Integer arrays should be exactly equal
                    if not np.array_equal(value, other.data[key]):
                        return False
                else:
                    # Float arrays should be close within tolerance
                    if not np.allclose(value, other.data[key], atol=tol):
                        return False
            
        return True

    def _populate(self, sim_cost_matrix: CostMatrixData, gt_to_tracker_id_map: dict[np.dtype[np.object_], np.dtype[np.object_]], gids: list[np.dtype[np.object_]] = None):
        lcl_ref_ids = sim_cost_matrix.i_ids
        lcl_comp_ids = sim_cost_matrix.j_ids

        if gt_to_tracker_id_map is None:
            frame_cost_matrix = sim_cost_matrix.copy()
            frame_cost_matrix.construct_assignment()
            frame_cost_matrix.construct_id2idx_lookup()
            gt_to_tracker_id_map = frame_cost_matrix.ref2comp_id_map

        if len(lcl_ref_ids) == 0 or len(lcl_comp_ids) == 0:
            match_ref_ids, match_comp_ids = [], []
        else:
            comp_ids_set = set(lcl_comp_ids)

            # Extract matches relevant to this frame
            frame_matches_id = []
            for gt_id in lcl_ref_ids:
                if gt_id in gt_to_tracker_id_map:
                    matched_tracker_id = gt_to_tracker_id_map[gt_id]
                    if matched_tracker_id in comp_ids_set:
                        frame_matches_id.append((gt_id, matched_tracker_id))

            if frame_matches_id:
                match_ref_ids, match_comp_ids = zip(*frame_matches_id)
            else:
                match_ref_ids, match_comp_ids = [], []

        # Convert to numpy arrays
        match_ref_ids = np.array(match_ref_ids, dtype=np.object_)
        match_comp_ids = np.array(match_comp_ids, dtype=np.object_)

        if gids is not None and len(gids) > 0:
            mask = np.isin(match_ref_ids, gids)
            removed_ref_ids = match_ref_ids[np.invert(mask)]
            removed_comp_ids = match_comp_ids[np.invert(mask)]
            match_ref_ids = match_ref_ids[mask]
            match_comp_ids = match_comp_ids[mask]

            # remove removed_ref_ids from lcl_ref_ids
            lcl_ref_ids = np.setdiff1d(lcl_ref_ids, removed_ref_ids)
            # remove removed_comp_ids from lcl_comp_ids
            lcl_comp_ids = np.setdiff1d(lcl_comp_ids, removed_comp_ids)

        purge_non_matched_comp_ids_flag = False # TODO find a way to make this a parameter
        if purge_non_matched_comp_ids_flag:
            valid_comp_ids = list(gt_to_tracker_id_map.values())
            lcl_comp_ids = lcl_comp_ids[np.isin(lcl_comp_ids, valid_comp_ids)]
            match_comp_ids = match_comp_ids[np.isin(match_comp_ids, valid_comp_ids)]

        matched_similarity_vals = np.array([sim_cost_matrix.get_cost(i, j) for i, j in zip(match_ref_ids, match_comp_ids)])

        # Calculate the total number of dets for each gt_id and tracker_id.
        for ref_id in lcl_ref_ids:
            self.data['ref_id_counts'].add_at(ref_id, 1)
        for comp_id in lcl_comp_ids:
            self.data['comp_id_counts'].add_at(comp_id, 1)
        
        if np.any(~np.isfinite(matched_similarity_vals)):
            print(f"Non-finite value in matched_similarity_vals for video {sim_cost_matrix.video_id} frame {sim_cost_matrix.frame}")
            print(f"sim_cost_matrix.i_ids: {sim_cost_matrix.i_ids}")
            print(f"sim_cost_matrix.j_ids: {sim_cost_matrix.j_ids}")
            print(f"sim_cost_matrix.cost_matrix: {sim_cost_matrix.cost_matrix}")
            raise ValueError("Non-finite value in matched_similarity_vals")
        
        # Calculate and accumulate basic statistics
        for a, alpha in enumerate(HOTA_DATA.array_labels):
            actually_matched_mask = matched_similarity_vals >= alpha - np.finfo('float').eps
            alpha_match_ref_ids = match_ref_ids[actually_matched_mask]
            alpha_match_comp_ids = match_comp_ids[actually_matched_mask]
            sub_match_sim_vals = matched_similarity_vals[actually_matched_mask]
            num_matches = len(alpha_match_ref_ids)

            self.data['TP'][a] += num_matches
            self.data['FN'][a] += len(lcl_ref_ids) - num_matches
            self.data['FP'][a] += len(lcl_comp_ids) - num_matches

            # TODO fix this to use the box_hash column
            
            matched_ref_indices_mask = np.zeros(len(sim_cost_matrix.i_ids), dtype=bool)
            matched_ref_indices_mask[np.isin(sim_cost_matrix.i_ids, alpha_match_ref_ids)] = True
            matched_comp_indices_mask = np.zeros(len(sim_cost_matrix.j_ids), dtype=bool)
            matched_comp_indices_mask[np.isin(sim_cost_matrix.j_ids, alpha_match_comp_ids)] = True

            if sim_cost_matrix.i_hashes is not None and sim_cost_matrix.j_hashes is not None:
                matched_ref_hashes = sim_cost_matrix.i_hashes[matched_ref_indices_mask]
                matched_comp_hashes = sim_cost_matrix.j_hashes[matched_comp_indices_mask]
                non_matched_ref_hashes = sim_cost_matrix.i_hashes[np.invert(matched_ref_indices_mask)]
                non_matched_comp_hashes = sim_cost_matrix.j_hashes[np.invert(matched_comp_indices_mask)]
                self.data['TP_hashes'][a].update(matched_ref_hashes)
                self.data['TP_hashes'][a].update(matched_comp_hashes)
                self.data['FN_hashes'][a].update(non_matched_ref_hashes)
                self.data['FP_hashes'][a].update(non_matched_comp_hashes)
            

            if num_matches > 0:
                self.data['LocA_unnorm'][a] += float(np.sum(sub_match_sim_vals))

                for k in range(len(alpha_match_ref_ids)):
                    ref_id = alpha_match_ref_ids[k]
                    comp_id = alpha_match_comp_ids[k]
                    self.data['matches_counts'][a].add_at(ref_id, comp_id, 1)    
                    
        self._finalize()
        
    def _finalize(self):
        for a, _ in enumerate(HOTA_DATA.array_labels):
            for k, v in self.data['matches_counts'][a].data_store.items():
                rid_count = self.data['ref_id_counts'].get(k[0])
                cid_count = self.data['comp_id_counts'].get(k[1])

                self.data['AssA'][a] += v * (v / max(1, (rid_count + cid_count - v)))
                self.data['AssRe'][a] += v *(v / max(1, rid_count))
                self.data['AssPr'][a] += v *(v / max(1, cid_count))
            
        denom = np.maximum(1, self.data['TP'])
        self.data['AssA'] /= denom
        self.data['AssRe'] /= denom
        self.data['AssPr'] /= denom
        self.data['LocA'] = self.data['LocA_unnorm'] / denom

        self.data['DetRe'] = self.data['TP'] / np.maximum(1, self.data['TP'] + self.data['FN'])
        self.data['DetPr'] = self.data['TP'] / np.maximum(1, self.data['TP'] + self.data['FP'])
        # DetPr = TP / (TP + FP) which is equivalent to precision used in mAP.
        self.data['DetA'] = self.data['TP'] / np.maximum(1, self.data['TP'] + self.data['FN'] + self.data['FP'])

        self.data['HOTA'] = np.sqrt(self.data['DetA'] * self.data['AssA'])
        self.data['OWTA'] = np.sqrt(self.data['DetRe'] * self.data['AssA'])
        # ID-Recall = DetRe
        # ID-Precision = DetPr
        self.data['IDF1'] = self.data['TP'] / np.maximum(1, self.data['TP'] + (0.5 * self.data['FN']) + (0.5 * self.data['FP']))
        