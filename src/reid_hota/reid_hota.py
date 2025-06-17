import os
import pandas as pd
import time

from .hota_utils import compute_cost_per_video_per_frame, jaccard_cost_matrices, build_HOTA_objects, merge_hota_data
from .config import HOTAConfig



class HOTAReIDEvaluator:
    """
    Evaluator for HOTA (Higher Order Tracking Accuracy) metrics with ReID extensions.
    
    This class provides functionality to compute HOTA metrics for multi-object tracking
    and re-identification evaluation, supporting various similarity metrics and 
    ID alignment strategies.
    """
    
    REQUIRED_COLUMNS = ['frame', 'object_id', 'x', 'y', 'w', 'h', 'class_id', 'lat', 'lon', 'alt', 'box_hash']

    def __init__(self, n_workers: int = 0, config: HOTAConfig = HOTAConfig()):
        """
        Initialize the HOTAReIDEvaluator

        Args:
            n_workers: Number of workers to use for parallel processing. 
            config: HOTAConfig object defining how the metric should be computed
        """
        self.n_workers = n_workers
        self.config = config
        self.config.validate()
       
        self.required_cols = self._determine_required_columns()
        self.global_hota_data = None
        self.per_video_hota_data = None
        self.per_frame_hota_data = None
        

    def _determine_required_columns(self) -> list[str]:
        """Determine which columns are required based on configuration."""
        required_cols = self.REQUIRED_COLUMNS.copy()
        
        if self.config.class_ids is None:
            required_cols.remove('class_id')
        if not self.config.track_fp_fn_tp_box_hashes:
            required_cols.remove('box_hash')
        if self.config.similarity_metric == 'latlonalt' or self.config.similarity_metric == 'latlon':
            for col in ['x', 'y', 'w', 'h']:
                required_cols.remove(col)
        if self.config.similarity_metric == 'iou':
            for col in ['lat', 'lon', 'alt']:
                required_cols.remove(col)
        if self.config.similarity_metric == 'latlon':
            required_cols.remove('alt')
        
        return required_cols

    def evaluate(self, ref_dfs: dict[str, pd.DataFrame], 
                 comp_dfs: dict[str, pd.DataFrame]):
        """
        Compute the HOTA metrics for a set of reference and comparison dataframes
        ref_dfs: dict[str, pd.DataFrame]
            A dictionary of reference dataframes, where the keys are the video ids and the values are the dataframes
        comp_dfs: dict[str, pd.DataFrame]
            A dictionary of comparison dataframes, where the keys are the video ids and the values are the dataframes
        """

        print(f"=== Computing ReID HOTA metrics ===")
        # Assert that ref_dfs is a dictionary of pandas dataframes
        assert isinstance(ref_dfs, dict), f"ref_dfs must be a dictionary, got {type(ref_dfs)}"
        for video_id, df in ref_dfs.items():
            assert isinstance(df, pd.DataFrame), f"ref_dfs[{video_id}] must be a pandas DataFrame, got {type(df)}"
        
        # Assert that comp_dfs is a dictionary of pandas dataframes
        assert isinstance(comp_dfs, dict), f"comp_dfs must be a dictionary, got {type(comp_dfs)}"
        for video_id, df in comp_dfs.items():
            assert isinstance(df, pd.DataFrame), f"comp_dfs[{video_id}] must be a pandas DataFrame, got {type(df)}"
        
        required_video_ids = set(ref_dfs.keys()) | set(comp_dfs.keys())
        # For any missing video IDs in dfs, create empty dataframes
        for id in required_video_ids:
            if id not in comp_dfs.keys():
                comp_dfs[id] = pd.DataFrame(columns=self.required_cols)
            if id not in ref_dfs.keys():
                ref_dfs[id] = pd.DataFrame(columns=self.required_cols)

        # verify columns in sequence_data
        for col in self.required_cols:
            for key in ref_dfs.keys():
                ref_df = ref_dfs[key]
                assert col in ref_df.columns, f"Column \"{col}\" not found in ref_df \"{key}\""
            for key in comp_dfs.keys():
                comp_df = ref_dfs[key]
                assert col in comp_df.columns, f"Column \"{col}\" not found in comp_df \"{key}\""

        # remove all but the required columns
        for key in ref_dfs.keys():
            ref_dfs[key] = ref_dfs[key][self.required_cols]
        for key in comp_dfs.keys():
            comp_dfs[key] = comp_dfs[key][self.required_cols]


        # Keep only the relevant classes
        if self.config.class_ids is not None:
            print(f"Keeping only the relevant class_ids: {self.class_ids}")
            for key in ref_dfs.keys():
                ref_dfs[key] = ref_dfs[key][ref_dfs[key]['class_id'].isin(self.class_ids)]
            for key in comp_dfs.keys():
                comp_dfs[key] = comp_dfs[key][comp_dfs[key]['class_id'].isin(self.class_ids)]

            

        start_time = time.time()
        # ************************************
        # build a per-video per-frame cost matrix
        # ************************************
        # returns a list[CostMatrixData] per video
        # each CostMatrixData stores the video_id and frame number for later reference
        st = time.time()
        print(f"Computing cost matrix for every frame")
        id_similarity_per_video = compute_cost_per_video_per_frame(ref_dfs, comp_dfs, self.n_workers, self.config.similarity_metric)
        print(f"  took: {time.time() - st} seconds")


        # ************************************
        # Convert the list[CostMatrixData] into a single CostMatrixData which represents the global cost matrix
        # jaccard is used to merge together the individual cost matrices, instead of average
        # ************************************
        st = time.time()
        print(f"Jaccard merge of per-frame cost")
        
        # None is a placeholder to tell later HOTA construction to use per-frame id alignment
        per_video_cost_matrices = None
        global_cost_matrix = None
        if self.config.id_alignment_method == 'per_video':
            # def jaccard_cost_matrices(matrices_dict: dict[str, list[CostMatrixData]], return_per_key:bool = False, n_workers: int = 1) -> CostMatrixData:
            per_video_cost_matrices = jaccard_cost_matrices(id_similarity_per_video, return_per_key=True, n_workers=self.n_workers)
            for video_id in per_video_cost_matrices.keys():
                per_video_cost_matrices[video_id].construct_assignment()
                per_video_cost_matrices[video_id].construct_id2idx_lookup()

        elif self.config.id_alignment_method == 'global':
            global_cost_matrix = jaccard_cost_matrices(id_similarity_per_video, return_per_key=False, n_workers=self.n_workers)
            global_cost_matrix = global_cost_matrix['global']
            # Construct the global assignment between ids
            global_cost_matrix.construct_assignment()
            # create mapping from global ids into the cost matrix index
            # preconstruct before copying to parallel workers, to save some time
            global_cost_matrix.construct_id2idx_lookup()

        elif self.config.id_alignment_method == 'per_frame':
            # None is a placeholder to tell HOTA construction to use per-frame id alignment
            per_video_cost_matrices = None
            global_cost_matrix = None

        # cost_matrix_data is the global id alignment cost matrix for all videos
        print(f"  took: {time.time() - st} seconds")

        

        
        # ************************************
        # Compute the per-frame HOTA data that will later be turned into HOTA data
        # ************************************
        st = time.time()
        print(f"Computing per-frame HOTA data")
        
        # Maintain video structure by processing each video separately
        self.per_video_hota_data, self.per_frame_hota_data = build_HOTA_objects(id_similarity_per_video, 
                                                                                 config=self.config, 
                                                                                 per_video_cost_matrices=per_video_cost_matrices, 
                                                                                 global_cost_matrix=global_cost_matrix, 
                                                                                 n_workers=self.n_workers)
        
        print(f"  took: {time.time() - st} seconds")


        # ************************************
        # Merge the per-frame hota data together into the global HOTA metric data class
        # ************************************
        st = time.time()
        print(f"Merging HOTA data")
        self.global_hota_data = merge_hota_data(list(self.per_video_hota_data.values()))
        self.global_hota_data.video_id = None  # remove the video_id from the global HOTA_DATA object
        print(f"  took: {time.time() - st} seconds")

        nb_frames = sum([len(v) for v in id_similarity_per_video.values()])
        print(f"Total time taken: {time.time() - start_time} seconds")
        print(f"Number of frames: {nb_frames}")
        print(f"fps: {nb_frames / (time.time() - start_time)}")

    def get_results(self) -> tuple[dict, dict, dict]:
        """
        Get the results of the evaluation
        """
        return self.global_hota_data.get_dict(), self.per_video_hota_data.get_dict(), self.per_frame_hota_data.get_dict()
    
    def get_global_hota_data(self) -> dict:
        """
        Get the global HOTA data
        """
        return self.global_hota_data.get_dict()
    
    def get_per_video_hota_data(self) -> dict:
        """
        Get the per-video HOTA data
        """
        res = dict()
        for video_id, video_data in self.per_video_hota_data.items():
            res[video_id] = video_data.get_dict()
        return res
    
    def get_per_frame_hota_data(self) -> dict:
        """
        Get the per-frame HOTA data
        """
        res = dict()
        for video_id, frame_data in self.per_frame_hota_data.items():
            res[video_id] = dict()
            for frame_dat in frame_data:
                res[video_id][frame_dat.frame] = frame_dat.get_dict()
        return res
    
    def export_to_file(self, output_dir: str, save_per_frame: bool = True, save_per_video: bool = True):
        if self.global_hota_data is None:
            print("Warning: Global HOTA data is not available")
            return
        if self.per_video_hota_data is None:
            print("Warning: Per-video HOTA data is not available")
            return
        if self.per_frame_hota_data is None:
            print("Warning: Per-frame HOTA data is not available")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        if save_per_video:
            df_source_list = []
            # Use the already structured per_video_hota_data instead of reconstructing it
            for video_id, video_data in self.per_video_hota_data.items():
                # Convert the list of HOTA_DATA for this video to a DataFrame
                df_source_list.append(video_data.get_dict())
            # Save the DataFrame to a parquet file in the output directory
            df = pd.DataFrame(df_source_list)
            output_file = os.path.join(output_dir, f'hota_per_video.parquet')
            df.to_parquet(output_file, index=False)
            # df.to_csv(output_file.replace('.parquet', '.csv'), index=False)

        if save_per_frame:
            df_source_list = []
            # Use the already structured per_video_hota_data instead of reconstructing it
            for video_id, frame_data in self.per_frame_hota_data.items():
                # Convert the list of HOTA_DATA for this video to a DataFrame
                for frame_dat in frame_data:
                    df_source_list.append(frame_dat.get_dict())
            # Save the DataFrame to a parquet file in the output directory
            df = pd.DataFrame(df_source_list)
            output_file = os.path.join(output_dir, f'hota_per_frame.parquet')
            df.to_parquet(output_file, index=False)
            # df.to_csv(output_file.replace('.parquet', '.csv'), index=False)

        df = pd.DataFrame(self.global_hota_data.get_dict())
        output_file = os.path.join(output_dir, f'hota.parquet')
        df.to_parquet(output_file, index=False)
        df.to_csv(output_file.replace('.parquet', '.csv'), index=False, float_format='%.4f')


            

