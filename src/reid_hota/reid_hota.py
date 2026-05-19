import os
import time
import warnings

import pandas as pd

from .config import HOTAConfig
from .constants import AnnotationColumn
from .hota_utils import build_HOTA_objects, compute_cost_per_video_per_frame, jaccard_cost_matrices, merge_hota_data


class HOTAReIDEvaluator:
    """
    Evaluator for HOTA (Higher Order Tracking Accuracy) metrics with ReID extensions.

    This class provides functionality to compute HOTA metrics for multi-object tracking
    and re-identification evaluation, supporting various similarity metrics and
    ID alignment strategies.
    """
    REQUIRED_COLUMNS = [
        AnnotationColumn.FRAME,
        AnnotationColumn.OBJECT_ID,
        AnnotationColumn.X1,
        AnnotationColumn.Y1,
        AnnotationColumn.X2,
        AnnotationColumn.Y2,
        AnnotationColumn.CLASS_ID,
        AnnotationColumn.LAT,
        AnnotationColumn.LON,
        AnnotationColumn.ALT,
        AnnotationColumn.BOX_HASH,
    ]

    def __init__(self, n_workers: int = 0, config: HOTAConfig | None = None):
        """
        Initialize the HOTAReIDEvaluator

        Args:
            n_workers: Number of workers to use for parallel processing.
            config: HOTAConfig object defining how the metric should be computed.
                    When None, a fresh default-valued HOTAConfig is constructed
                    per evaluator (avoids the mutable-default-argument trap).
        """
        self.n_workers = n_workers
        self.config = config if config is not None else HOTAConfig()
        self.config.validate()

        self.required_cols = self._determine_required_columns()
        self.global_hota_data = None
        self.per_video_hota_data = None
        self.per_frame_hota_data = None


    def _determine_required_columns(self) -> list[str]:
        """Determine which columns are required based on configuration."""
        required_cols = self.REQUIRED_COLUMNS.copy()

        if self.config.class_ids is None:
            required_cols.remove(AnnotationColumn.CLASS_ID)
        if not self.config.track_fp_fn_tp_box_hashes:
            required_cols.remove(AnnotationColumn.BOX_HASH)
        if self.config.similarity_metric == 'latlonalt' or self.config.similarity_metric == 'latlon':
            for col in [AnnotationColumn.X1, AnnotationColumn.Y1, AnnotationColumn.X2, AnnotationColumn.Y2]:
                required_cols.remove(col)
        if self.config.similarity_metric == 'iou':
            for col in [AnnotationColumn.LAT, AnnotationColumn.LON, AnnotationColumn.ALT]:
                required_cols.remove(col)
        if self.config.similarity_metric == 'latlon':
            required_cols.remove(AnnotationColumn.ALT)

        return required_cols

    def evaluate(self, ref_dfs: dict[str, pd.DataFrame],
                 comp_dfs: dict[str, pd.DataFrame]):
        """
        Compute the HOTA metrics for a set of reference and comparison dataframes.

        Args:
            ref_dfs: ``{video_id: ground-truth DataFrame}``
            comp_dfs: ``{video_id: tracker-output DataFrame}``

        Pipeline (see src/reid_hota/DESIGN.md):
            1. Per-frame similarity cost matrices
            2. Jaccard merge across frames → id alignment cost matrix
            3. Hungarian assignment → ref→comp id map
            4. Per-frame HOTAData construction + per-video/global aggregation
        """
        if not self.config.suppress_print_statements:
            print("=== Computing ReID HOTA metrics ===")

        self._validate_and_prepare_inputs(ref_dfs, comp_dfs)

        start_time = time.time()
        id_similarity_per_video = self._stage1_per_frame_cost_matrices(ref_dfs, comp_dfs)
        per_video_cost_matrices, global_cost_matrix = self._stage2_jaccard_and_assign(id_similarity_per_video)
        self._stage3_build_hota_objects(id_similarity_per_video, per_video_cost_matrices, global_cost_matrix)
        self._stage4_merge_global(id_similarity_per_video, start_time)

    # ------------------------------------------------------------------
    # Stage helpers — each one runs a single numbered block in evaluate()
    # ------------------------------------------------------------------

    def _validate_and_prepare_inputs(
        self,
        ref_dfs: dict[str, pd.DataFrame],
        comp_dfs: dict[str, pd.DataFrame],
    ) -> None:
        """Type-check inputs, backfill missing video keys, and project required columns."""
        # TypeError/ValueError so failures survive `python -O` (which strips asserts).
        if not isinstance(ref_dfs, dict):
            raise TypeError(f"ref_dfs must be a dictionary, got {type(ref_dfs)}")
        for video_id, df in ref_dfs.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"ref_dfs[{video_id}] must be a pandas DataFrame, got {type(df)}")
        if not isinstance(comp_dfs, dict):
            raise TypeError(f"comp_dfs must be a dictionary, got {type(comp_dfs)}")
        for video_id, df in comp_dfs.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"comp_dfs[{video_id}] must be a pandas DataFrame, got {type(df)}")

        # Backfill empty DataFrames for video ids missing on either side so that
        # downstream stages see a symmetric ref/comp key set.
        required_video_ids = set(ref_dfs.keys()) | set(comp_dfs.keys())
        for vid in required_video_ids:
            if vid not in comp_dfs:
                comp_dfs[vid] = pd.DataFrame(columns=self.required_cols)
            if vid not in ref_dfs:
                ref_dfs[vid] = pd.DataFrame(columns=self.required_cols)

        # Required-column check before projection so error messages name the missing col.
        for col in self.required_cols:
            for key, ref_df in ref_dfs.items():
                if col not in ref_df.columns:
                    raise ValueError(f"Column \"{col}\" not found in ref_df \"{key}\"")
            for key, comp_df in comp_dfs.items():
                if col not in comp_df.columns:
                    raise ValueError(f"Column \"{col}\" not found in comp_df \"{key}\"")

        for key in ref_dfs:
            ref_dfs[key] = ref_dfs[key][self.required_cols]
        for key in comp_dfs:
            comp_dfs[key] = comp_dfs[key][self.required_cols]

        if self.config.class_ids is not None:
            if not self.config.suppress_print_statements:
                print(f"Keeping only the relevant class_ids: {self.config.class_ids}")
            for key in ref_dfs:
                ref_dfs[key] = ref_dfs[key][ref_dfs[key][AnnotationColumn.CLASS_ID].isin(self.config.class_ids)]
            for key in comp_dfs:
                comp_dfs[key] = comp_dfs[key][comp_dfs[key][AnnotationColumn.CLASS_ID].isin(self.config.class_ids)]

    def _stage1_per_frame_cost_matrices(self, ref_dfs, comp_dfs):
        """Stage 1: per-video, per-frame similarity matrices."""
        st = time.time()
        if not self.config.suppress_print_statements:
            print("Computing cost matrix for every frame")
        result = compute_cost_per_video_per_frame(
            ref_dfs, comp_dfs, self.n_workers, self.config.similarity_metric,
        )
        if not self.config.suppress_print_statements:
            print(f"  took: {time.time() - st} seconds")
        return result

    def _stage2_jaccard_and_assign(self, id_similarity_per_video):
        """Stage 2/3: Jaccard merge across frames + Hungarian assignment.

        Returns ``(per_video_cost_matrices, global_cost_matrix)`` with the
        non-active branch set to None. For ``per_frame`` alignment both are
        None — assignment is rebuilt per frame inside HOTAData.
        """
        st = time.time()
        if not self.config.suppress_print_statements:
            print("Jaccard merge of per-frame cost")

        per_video_cost_matrices = None
        global_cost_matrix = None

        if self.config.id_alignment_method == 'per_video':
            per_video_cost_matrices = jaccard_cost_matrices(
                id_similarity_per_video, return_per_key=True, n_workers=self.n_workers,
            )
            for cm in per_video_cost_matrices.values():
                cm.construct_assignment()
                cm.construct_id2idx_lookup()

        elif self.config.id_alignment_method == 'global':
            global_cost_matrix = jaccard_cost_matrices(
                id_similarity_per_video, return_per_key=False, n_workers=self.n_workers,
            )['global']
            global_cost_matrix.construct_assignment()
            # Precompute id→idx lookup before fork so each worker inherits it.
            global_cost_matrix.construct_id2idx_lookup()

        if not self.config.suppress_print_statements:
            print(f"  took: {time.time() - st} seconds")
        return per_video_cost_matrices, global_cost_matrix

    def _stage3_build_hota_objects(self, id_similarity_per_video, per_video_cost_matrices, global_cost_matrix):
        """Stage 4 (first half): build per-frame and per-video HOTAData."""
        st = time.time()
        if not self.config.suppress_print_statements:
            print("Computing per-frame HOTA data")
        self.per_video_hota_data, self.per_frame_hota_data = build_HOTA_objects(
            id_similarity_per_video,
            config=self.config,
            per_video_cost_matrices=per_video_cost_matrices,
            global_cost_matrix=global_cost_matrix,
            n_workers=self.n_workers,
        )
        if not self.config.suppress_print_statements:
            print(f"  took: {time.time() - st} seconds")

    def _stage4_merge_global(self, id_similarity_per_video, start_time):
        """Stage 4 (second half): merge per-video → global HOTAData."""
        st = time.time()
        if not self.config.suppress_print_statements:
            print("Merging HOTA data")
        self.global_hota_data = merge_hota_data(
            list(self.per_video_hota_data.values()), config=self.config,
        )
        if not self.config.suppress_print_statements:
            print(f"  took: {time.time() - st} seconds")

        if not self.config.suppress_print_statements:
            nb_frames = sum(len(v) for v in id_similarity_per_video.values())
            elapsed = time.time() - start_time
            print(f"Total time taken: {elapsed} seconds")
            print(f"Number of frames: {nb_frames}")
            print(f"fps: {nb_frames / elapsed}")

    def get_results(self) -> tuple[dict, dict, dict]:
        """
        Get the results of the evaluation
        """
        return self.get_global_hota_data(), self.get_per_video_hota_data(), self.get_per_frame_hota_data()

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
            warnings.warn("Global HOTA data is not available", stacklevel=2)
            return
        if self.per_video_hota_data is None:
            warnings.warn("Per-video HOTA data is not available", stacklevel=2)
            return
        if self.per_frame_hota_data is None:
            warnings.warn("Per-frame HOTA data is not available", stacklevel=2)
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
            output_file = os.path.join(output_dir, 'hota_per_video.parquet')
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
            output_file = os.path.join(output_dir, 'hota_per_frame.parquet')
            df.to_parquet(output_file, index=False)
            # df.to_csv(output_file.replace('.parquet', '.csv'), index=False)

        df = pd.DataFrame(self.global_hota_data.get_dict())
        output_file = os.path.join(output_dir, 'hota.parquet')
        df.to_parquet(output_file, index=False)
        df.to_csv(output_file.replace('.parquet', '.csv'), index=False, float_format='%.4f')
