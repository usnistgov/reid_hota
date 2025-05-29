import os
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool

from .fast_hota_utils import compute_cost_per_video_per_frame, jaccard_cost_matrices, build_HOTA_objects, merge_hota_data, jaccard_cost_matrices_parallel




ID_ALIGNMENT_METHODS = ['global','per_video','per_frame']
SIMILARITY_METRICS = ['iou', 'latlonalt']
REQUIRED_COLUMNS = ['frame_id', 'object_id', 'x', 'y', 'w', 'h', 'class_id', 'lat', 'lon', 'alt'] #, 'box_hash']



# TODO update to support https://git.codev.mitre.org/projects/VLINCS/repos/vlincs-database/browse/src/vlincs/database/data/Annotation.py
# _annotation_schema = pa.DataFrameSchema({
#     "frame": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
#     "time": pa.Column(str),
#     "id": pa.Column(str, checks=pa.Check.str_startswith("M"), nullable=True),
#     "x1": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
#     "y1": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
#     "x2": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
#     "y2": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
#     "width": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
#     "height": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
#     "lat": pa.Column(float, nullable=True),
#     "lon": pa.Column(float, nullable=True),
#     "elevation": pa.Column(float, nullable=True),
#     "depth": pa.Column(float, nullable=True),
#     "object_type": pa.Column(str, checks=pa.Check(lambda x: x.str.lower().isin(list(ObjectType.__members__.keys())))),
#     "source": pa.Column(str, checks=pa.Check(lambda x: x.str.lower().isin(list(SourceType.__members__.keys())))),
#     "confidence": pa.Column(float, checks=[pa.Check.greater_than_or_equal_to(0), pa.Check.less_than_or_equal_to(1.0)], nullable=True),
#     "occluded": pa.Column(bool),
#     "tracklet_id": pa.Column(int, nullable=True),
#     "box_hash": pa.Column(str),
#     },
#     strict="filter"
# )



# TODO make the set of alpha values a parameter to this function?
def compute_hota(
        ref_dfs: dict[str, pd.DataFrame], 
        comp_dfs: dict[str, pd.DataFrame], 
        output_dir: str = None, 
        n_workers:int=0, 
        class_ids: list[np.dtype[np.object_]] = None, 
        gids: list[np.dtype[np.object_]] = None, 
        id_alignment_method: str = 'global',
        similarity_metric: str = 'iou',
        restrict_box_hashes_to_per_frame: bool = True,
        ):
    """
    Compute the HOTA metrics for a set of reference and comparison dataframes
    ref_dfs: dict[str, pd.DataFrame]
        A dictionary of reference dataframes, where the keys are the video ids and the values are the dataframes
    comp_dfs: dict[str, pd.DataFrame]
        A dictionary of comparison dataframes, where the keys are the video ids and the values are the dataframes
    output_dir: str
        The directory to save the HOTA precursor data and per-alpha plots to
    n_workers: int
        The number of workers to use for the computation
    class_ids: list[np.dtype[np.object_]]
        The list of class ids to use for the computation. If None, all classes will be used.
    gids: list[np.dtype[np.object_]]
        The global ids to evaluate. If None/empty, all global ids will be used.
    id_alignment_method: str
        The method to use for id alignment. Must be one of: 'global', 'per_video', 'per_frame'
    similarity_metric: str
        The similarity metric to use. Must be one of: 'iou', 'latlonalt'
    restrict_box_hashes_to_per_frame: bool
        If True, the box hashes tracking TP, FP, FN will only be kept in the per-frame HOTA data, to avoid sigificant overhead.
    Returns:
        A dictionary of HOTA_DATA objects, where the keys are the video ids and the values are the HOTA_DATA objects. The 'COMBINED_SEQ' key contains the global HOTA_DATA object. If global_hota_merge is used, only the 'COMBINED_SEQ' key is present.
    """


    # TODO swap valiation to use pandera
    # https://git.codev.mitre.org/projects/VLINCS/repos/vlincs-database/browse/src/vlincs/database/data/Annotation.py
    # Assert that ref_dfs is a dictionary of pandas dataframes
    assert isinstance(ref_dfs, dict), f"ref_dfs must be a dictionary, got {type(ref_dfs)}"
    for video_id, df in ref_dfs.items():
        assert isinstance(df, pd.DataFrame), f"ref_dfs[{video_id}] must be a pandas DataFrame, got {type(df)}"
    
    # Assert that comp_dfs is a dictionary of pandas dataframes
    assert isinstance(comp_dfs, dict), f"comp_dfs must be a dictionary, got {type(comp_dfs)}"
    for video_id, df in comp_dfs.items():
        assert isinstance(df, pd.DataFrame), f"comp_dfs[{video_id}] must be a pandas DataFrame, got {type(df)}"

    assert id_alignment_method in ID_ALIGNMENT_METHODS, f"id_alignment_method must be one of: {ID_ALIGNMENT_METHODS}"
    assert similarity_metric in SIMILARITY_METRICS, f"similarity_metric must be one of: {SIMILARITY_METRICS}"

    requried_cols = REQUIRED_COLUMNS.copy()
    if class_ids is None:
        requried_cols.remove('class_id')
    if similarity_metric == 'latlonalt':
        requried_cols.remove('x')
        requried_cols.remove('y')
        requried_cols.remove('w')
        requried_cols.remove('h')
    if similarity_metric == 'iou':
        requried_cols.remove('lat')
        requried_cols.remove('lon')
        requried_cols.remove('alt')

    required_video_ids = ref_dfs.keys()
    # Keep only the required video IDs from comp_dfs
    comp_dfs = {id: comp_dfs[id] for id in required_video_ids if id in comp_dfs}
    # For any missing video IDs in comp_dfs, create empty dataframes
    for id in required_video_ids:
        if id not in comp_dfs.keys():
            comp_dfs[id] = pd.DataFrame(columns=requried_cols)

    # verify columns in sequence_data
    for col in requried_cols:
        for key in ref_dfs.keys():
            ref_df = ref_dfs[key]
            assert col in ref_df.columns, f"Column \"{col}\" not found in ref_df \"{key}\""
        for key in comp_dfs.keys():
            comp_df = ref_dfs[key]
            assert col in comp_df.columns, f"Column \"{col}\" not found in comp_df \"{key}\""

    # remove all but the required columns
    for key in ref_dfs.keys():
        ref_dfs[key] = ref_dfs[key][requried_cols]
    for key in comp_dfs.keys():
        comp_dfs[key] = comp_dfs[key][requried_cols]


    start_time = time.time()
    
    # Keep only the relevant classes
    st = time.time()
    
    if class_ids is not None:
        print(f"Keeping only the relevant class_ids: {class_ids}")
        for key in ref_dfs.keys():
            ref_dfs[key] = ref_dfs[key][ref_dfs[key]['class_id'].isin(class_ids)]
        for key in comp_dfs.keys():
            comp_dfs[key] = comp_dfs[key][comp_dfs[key]['class_id'].isin(class_ids)]
        print(f"  took: {time.time() - st} seconds")



    # ************************************
    # build a per-video per-frame cost matrix
    # ************************************
    # returns a list[CostMatrixData] per video
    # each CostMatrixData stores the video_id and frame number for later reference
    st = time.time()
    print(f"Computing cost matrix for every frame")
    id_similarity_per_video = compute_cost_per_video_per_frame(ref_dfs, comp_dfs, n_workers, similarity_metric)
    print(f"  took: {time.time() - st} seconds")


    # ************************************
    # Convert the list[CostMatrixData] into a single CostMatrixData which represents the global cost matrix
    # jaccard is used to merge together the individual cost matrices, instead of average
    # ************************************
    st = time.time()
    print(f"Jaccard merge of per-frame cost")
    
    if id_alignment_method == 'per_video':
        per_video_cost_matrices = dict()
        for video_id in id_similarity_per_video.keys():
            video_cost_matrix = jaccard_cost_matrices(id_similarity_per_video[video_id])
            # Construct the assignment between ids
            video_cost_matrix.construct_assignment()
            # create mapping from ids into the cost matrix index. This translates between the global id space and incides into the cost matrix
            # preconstruct before copying to parallel workers, to save some time
            video_cost_matrix.construct_id2idx_lookup()
            per_video_cost_matrices[video_id] = video_cost_matrix
    elif id_alignment_method == 'global':
        if n_workers > 1:
            global_cost_matrix = jaccard_cost_matrices_parallel(id_similarity_per_video, n_workers=n_workers)
        else:
            # flatten similarity_per_video into a single 1D list to pass to jaccard_cost_matrices
            flattened_similarity = []
            for v in id_similarity_per_video.values():
                flattened_similarity.extend(v)
            global_cost_matrix = jaccard_cost_matrices(flattened_similarity)    

        # Construct the global assignment between ids
        global_cost_matrix.construct_assignment()
        # create mapping from global ids into the cost matrix index
        # preconstruct before copying to parallel workers, to save some time
        global_cost_matrix.construct_id2idx_lookup()
    elif id_alignment_method == 'per_frame':
        per_video_cost_matrices = dict()
        global_cost_matrix = None
    else:
        raise ValueError(f"id_alignment_method must be one of: {ID_ALIGNMENT_METHODS}")

    # cost_matrix_data is the global id alignment cost matrix for all videos
    print(f"  took: {time.time() - st} seconds")

    

    
    # ************************************
    # Compute the per-frame HOTA data that will later be turned into HOTA data
    # ************************************
    st = time.time()
    # utilization here is meh
    print(f"Computing per-frame HOTA data")
    
    # Maintain video structure by processing each video separately
    per_video_hota_data = {}
    per_frame_hota_data = {}
    
    if n_workers > 1:
        # Process all videos in parallel - one chunk per video
        with Pool(processes=n_workers) as pool:
            # Create a list of (video_id, frames) tuples to process
            video_chunks = list(id_similarity_per_video.values())
            
            # Process each video in parallel
            if id_alignment_method == 'per_video':
                # use the per-video id alignment cost matrix instead of the global one
                video_results = pool.starmap(build_HOTA_objects, [(chunk, per_video_cost_matrices[chunk[0].video_id].ref2comp_id_map, gids) for chunk in video_chunks])
            elif id_alignment_method == 'per_frame':
                video_results = pool.starmap(build_HOTA_objects, [(chunk, None, gids) for chunk in video_chunks])
            else:
                video_results = pool.starmap(build_HOTA_objects, [(chunk, global_cost_matrix.ref2comp_id_map, gids) for chunk in video_chunks])

            # video_results is a list[HOTA_DATA]
            
            # Organize results into per-video structure
            per_frame_hota_data = {res[0].video_id: res for res in video_results}
            per_video_hota_data = {res[0].video_id: merge_hota_data(res, restrict_box_hashes_to_per_frame) for res in video_results}
    else:
        for video_id, cm_values in id_similarity_per_video.items():
            # Process frames for this video sequentially
            if id_alignment_method == 'per_video':
                frame_dat = build_HOTA_objects(cm_values, per_video_cost_matrices[video_id].ref2comp_id_map, gids)
            elif id_alignment_method == 'per_frame':
                frame_dat = build_HOTA_objects(cm_values, None, gids)
            else:
                frame_dat = build_HOTA_objects(cm_values, global_cost_matrix.ref2comp_id_map, gids)

            per_frame_hota_data[video_id] = frame_dat
            per_video_hota_data[video_id] = merge_hota_data(frame_dat, restrict_box_hashes_to_per_frame)

    print(f"  took: {time.time() - st} seconds")


    # ************************************
    # Merge the per-frame hota data together into the global HOTA metric data class
    # ************************************
    st = time.time()
    print(f"Merging HOTA data")
    global_hota_data = merge_hota_data(list(per_video_hota_data.values()), restrict_box_hashes_to_per_frame)
    global_hota_data.video_id = None  # remove the video_id from the global HOTA_DATA object
    print(f"  took: {time.time() - st} seconds")


    # ************************************
    # Plot the HOTA values from each frame
    # ************************************
    if output_dir is not None:
        # TODO add an option to save only per video data, instead of per frame
        st = time.time()
        os.makedirs(output_dir, exist_ok=True)
        
        df_source_list = []
        # Use the already structured per_video_hota_data instead of reconstructing it
        for video_id, video_data in per_video_hota_data.items():
            # Convert the list of HOTA_DATA for this video to a DataFrame
            df_source_list.append(video_data.get_dict())
        # Save the DataFrame to a parquet file in the output directory
        df = pd.DataFrame(df_source_list)
        output_file = os.path.join(output_dir, f'metrics_per_video.parquet')
        df.to_parquet(output_file, index=False)
        # df.to_csv(output_file.replace('.parquet', '.csv'), index=False)


        df_source_list = []
        # Use the already structured per_video_hota_data instead of reconstructing it
        for video_id, frame_data in per_frame_hota_data.items():
            # Convert the list of HOTA_DATA for this video to a DataFrame
            for frame_dat in frame_data:
                df_source_list.append(frame_dat.get_dict())
        # Save the DataFrame to a parquet file in the output directory
        df = pd.DataFrame(df_source_list)
        output_file = os.path.join(output_dir, f'metrics_per_frame.parquet')
        df.to_parquet(output_file, index=False)
        # df.to_csv(output_file.replace('.parquet', '.csv'), index=False)

        df = pd.DataFrame(global_hota_data.get_dict())
        output_file = os.path.join(output_dir, f'hota.parquet')
        df.to_parquet(output_file, index=False)
        df.to_csv(output_file.replace('.parquet', '.csv'), index=False, float_format='%.4f')

        # plot_hota(global_hota_data, output_dir)
            
        print(f"HOTA per-frame metric data write took: {time.time() - st} seconds")
   

    
        

    nb_frames = sum([len(v) for v in id_similarity_per_video.values()])
    print(f"Total time taken: {time.time() - start_time} seconds")
    print(f"Number of frames: {nb_frames}")
    print(f"fps: {nb_frames / (time.time() - start_time)}")
    

    # The HOTA results consist of a series of thresholds for alignment for each metric.
    # So each value in the global_hota_data is a list of values, one per threshold.
    return global_hota_data, per_video_hota_data, per_frame_hota_data
