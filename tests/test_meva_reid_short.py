import pytest
import os
import sys
import numpy as np
import pandas as pd
import json
# from pathlib import Path
import multiprocessing as mp

@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing():
    """Set multiprocessing to use spawn method to avoid fork warnings in pytest"""
    mp.set_start_method('spawn', force=True)


# Add the src directory to Python path to import local reid_hota
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reid_hota import HOTAReIDEvaluator, HOTAConfig
from test_utils import validate_results



@pytest.fixture
def tracking_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Pytest fixture that creates tracking data for testing."""
    gt_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'ref')
    pred_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'comp')

    fns = [fn for fn in os.listdir(gt_fp) if fn.endswith('.csv')]
    ref_dfs = {}
    comp_dfs = {}
    for fn in fns:
        gt_df = pd.read_csv(os.path.join(gt_fp, fn))
        pred_df = pd.read_csv(os.path.join(pred_fp, fn))

        
        if 'frame' in gt_df.columns:
            gt_df['frame'] = gt_df['frame']
        if 'frame' in pred_df.columns:
            pred_df['frame'] = pred_df['frame']
        # Rename width column to w if it exists
        if 'width' in gt_df.columns:
            gt_df['w'] = gt_df['width']
        if 'width' in pred_df.columns:
            pred_df['w'] = pred_df['width']
        
        # Rename height column to h if it exists
        if 'height' in gt_df.columns:
            gt_df['h'] = gt_df['height']
        if 'height' in pred_df.columns:
            pred_df['h'] = pred_df['height']

        gt_df['lat'] = np.random.rand(len(gt_df)) * 10
        pred_df['lat'] = np.random.rand(len(pred_df)) * 10

        gt_df['lon'] = np.random.rand(len(gt_df)) * 10
        pred_df['lon'] = np.random.rand(len(pred_df)) * 10

        gt_df['alt'] = np.random.rand(len(gt_df)) * 10
        pred_df['alt'] = np.random.rand(len(pred_df)) * 10

        # Create a new random box for each frame
        # Create new rows for each frame this tests the FP purge
        # new_rows = []
        # for frame in gt_df['frame'].unique():
        #     new_row = {
        #         'frame': frame,
        #         'object_id': int(np.random.rand() * 100000) + 100000,
        #         'x': np.random.rand() * 1920,  # Random x coordinate
        #         'y': np.random.rand() * 1080,  # Random y coordinate
        #         'w': np.random.rand() * 100 + 50,  # Random width between 50-150
        #         'h': np.random.rand() * 100 + 50,  # Random height between 50-150
        #         'class_id': 1, 
        #         'lat': np.random.rand() * 10,
        #         'lon': np.random.rand() * 10,
        #         'alt': np.random.rand() * 10
        #     }
        #     new_rows.append(new_row)
        
        # # Add all new rows at once to both dataframes
        # new_df = pd.DataFrame(new_rows)
        # pred_df = pd.concat([pred_df, new_df], ignore_index=True)

        # hash(f"frame:{frame} object_id:{object_id} x:{x} y:{y} w:{w} h:{h} class_id:{class_id} lat:{lat} lon:{lon} alt:{alt}")
        pred_df['box_hash'] = pred_df.apply(lambda row: str(hash(f"frame:{row['frame']} object_id:{row['object_id']} x:{row['x']} y:{row['y']} w:{row['w']} h:{row['h']} class_id:{row['class_id']} lat:{row['lat']} lon:{row['lon']} alt:{row['alt']}")), axis=1)
        gt_df['box_hash'] = gt_df.apply(lambda row: str(hash(f"frame:{row['frame']} object_id:{row['object_id']} x:{row['x']} y:{row['y']} w:{row['w']} h:{row['h']} class_id:{row['class_id']} lat:{row['lat']} lon:{row['lon']} alt:{row['alt']}")), axis=1)

        


        ref_dfs[fn.replace('.csv', '')] = gt_df
        comp_dfs[fn.replace('.csv', '')] = pred_df

    # Print statistics about the number of frames in each video
    frame_counts = []
    for video_id in ref_dfs:
        unique_frames = ref_dfs[video_id]['frame'].nunique()
        frame_counts.append(unique_frames)

    # Print class IDs present in both ref_dfs and comp_dfs
    ref_class_ids = set()
    comp_class_ids = set()
    
    for video_id in ref_dfs:
        ref_class_ids.update(ref_dfs[video_id]['class_id'].unique())
    for video_id in comp_dfs:
        comp_class_ids.update(comp_dfs[video_id]['class_id'].unique())
        
    print("Class IDs in reference data:", sorted(list(ref_class_ids)))
    print("Class IDs in comparison data:", sorted(list(comp_class_ids)))

    
    if frame_counts:
        print(f"Frame count statistics:")
        print(f"  Min frames: {min(frame_counts)}")
        print(f"  Max frames: {max(frame_counts)}")
        print(f"  Mean frames: {sum(frame_counts)/len(frame_counts):.2f}")
        print(f"  total frames: {sum(frame_counts)}")
    return ref_dfs, comp_dfs




       


        


class TestHOTA_meva_reid_short_global_id_alignment:
    """Test class for HOTA metric functionality."""
    
    def test_compute_hota(self, tracking_data):
        """Test the HOTA metric computation."""
        ref_dfs, comp_dfs = tracking_data

        config = HOTAConfig(id_alignment_method='global', similarity_metric='iou')
        evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
        evaluator.evaluate(ref_dfs, comp_dfs)
        global_hota_data = evaluator.get_global_hota_data()
        per_video_hota_data = evaluator.get_per_video_hota_data()
        per_frame_hota_data = evaluator.get_per_frame_hota_data()
        
        gt_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'results_global_id_alignment.json')
        validate_results(global_hota_data, gt_fp)  # raises AssertionError if any keys fail

        


class TestHOTA_meva_reid_short_video_id_alignment:
    """Test class for HOTA metric functionality."""
    
    def test_compute_hota(self, tracking_data):
        """Test the HOTA metric computation."""
        ref_dfs, comp_dfs = tracking_data

        config = HOTAConfig(id_alignment_method='per_video', similarity_metric='iou')
        evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
        evaluator.evaluate(ref_dfs, comp_dfs)
        global_hota_data = evaluator.get_global_hota_data()
        per_video_hota_data = evaluator.get_per_video_hota_data()
        per_frame_hota_data = evaluator.get_per_frame_hota_data()

        gt_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'results_video_id_alignment.json')
        validate_results(global_hota_data, gt_fp)  # raises AssertionError if any keys fail
        

class TestHOTA_meva_reid_short_frame_id_alignment:
    """Test class for HOTA metric functionality."""
    
    def test_compute_hota(self, tracking_data):
        """Test the HOTA metric computation."""
        ref_dfs, comp_dfs = tracking_data


        config = HOTAConfig(id_alignment_method='per_frame', similarity_metric='iou')
        evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
        evaluator.evaluate(ref_dfs, comp_dfs)
        global_hota_data = evaluator.get_global_hota_data()
        per_video_hota_data = evaluator.get_per_video_hota_data()
        per_frame_hota_data = evaluator.get_per_frame_hota_data()

        gt_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'results_frame_id_alignment.json')
        validate_results(global_hota_data, gt_fp)  # raises AssertionError if any keys fail
        



    
    

    

    




