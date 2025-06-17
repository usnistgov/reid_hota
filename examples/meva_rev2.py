import os
import sys
import numpy as np
import pandas as pd

# Add the src directory to Python path to import local reid_hota
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


from reid_hota import HOTAReIDEvaluator, HOTAConfig



def load_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = '/home/mmajursk/usnistgov/reid_hota/tests/data/meva_rid_short/'
    gt_fp = os.path.join(script_dir, 'ref')
    pred_fp = os.path.join(script_dir, 'comp')

    fns = [fn for fn in os.listdir(gt_fp) if fn.endswith('.csv')]
    ref_dfs = {}
    comp_dfs = {}
    for fn in fns:
        gt_df = pd.read_csv(os.path.join(gt_fp, fn))
        pred_df = pd.read_csv(os.path.join(pred_fp, fn))

        if 'w' not in gt_df.columns:
            gt_df['w'] = gt_df['width']
        if 'w' not in pred_df.columns:
            pred_df['w'] = pred_df['width']
        if 'h' not in gt_df.columns:
            gt_df['h'] = gt_df['height']
        if 'h' not in pred_df.columns:
            pred_df['h'] = pred_df['height']
        

        # add fake lat/lon/alt to the data
        gt_df['lat'] = np.random.rand(len(gt_df)) * 10
        pred_df['lat'] = np.random.rand(len(pred_df)) * 10

        gt_df['lon'] = np.random.rand(len(gt_df)) * 10
        pred_df['lon'] = np.random.rand(len(pred_df)) * 10

        gt_df['alt'] = np.random.rand(len(gt_df)) * 10
        pred_df['alt'] = np.random.rand(len(pred_df)) * 10

        # Create fake hashes to utilize the FP/FN/TP tracking per frame
        pred_df['box_hash'] = pred_df.apply(lambda row: str(hash(f"frame:{row['frame']} object_id:{row['object_id']} x:{row['x']} y:{row['y']} w:{row['w']} h:{row['h']} class_id:{row['class_id']} lat:{row['lat']} lon:{row['lon']} alt:{row['alt']}")), axis=1)
        gt_df['box_hash'] = gt_df.apply(lambda row: str(hash(f"frame:{row['frame']} object_id:{row['object_id']} x:{row['x']} y:{row['y']} w:{row['w']} h:{row['h']} class_id:{row['class_id']} lat:{row['lat']} lon:{row['lon']} alt:{row['alt']}")), axis=1)


        ref_dfs[fn.replace('.csv', '')] = gt_df
        comp_dfs[fn.replace('.csv', '')] = pred_df

    return ref_dfs, comp_dfs

    
def hota_meva_subset():
    """Test the HOTA metric computation."""

    ref_dfs, comp_dfs = load_data()

    # config = HOTAConfig(id_alignment_method='global', similarity_metric='iou', purge_non_matched_comp_ids=True)
    config = HOTAConfig(id_alignment_method='global', similarity_metric='latlon', purge_non_matched_comp_ids=True)
    evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
    evaluator.evaluate(ref_dfs, comp_dfs)
    global_hota_data = evaluator.get_global_hota_data()
    per_video_hota_data = evaluator.get_per_video_hota_data()
    per_frame_hota_data = evaluator.get_per_frame_hota_data()

    evaluator.export_to_file('./plots')


    print("combined HOTA data keys (at 0.5):")
    idx = np.where(global_hota_data['IOU Thresholds'] == 0.5)[0][0]
    hota_value = global_hota_data['HOTA'][global_hota_data['IOU Thresholds'] == 0.5]
    for key in global_hota_data.keys():
        val = global_hota_data[key]
        if val is not None:
            if isinstance(val, np.ndarray):
                print(f"{key}: {global_hota_data[key][idx]}")
            else:
                print(f"{key}: {global_hota_data[key]}")

        
        

if __name__ == "__main__":

    hota_meva_subset()
