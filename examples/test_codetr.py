import os
import time
import numpy as np
import pandas as pd

from reid_hota import fast_hota as fh
from reid_hota.fast_hota import HOTA_DATA



ifp = '/home/mmajursk/github/Co-DETR/'

ref_dfs = {}
comp_dfs = {}
avi_files = [f for f in os.listdir(ifp) if f.endswith('.avi')]
mp4_files = [f for f in os.listdir(ifp) if f.endswith('.mp4')]
# video_files = avi_files # + mp4_files
video_files = mp4_files

comp_ifp = '/home/mmajursk/github/Co-DETR/results-120-nontiled/'
# comp_ifp = '/home/mmajursk/github/Co-DETR/results-120-tiled-0.1/'
# comp_ifp = '/home/mmajursk/github/Co-DETR/results-120-tiled-0.2/'
# comp_ifp = '/home/mmajursk/github/Co-DETR/results-120-tiled-0.3/'
# comp_ifp = '/home/mmajursk/github/Co-DETR/results-120-tiled-0.5/'
# comp_ifp = '/home/mmajursk/github/Co-DETR/results-f0.5-120-nontiled/'


frame_skip = 300  # 1
for video_file in video_files:
    

    if video_file.endswith('.avi'):
        gt_fp = '/home/mmajursk/data/vlincs/meva-rev2/'
    else:
        if 'Tc1' in video_file:
            gt_fp = '/home/mmajursk/data/vlincs/mitre-pilot-data-v3.1.3.5/Tc1/'
        else:
            gt_fp = '/home/mmajursk/data/vlincs/mitre-pilot-data-v3.1.3.5/Tc3/'

    fn = os.path.basename(video_file).replace('.avi', '').replace('.mp4', '')
    ref_fp = os.path.join(gt_fp, fn, 'gt.parquet')
    comp_fp = os.path.join(comp_ifp, f'{fn}_output.csv')
    if os.path.exists(ref_fp) and os.path.exists(comp_fp):
        print(video_file)
        print(f"ref_fp: {ref_fp}")
        st = time.time()
        ref_df = pd.read_parquet(ref_fp)
        ref_df = ref_df[ref_df['frame_id'] % frame_skip == 0]

        # Check which frames need updating (where object_ids are not unique per frame)
        frame_counts = ref_df.groupby('frame_id').size()
        frame_unique_objects = ref_df.groupby('frame_id')['object_id'].nunique()
        frames_to_update = frame_counts[frame_counts != frame_unique_objects].index

        if len(frames_to_update) > 0:
            # Only update frames that need it
            mask = ref_df['frame_id'].isin(frames_to_update)
            ref_df.loc[mask, 'object_id'] = ref_df[mask].groupby('frame_id').cumcount()

        if video_file.endswith('.avi'):
            # Fix the class id for the MEVA dataset
            ref_df.loc[ref_df['class_id'] == 1, 'class_id'] = 0

        print(f"    time: {time.time() - st}")

        st = time.time()
        print(f"comp_fp: {comp_fp}")
        comp_df = pd.read_csv(comp_fp)
        comp_df['frame_id'] = comp_df['frame']
        comp_df = comp_df[comp_df['frame_id'] % frame_skip == 0]
        comp_df['object_id'] = comp_df.groupby('frame_id').cumcount()


        ref_df.to_csv('./ref_df.csv')
        comp_df.to_csv('./comp_df.csv')

        ref_dfs[fn] = ref_df
        comp_dfs[fn] = comp_df
    else:
        print(f"Skipping {video_file} because gt or comp does not exist")


            

print(f"ref_dfs: {len(ref_dfs)}")
print(f"comp_dfs: {len(comp_dfs)}")

combined_hota, per_video_hota_data, per_frame_hota = fh.compute_hota(ref_dfs, comp_dfs, n_workers=20, id_alignment_method='per_frame', output_dir='./hota_plots', similarity_metric='iou', class_ids=[0])

print("combined HOTA data keys (at 0.5):")
idx = np.where(HOTA_DATA.array_labels == 0.5)[0][0]
for key in combined_hota.data.keys():
    if key == 'matches_counts':
        continue
    print(f"{key}: {combined_hota.data[key][idx]}")
    # print(f"{key}: {combined_hota.data[key]}")

