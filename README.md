# HOTA-ReID: Accelerated Higher Order Tracking Accuracy for Re-Identification


**HOTA-ReID** is a modified version of the Higher Order Tracking Accuracy (HOTA) metric specifically designed to support Re-Identification (ReID) problems while providing significant performance improvements through parallel processing acceleration.

### Key Features

- **ReID-Aware Evaluation**: Handles identity switches and re-appearances common in ReID scenarios
- **Parallel Processing**: Multi-threaded computation for faster evaluation
- **Flexible ID Assignment**: Flexible ID assignment per frame, video, and global
- **Extraneous Box Handling**: Optional removal of comparison ids which don't have an assignment to a ground truth id, with those FP tracked separately. 
- **Flexible ID Assignment Cost**: ID assignment cost can be box IOU or L2 distance in Lat/Long/Alt space.

## Installation

Using uv:
```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install reid_hota
```


Or from source:

```bash
git clone https://github.com/usnistgov/reid_hota.git
cd hota_reid
uv venv --python=3.12
source .venv/bin/activate
uv sync
```

## Quick Start

### Basic Usage

```python
from reid_hota import HOTAReIDEvaluator, HOTAConfig
# create a reference and comparison dictionary of pandas dataframes. Each dataframe contains all detection boxes from that video.

input_dir = "./examples"
gt_fp = os.path.join(input_dir, 'ref')
pred_fp = os.path.join(input_dir, 'comp')
fns = [fn for fn in os.listdir(gt_fp) if fn.endswith('.csv')]

ref_dfs = {}
comp_dfs = {}
for fn in fns:
    gt_df = pd.read_csv(os.path.join(gt_fp, fn))
    pred_df = pd.read_csv(os.path.join(pred_fp, fn))

    ref_dfs[fn.replace('.csv', '')] = gt_df
    comp_dfs[fn.replace('.csv', '')] = pred_df

# create the Config controlling the Metric calculation
config = HOTAConfig(id_alignment_method='global', similarity_metric='iou', purge_non_matched_comp_ids=True)
# create the evaluator
evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
# evaluate on data
evaluator.evaluate(ref_dfs, comp_dfs)  # computes HOTA metrics
# extract results
# returns a dict of HOTA values
global_hota_data = evaluator.get_global_hota_data()
# returns a dict[dict] where outer dict is keyed on video names (identical to ref_dfs or comp_dfs) 
# the inner dict contains HOTA values
per_video_hota_data = evaluator.get_per_video_hota_data()
# returns a dict[dict[dict]] where outer dict is keyed on video names (identical to ref_dfs or comp_dfs)
# the second layer is keyed on frames (the contents of the frame column in ref_dfs or comp_dfs)
# and the inner dict contains HOTA values
per_frame_hota_data = evaluator.get_per_frame_hota_data()

print(f"HOTA-ReID Score: {global_hota_data['HOTA']:.3f}")
```

### Example Data

The input contained in ref_dfs and comp_dfs consists of a dictionary of pandas dataframes. Each dict entry corresponds to a video, and contains a pandas dataframe with all the detections/boxes within that video.
Traditionally, the set of video keys in the ref_dfs, and comp_dfs dictionaries would be identical, but if not, reid_hota computes over the union of the two sets of dictionary keys. Usually the dictionary keys refer to the video names, but any valid python dict key is acceptable. 

Each dataframe has the following minimum required columns:
```python
['frame', 'object_id', 'x', 'y', 'w', 'h', 'class_id']
```

```csv
frame,object_id,x,y,w,h,class_id
0,3,1596,906,123,163,1
1,3,1598,914,135,156,1
2,3,1602,926,144,144,1
```


### Lat/Long/Alt Assignment Cost 
In addition to the traditional IOU cost between boxes, reid_hota supports performing detection assignment in lat/long and lat/long/alt space.

If `HOTAConfig(similarity_metric='latlon')` then in addition to the normal columns, `['lat', 'lon']` are required.

If  `HOTAConfig(similarity_metric='latlonalt')` then the required columns include `['lat', 'lon', 'alt']`

### Keeping Track of Errors

If `HOTAConfig(track_fp_fn_tp_box_hashes=True)` then the column `['box_hash']` is also required, so reid_hota has a per-box hash to keep track of for later grouping into TP, FP, FN.

The full set of allowable input dataframe columns is:
`['frame', 'object_id', 'x', 'y', 'w', 'h', 'class_id', 'lat', 'lon', 'alt', 'box_hash']`

### HOTAConfig Options

```python
class HOTAConfig:
    """
    Configuration for HOTA calculation.
    
    This class defines all parameters needed for computing HOTA metrics,
    including alignment methods, similarity metrics, and filtering options.
    """
    
    class_ids: Optional[List[int]] = None
    """List of class IDs to evaluate. If None, all classes are evaluated."""
    
    gids: Optional[List[int]] = None  
    """Ground truth IDs to use for evaluation. If provided, all other IDs are ignored."""
    
    id_alignment_method: Literal['global', 'per_video', 'per_frame'] = 'global'
    """Method for aligning IDs between reference and comparison data:
    - 'global': Align IDs across all videos globally
    - 'per_video': Align IDs separately for each video  
    - 'per_frame': Align IDs separately for each frame
    """
    
    track_fp_fn_tp_box_hashes: bool = False
    """Whether to track box hashes for detailed FP/FN/TP analysis."""
    
    purge_non_matched_comp_ids: bool = False
    """Whether to remove non-matched comparison IDs to reduce FP counts 
    for data without full dense annotations. Purged ids are counted in an unmatched_FP field."""
    
    iou_thresholds: NDArray[np.float64] = field(default_factory=lambda: np.arange(0.1, 0.99, 0.1))
    """Array of IoU thresholds to evaluate at."""
    
    similarity_metric: Literal['iou', 'latlon', 'latlonalt'] = 'iou'
    """Similarity metric to use:
    - 'iou': Intersection over Union for bounding boxes
    - 'latlon': L2 distance for lat/lon coordinates
    - 'latlonalt': L2 distance for lat/lon/alt coordinates
    """
```

### Global Outputs

Once a call to `.evalauate(ref_dfs, comp_dfs)` has been made, the `evaluator` object contains all HOTA results.

Three sets of evaluation results are generated, first is the global (across all videos) HOTA ReID metrics.
Additionally, there is a per_video HOTA data, and a per_frame HOTA data.

To access the results, use `evaluator.get_global_hota_data()` (or per_video or per_frame).
This will return a python dictionary. 

```python
# create the evaluator
evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
# evaluate on data
evaluator.evaluate(ref_dfs, comp_dfs)  # computes HOTA metrics
# extract results
# returns a dict of HOTA values
global_hota_data = evaluator.get_global_hota_data()
# returns a dict[dict] where outer dict is keyed on video names (identical to ref_dfs or comp_dfs) 
# the inner contains HOTA values
per_video_hota_data = evaluator.get_per_video_hota_data()
# returns a dict[dict[dict]] where outer dict is keyed on video names (identical to ref_dfs or comp_dfs)
# the second layer is keyed on frames (the contents of the frame column in ref_dfs or comp_dfs)
# and the inner contains HOTA values
per_frame_hota_data = evaluator.get_per_frame_hota_data()
```

This results dictionary will have the following structure:
```python
def get_dict(self) -> dict:
    """Get dictionary representation of HOTA data."""
    global_hota_data = {
        'IOU Thresholds': np.array(len(self.iou_thresholds)),
        'video_id': Optional[str],
        'frame': Optional[str],
        'TP': np.array(len(self.iou_thresholds)),
        'FN': np.array(len(self.iou_thresholds))
        'FP': np.array(len(self.iou_thresholds))
        'UnmatchedFP': int
        'LocA': np.array(len(self.iou_thresholds))
        'HOTA': np.array(len(self.iou_thresholds))
        'AssA': np.array(len(self.iou_thresholds))
        'AssRe': np.array(len(self.iou_thresholds))
        'AssPr': np.array(len(self.iou_thresholds))
        'DetA': np.array(len(self.iou_thresholds))
        'DetRe': np.array(len(self.iou_thresholds))
        'DetPr': np.array(len(self.iou_thresholds))
        'OWTA': np.array(len(self.iou_thresholds))
        'IDF1': np.array(len(self.iou_thresholds))}
    if track_hashes:
        global_hota_data['FP_hashes'] = list(hashable)
        global_hota_data['FN_hashes'] = list(hashable)
        global_hota_data['TP_hashes'] = list(hashable)
    return global_hota_data
```

The IOU thresholds match what you specified in the `HOTAConfig`.
The video_id will be None for global results, or have the video id (key into ref_dfs dict) for `per_video` and `per_frame` results.

- `frame` will be None unless its the per_frame results.
- `TP`, `FP`, `FN` will contain TP/FP/FN counts per IOU threshold. UnmatchedFP will contain any FP counts for which there was no assignment to a ground truth track. This behavior can be controlled using `purge_non_matched_comp_ids` in the config.
- `LocA` is effectivly the average matching box IOU.
- `HOTA` is the final composite metric. HOTA = sqrt(DetA * AssA)
- `AssA` is the association accuracy.
- `AssRe` is the association recall.
- `AssPr` is the association precision.
- `DetA` is the detection accuracy.
- `DetRe` is the detection recall.
- `DetPr` is the detection precision.
- `OWTA` is OWTA = sqrt(DetRe * AssA).
- `IDF1` is the IDF1 metric = TP / (TP + (0.5 * FN) + (0.5 * FP)).

The hashes will only exist if the `box_hash` column is present and the config has `track_fp_fn_tp_box_hashes` enabled.
- `FP_hashes` is a the list of `box_hash` for all false positives. These are only kept in the per_frame data.
- `FN_hashes` is a the list of `box_hash` for all false negatives. These are only kept in the per_frame data.
- `TP_hashes` is a the list of `box_hash` for all true positives. These are only kept in the per_frame data.

### Lat/Lon Distance Similarities

When using a `HOTAConfig(similarity_metric='latlon')` similarity score, the L2 distance between points is used for similarity. That L2 is converted into a similarity cost function [0, 1] as follows:

```python
# Calculate squared differences for all pairs
squared_diff = np.sum((points1 - points2) ** 2, axis=2)
# Take square root to get Euclidean distance
distances = np.sqrt(squared_diff)
# use exp(-dist) to convert [0, inf] into [0, 1] with smaller distances being closer to similarity 1
# dist/10 normalizes the L2 values over human relavant distances nicely into [0, 1] scores. 
similarities = np.exp(-distances / 10)
```


### HOTA Metrics (and sub-metrics) are Vectors

The `HOTA` metric results in a vector of numbers of length IOU_Thresholds. Each metric value in that list represents thresholding the cost matrix at the specific IOU Threshold (between 0 and 1). So if you want the HOTA metric for an IOU threshold of 0.5:

```python
idx = global_hota_data['IOU Thresholds'] == 0.5
hota_value = global_hota_data['HOTA'][idx]
```





## License

This software was developed by employees of the National Institute of
Standards and Technology (NIST), an agency of the Federal Government and is
being made available as a public service. Pursuant to title 17 United States
Code Section 105, works of NIST employees are not subject to copyright
protection in the United States.  This software may be subject to foreign
copyright.  Permission in the United States and in foreign countries, to the
extent that NIST may hold copyright, to use, copy, modify, create derivative
works, and distribute this software and its documentation without fee is hereby
granted on a non-exclusive basis, provided that this notice and disclaimer of
warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER
EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM
INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE
SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT
SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT,
INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM,
OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON
WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED
BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED
FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES
PROVIDED HEREUNDER.

To see the latest statement, please visit:
[https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software](Copyright, Fair Use, and Licensing Statements for SRD, Data, and Software)

## Acknowledgments

- Original HOTA implementation by [Jonathon Luiten](https://github.com/JonathonLuiten/TrackEval)

## Contact

- **Author**: Michael Majurski
- **Email**: michael.majurski@nist.gov
- **Project Link**: https://github.com/usnistgov/reid_hota





