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
ref_dfs, comp_dfs = load_data()

# create the Config controlling the Metric calculation
config = HOTAConfig(id_alignment_method='global', similarity_metric='iou', purge_non_matched_comp_ids=True)
# create the evaluator
evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
# evaluate on data
evaluator.evaluate(ref_dfs, comp_dfs)
# extract results
global_hota_data = evaluator.get_global_hota_data().get_dict()
per_video_hota_data = evaluator.get_per_video_hota_data()
per_frame_hota_data = evaluator.get_per_frame_hota_data()

print(f"HOTA-ReID Score: {global_hota_data['HOTA']:.3f}")
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





