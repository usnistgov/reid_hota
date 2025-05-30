import numpy as np
from typing import Optional
from numpy.typing import NDArray
from dataclasses import dataclass, field




@dataclass
class HOTAConfig:
    """Configuration for HOTA calculation"""
    
    class_ids: Optional[list[np.dtype[np.object_]]] = None
    gids: Optional[list[np.dtype[np.object_]]] = None
    id_alignment_method: str = 'global'
    track_fp_fn_tp_box_hashes: bool = False
    purge_non_matched_comp_ids: bool = False
    iou_thresholds: NDArray[np.float64] = field(default_factory=lambda: np.arange(0.1, 0.99, 0.1))
    similarity_metric: str = 'iou'

    def validate(self):
        ID_ALIGNMENT_METHODS = ['global','per_video','per_frame']
        SIMILARITY_METRICS = ['iou', 'latlonalt']

        assert self.id_alignment_method in ID_ALIGNMENT_METHODS, f"id_alignment_method must be one of: {ID_ALIGNMENT_METHODS}"
        assert self.similarity_metric in SIMILARITY_METRICS, f"similarity_metric must be one of: {SIMILARITY_METRICS}"