import numpy as np
from typing import Optional, List, Literal
from numpy.typing import NDArray
from dataclasses import dataclass, field


@dataclass
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

    def validate(self) -> None:
        """Validate configuration parameters."""
        ID_ALIGNMENT_METHODS = ['global','per_video','per_frame']
        SIMILARITY_METRICS = ['iou', 'latlon', 'latlonalt']

        if self.id_alignment_method not in ID_ALIGNMENT_METHODS:
            raise ValueError(f"id_alignment_method must be one of: {ID_ALIGNMENT_METHODS}")
        if self.similarity_metric not in SIMILARITY_METRICS:
            raise ValueError(f"similarity_metric must be one of: {SIMILARITY_METRICS}")
        if len(self.iou_thresholds) == 0:
            raise ValueError("iou_thresholds cannot be empty")
        if not np.all((self.iou_thresholds >= 0) & (self.iou_thresholds <= 1)):
            raise ValueError("iou_thresholds must be in range [0, 1]")