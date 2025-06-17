"""
Modified HOTA (Higher Order Tracking Accuracy) extended for ReID evaluation.

A fast, parallel implementation of HOTA metrics for re-identification and tracking evaluation.
"""

__version__ = "0.1.3"

from .reid_hota import HOTAReIDEvaluator
from .hota_data import HOTAData, VideoFrameData, FrameExtractionInputData
from .hota_utils import merge_hota_data, jaccard_cost_matrices
from .cost_matrix import CostMatrixData
from .sparse_matrix import Sparse2DMatrix, Sparse1DMatrix
from .config import HOTAConfig

__all__ = [
    "HOTAReIDEvaluator",
    "HOTAData", 
    "CostMatrixData",
    "VideoFrameData",
    "merge_hota_data",
    "jaccard_cost_matrices",
    "Sparse2DMatrix",
    "Sparse1DMatrix",
    "FrameExtractionInputData",
    "HOTAConfig",
] 