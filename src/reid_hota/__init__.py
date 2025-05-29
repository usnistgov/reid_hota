"""
Modified HOTA (Higher Order Tracking Accuracy) extended for ReID evaluation.

A fast, parallel implementation of HOTA metrics for re-identification and tracking evaluation.
"""

__version__ = "0.1.0"

from .fast_hota import compute_hota
from .hota_data import HOTA_DATA, VideoFrameData, FrameExtractionInputData
from .fast_hota_utils import merge_hota_data, jaccard_cost_matrices
from .cost_matrix import CostMatrixData
from .sparse_matrix import Sparse2DMatrix, Sparse1DMatrix

__all__ = [
    "compute_hota",
    "HOTA_DATA", 
    "CostMatrixData",
    "VideoFrameData",
    "merge_hota_data",
    "jaccard_cost_matrices",
    "Sparse2DMatrix",
    "Sparse1DMatrix",
    "FrameExtractionInputData",
] 