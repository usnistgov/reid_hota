"""
Modified HOTA (Higher Order Tracking Accuracy) extended for ReID evaluation.

A fast, parallel implementation of HOTA metrics for re-identification and tracking evaluation.
"""

__version__ = "0.3.2"

from .reid_hota import HOTAReIDEvaluator
from .hota_data import HOTAData
from .config import HOTAConfig

__all__ = [
    "HOTAReIDEvaluator",
    "HOTAData", 
    "HOTAConfig"
] 