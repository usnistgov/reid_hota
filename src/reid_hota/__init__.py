"""
Modified HOTA (Higher Order Tracking Accuracy) extended for ReID evaluation.

A fast, parallel implementation of HOTA metrics for re-identification and tracking evaluation.
"""

__version__ = "0.3.6"

from .config import HOTAConfig
from .hota_data import HOTAData
from .reid_hota import HOTAReIDEvaluator

__all__ = [
    "HOTAReIDEvaluator",
    "HOTAData",
    "HOTAConfig"
]
