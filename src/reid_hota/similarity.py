"""
Similarity primitives used by the cost-matrix pipeline.

Three functions, one per `HOTAConfig.similarity_metric`:
- ``calculate_box_ious``     — IoU on bounding boxes (xyxy / xywh)
- ``calculate_latlon_l2``    — exp(-ECEF distance) on (lat, lon)
- ``calculate_latlonalt_l2`` — exp(-ECEF distance) on (lat, lon, alt)

All return an ``(N, M)`` array of pairwise similarity in [0, 1].
The geographic functions reproject to ECEF meters (WGS84 → EPSG:4978) so the
exponential decay is interpretable in meters regardless of latitude.
"""
import numpy as np
from pyproj import Transformer

from .hota_errors import UnsupportedBoxFormatError

# WGS84 geographic (lat, lon, alt) → WGS84 geocentric Cartesian (ECEF x, y, z), all in meters
_WGS84_TO_ECEF = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=False)

# Decay constant for exp(-d / ECEF_L2_DECAY_METERS): similarity = 0.37 at this distance (meters)
ECEF_L2_DECAY_METERS = 10.0


def calculate_box_ious(bboxes1: np.ndarray, bboxes2: np.ndarray, box_format='xywh'):
    """
    Vectorized pairwise IoU between two arrays of bounding boxes.

    Args:
        bboxes1: Array of shape (N, 4) containing first set of bounding boxes
        bboxes2: Array of shape (M, 4) containing second set of bounding boxes
        box_format: 'xywh' (x, y, width, height) or
                    'x0y0x1y1' (x_min, y_min, x_max, y_max) (alias 'xyxy')

    Returns:
        Array of shape (N, M) containing pairwise IoU values
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))

    if box_format == 'xywh':
        boxes1 = np.column_stack([
            bboxes1[:, 0], bboxes1[:, 1],
            bboxes1[:, 0] + bboxes1[:, 2], bboxes1[:, 1] + bboxes1[:, 3]
        ])
        boxes2 = np.column_stack([
            bboxes2[:, 0], bboxes2[:, 1],
            bboxes2[:, 0] + bboxes2[:, 2], bboxes2[:, 1] + bboxes2[:, 3]
        ])
    elif box_format in ('x0y0x1y1', 'xyxy'):
        boxes1, boxes2 = bboxes1, bboxes2
    else:
        raise UnsupportedBoxFormatError(box_format)

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    left = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    top = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    right = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    bottom = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    # Negative width/height (degenerate boxes) clamp to 0 intersection.
    width = np.maximum(0, right - left)
    height = np.maximum(0, bottom - top)
    intersection = width * height
    union = boxes1_area[:, None] + boxes2_area[None, :] - intersection

    epsilon = 1e-8  # numerical guard for degenerate unions
    return np.divide(intersection, np.maximum(union, epsilon))


def _latlon_to_ecef(latlonalt: np.ndarray) -> np.ndarray:
    """Convert geographic coordinates to ECEF meters.

    Args:
        latlonalt: shape (N, 2) or (N, 3) with columns [lat_deg, lon_deg, alt_m].
                   Altitude defaults to 0 when not provided.
    Returns:
        Shape (N, 3) array of [x, y, z] in meters.
    """
    alt = latlonalt[:, 2] if latlonalt.shape[1] >= 3 else np.zeros(len(latlonalt))
    x, y, z = _WGS84_TO_ECEF.transform(latlonalt[:, 0], latlonalt[:, 1], alt)
    return np.column_stack([x, y, z])


def calculate_latlonalt_l2(latlonalt1: np.ndarray, latlonalt2: np.ndarray):
    """Pairwise similarity from ECEF L2 distance on (lat, lon, alt) points.

    Args:
        latlonalt1: shape (N, 3) with [lat_deg, lon_deg, alt_m]
        latlonalt2: shape (M, 3) with [lat_deg, lon_deg, alt_m]
    Returns:
        Shape (N, M) similarity in [0, 1] via exp(-d / ECEF_L2_DECAY_METERS).
    """
    ecef1 = _latlon_to_ecef(latlonalt1)
    ecef2 = _latlon_to_ecef(latlonalt2)
    diff = ecef1[:, np.newaxis, :] - ecef2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.exp(-distances / ECEF_L2_DECAY_METERS)


def calculate_latlon_l2(latlon1: np.ndarray, latlon2: np.ndarray):
    """Pairwise similarity from ECEF L2 distance on (lat, lon) points (alt=0).

    Args:
        latlon1: shape (N, 2) with [lat_deg, lon_deg]
        latlon2: shape (M, 2) with [lat_deg, lon_deg]
    Returns:
        Shape (N, M) similarity in [0, 1] via exp(-d / ECEF_L2_DECAY_METERS).
    """
    ecef1 = _latlon_to_ecef(latlon1)
    ecef2 = _latlon_to_ecef(latlon2)
    diff = ecef1[:, np.newaxis, :] - ecef2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.exp(-distances / ECEF_L2_DECAY_METERS)
