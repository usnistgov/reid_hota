import os

import pytest
from test_utils import validate_results

from reid_hota import HOTAConfig, HOTAReIDEvaluator


@pytest.fixture
def tracking_data(base_tracking_data):
    """Alias of conftest's base_tracking_data — shared meva_rid_short CSV loader."""
    return base_tracking_data


class TestHOTA_meva_reid_short_global_id_alignment:
    """Test class for HOTA metric functionality."""

    def test_compute_hota(self, tracking_data):
        """Test the HOTA metric computation."""
        ref_dfs, comp_dfs = tracking_data

        config = HOTAConfig(id_alignment_method='global', similarity_metric='iou', suppress_print_statements=True, reference_contains_dense_annotations=True)
        evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
        evaluator.evaluate(ref_dfs, comp_dfs)
        global_hota_data = evaluator.get_global_hota_data()
        per_video_hota_data = evaluator.get_per_video_hota_data()
        per_frame_hota_data = evaluator.get_per_frame_hota_data()

        gt_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'results_global_id_alignment.json')
        validate_results(global_hota_data, gt_fp)  # raises AssertionError if any keys fail


class TestHOTA_meva_reid_short_video_id_alignment:
    """Test class for HOTA metric functionality."""

    def test_compute_hota(self, tracking_data):
        """Test the HOTA metric computation."""
        ref_dfs, comp_dfs = tracking_data

        config = HOTAConfig(id_alignment_method='per_video', similarity_metric='iou', suppress_print_statements=True, reference_contains_dense_annotations=True)
        evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
        evaluator.evaluate(ref_dfs, comp_dfs)
        global_hota_data = evaluator.get_global_hota_data()
        per_video_hota_data = evaluator.get_per_video_hota_data()
        per_frame_hota_data = evaluator.get_per_frame_hota_data()

        gt_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'results_video_id_alignment.json')
        validate_results(global_hota_data, gt_fp)  # raises AssertionError if any keys fail


class TestHOTA_meva_reid_short_frame_id_alignment:
    """Test class for HOTA metric functionality."""

    def test_compute_hota(self, tracking_data):
        """Test the HOTA metric computation."""
        ref_dfs, comp_dfs = tracking_data

        config = HOTAConfig(id_alignment_method='per_frame', similarity_metric='iou', suppress_print_statements=True, reference_contains_dense_annotations=True)
        evaluator = HOTAReIDEvaluator(n_workers=20, config=config)
        evaluator.evaluate(ref_dfs, comp_dfs)
        global_hota_data = evaluator.get_global_hota_data()
        per_video_hota_data = evaluator.get_per_video_hota_data()
        per_frame_hota_data = evaluator.get_per_frame_hota_data()

        gt_fp = os.path.join(os.path.dirname(__file__), 'data', 'meva_rid_short', 'results_frame_id_alignment.json')
        validate_results(global_hota_data, gt_fp)  # raises AssertionError if any keys fail
