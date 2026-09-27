"""Tests for active learning V1: FailureAnalyzer and AdaptiveSampler."""
import numpy as np
import pytest
from shark_pose.active_learning.failure_analyzer import FailureAnalyzer, FailureCluster, SampleResult
from shark_pose.active_learning.sampling_strategy import AdaptiveSampler, DEFAULT_MODE_WEIGHTS
from shark_pose.core.skeleton import SHARK_KEYPOINT_SEQUENCE

def _make_result(sample_id: str='synth_0000000', pa_mpjpe: float=10.0, pose_mode: str='cruise', camera_elevation: float=0.0, camera_distance: float=5.0, per_joint_errors: np.ndarray | None=None) -> SampleResult:
    ...

def _make_results_batch(modes_and_errors: list[tuple[str, float]], elevation: float=0.0, distance: float=5.0) -> list[SampleResult]:
    ...

class TestFailureAnalyzer:

    def test_empty_results(self):
        ...

    def test_single_cluster(self):
        ...

    def test_multiple_modes(self):
        ...

    def test_camera_binning(self):
        ...

    def test_distance_binning(self):
        ...

    def test_get_mode_difficulty(self):
        ...

    def test_get_worst_joints(self):
        ...

    def test_worst_joint_per_cluster(self):
        ...

    def test_from_annotation(self):
        ...

    def test_summary_runs(self):
        ...

class TestAdaptiveSampler:

    def test_weights_sum_to_one(self):
        ...

    def test_floor_respected(self):
        ...

    def test_high_temperature_near_uniform(self):
        ...

    def test_low_temperature_favors_hard(self):
        ...

    def test_missing_modes_get_zero_difficulty(self):
        ...

    def test_compute_from_clusters(self):
        ...

    def test_write_and_load_weights(self, tmp_path):
        ...

    def test_summary_runs(self):
        ...
