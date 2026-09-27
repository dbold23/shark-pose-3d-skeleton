"""Tests for YOLOKeypointBackend.

The actual ultralytics YOLO is not exercised here — we mock the model so
tests run without ultralytics installed and without GPU. The goal is to
verify our adapter logic: result unpacking, multiple-detection selection,
no-detection fallback, keypoint/confidence shape contracts.
"""
from __future__ import annotations
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

def _mock_tensor(arr: np.ndarray):
    """Build a mock object that has .cpu().numpy() returning arr."""
    ...

def _make_result(keypoints_xy: np.ndarray | None, keypoints_conf: np.ndarray | None, boxes_conf: np.ndarray | None):
    """Build a mock ultralytics Results object."""
    ...

@pytest.fixture
def mock_ultralytics(tmp_path):
    """Patch ultralytics.YOLO with a controllable mock; yield the predict callable."""
    ...

class TestYOLOKeypointBackend:

    def test_missing_weights_raises(self, tmp_path):
        ...

    def test_single_detection_returns_correct_shapes(self, mock_ultralytics):
        ...

    def test_multiple_detections_picks_highest_box_conf(self, mock_ultralytics):
        ...

    def test_zero_detections_returns_zeros(self, mock_ultralytics):
        ...

    def test_no_keypoints_attr_returns_zeros(self, mock_ultralytics):
        ...

    def test_empty_results_list_returns_zeros(self, mock_ultralytics):
        ...

    def test_wrong_keypoint_count_pads_or_truncates(self, mock_ultralytics):
        ...

    def test_predict_passes_thresholds_and_device(self, mock_ultralytics):
        ...
