"""
Failure analysis for active learning.

Clusters evaluation results by generation metadata (pose mode, camera
elevation, camera distance) and computes per-cluster difficulty scores.
These scores drive the next round of synthetic data generation.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class SampleResult:
    """Per-sample evaluation result with generation metadata."""
    sample_id: str
    pa_mpjpe: float
    per_joint_errors: np.ndarray
    pose_mode: str
    camera_elevation: float
    camera_azimuth: float
    camera_distance: float

    @classmethod
    def from_annotation(cls, sample_id: str, pa_mpjpe: float, per_joint_errors: np.ndarray, annotation: dict) -> SampleResult:
        """Build from a sample ID, error, and annotation JSON dict."""
        ...

@dataclass
class FailureCluster:
    """A cluster of samples grouped by generation metadata."""
    name: str
    sample_count: int
    mean_error: float
    std_error: float
    worst_joint: str
    worst_joint_error: float
    difficulty_score: float
    metadata_profile: Dict

def _bin_elevation(elev: float) -> str:
    ...

def _bin_distance(dist: float) -> str:
    ...

class FailureAnalyzer:
    """Clusters evaluation failures and computes difficulty scores.

    Bins samples by (pose_mode, elevation_bin, distance_bin), computes
    mean PA-MPJPE per bin, and normalizes to a [0, 1] difficulty score.

    Args:
        joint_names: ordered list of joint names (16 entries).
    """

    def __init__(self, joint_names: List[str]):
        ...

    def analyze(self, results: List[SampleResult]) -> List[FailureCluster]:
        """Cluster samples and compute difficulty scores.

        Args:
            results: per-sample evaluation results.

        Returns:
            List of FailureCluster, one per non-empty bin, sorted by
            difficulty_score descending.
        """
        ...

    def get_mode_difficulty(self, results: List[SampleResult]) -> Dict[str, float]:
        """Aggregate difficulty per pose mode (ignoring camera bins).

        Returns:
            Dict mapping mode_name -> difficulty score (0-1).
        """
        ...

    def get_worst_joints(self, results: List[SampleResult], top_k: int=5) -> List[Tuple[str, float]]:
        """Rank joints by mean error across all samples.

        Args:
            results: evaluation results.
            top_k: number of joints to return.

        Returns:
            List of (joint_name, mean_error) sorted descending.
        """
        ...

    def summary(self, clusters: List[FailureCluster]) -> str:
        """Human-readable summary of failure clusters."""
        ...
