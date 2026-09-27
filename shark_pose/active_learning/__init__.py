"""
Active learning loop for shark pose estimation.

V1: Weighted sampling from failure clusters. Evaluates model performance,
clusters failures by generation metadata (pose mode, camera angles),
computes difficulty scores, and reweights synthetic data generation
to oversample hard cases.

Components:
    FailureAnalyzer: Clusters evaluation failures and computes difficulty scores.
    AdaptiveSampler: Converts difficulty scores to sampling weights.
"""
from .failure_analyzer import FailureAnalyzer, FailureCluster, SampleResult
from .sampling_strategy import AdaptiveSampler
__all__ = ['FailureAnalyzer', 'FailureCluster', 'SampleResult', 'AdaptiveSampler']
