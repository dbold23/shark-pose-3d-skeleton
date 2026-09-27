"""
Adaptive sampling strategy for active learning.

Converts failure cluster difficulty scores into sampling weights for
the next round of synthetic data generation. Uses temperature-controlled
softmax to balance exploration (uniform) vs exploitation (hard cases).
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from .failure_analyzer import FailureCluster

class AdaptiveSampler:
    """Converts difficulty scores to sampling weights.

    Uses softmax over per-mode difficulty scores with a temperature
    parameter that controls how aggressively we oversample hard cases:
        - High temperature (3.0+): near-uniform sampling (exploration)
        - Low temperature (1.0):   heavily biased towards failures (exploitation)

    A minimum weight floor prevents any mode from dropping to zero,
    avoiding catastrophic forgetting.

    Args:
        temperature: softmax temperature. Higher = more uniform.
        min_weight: minimum weight per mode (floor).
        base_weights: starting mode weights. Defaults to synthetic.yaml values.
    """

    def __init__(self, temperature: float=2.0, min_weight: float=0.05, base_weights: Optional[Dict[str, float]]=None):
        ...

    def compute_mode_weights(self, mode_difficulty: Dict[str, float]) -> Dict[str, float]:
        """Compute new mode sampling weights from difficulty scores.

        Args:
            mode_difficulty: dict mapping mode_name -> difficulty score (0-1).
                Modes not present get difficulty 0.

        Returns:
            Dict mapping mode_name -> new probability (sums to 1.0).
        """
        ...

    def compute_weights_from_clusters(self, clusters: List[FailureCluster]) -> Dict[str, float]:
        """Compute mode weights from full cluster list.

        Aggregates cluster difficulty by pose mode (averaging across
        camera bins) before computing softmax weights.

        Args:
            clusters: failure clusters from FailureAnalyzer.analyze().

        Returns:
            Dict mapping mode_name -> new probability.
        """
        ...

    def write_weights_file(self, weights: Dict[str, float], output_path: str, round_idx: int=0) -> None:
        """Write weights JSON for generate_synthetic_data.py --weights-file.

        Args:
            weights: mode weights dict.
            output_path: path to write JSON.
            round_idx: active learning round number.
        """
        ...

    @staticmethod
    def load_weights_file(path: str) -> Dict:
        """Load a weights JSON file.

        Returns:
            Full dict with 'mode_weights', 'round', 'temperature'.
        """
        ...

    def summary(self, old_weights: Dict[str, float], new_weights: Dict[str, float]) -> str:
        """Human-readable comparison of old vs new weights."""
        ...
