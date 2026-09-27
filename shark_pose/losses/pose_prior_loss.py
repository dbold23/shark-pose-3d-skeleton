"""
Pose prior loss: negative log-likelihood under a Gaussian Mixture Model (GMM).

The GMM is fitted offline on plausible shark poses (from synthetic data or
biomechanical constraints) and acts as a regulariser to prevent anatomically
implausible pose predictions.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

class PosePriorLoss(nn.Module):
    """Negative log-likelihood of pose parameters under a fitted GMM.

    The GMM is parameterised by means, covariances (precision-weighted),
    and mixture weights.  These can be loaded from a pre-fitted checkpoint
    or provided at construction time.

    The loss is computed as::

        NLL = -log sum_k pi_k * N(pose | mu_k, Sigma_k)

    For numerical stability we work in log-space via logsumexp.

    Args:
        weight: Global scalar weight for this loss term.
        num_components: Number of GMM mixture components.
        pose_dim: Dimensionality of the pose vector.
    """

    def __init__(self, weight: float=1.0, num_components: int=8, pose_dim: int=48) -> None:
        ...

    def load_gmm(self, means: Tensor, precisions: Tensor, weights: Tensor) -> None:
        """Load pre-fitted GMM parameters.

        Args:
            means: Component means, shape (K, D).
            precisions: Precision matrices, shape (K, D, D).
            weights: Mixture weights, shape (K,). Must sum to 1.
        """
        ...

    def _log_component_density(self, pose: Tensor) -> Tensor:
        """Compute log N(pose | mu_k, Sigma_k) for all components.

        Args:
            pose: (B, D)

        Returns:
            (B, K) log-density per component.
        """
        ...

    def forward(self, pose: Tensor) -> Tensor:
        """Compute negative log-likelihood under GMM prior.

        Args:
            pose: Pose parameters, shape (B, D) where D = num_joints * 3 = 48.

        Returns:
            Scalar loss tensor.
        """
        ...
