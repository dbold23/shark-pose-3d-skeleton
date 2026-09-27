"""
Gaussian Mixture Model pose prior for shark body poses.

Learns a distribution over the 48-dimensional pose parameter space
(16 joints x 3 axis-angle) from sampled or observed poses. Provides:

    - ``fit(poses)``: fit the GMM to a collection of poses.
    - ``log_prob(theta)``: evaluate log-probability for use as a pose prior loss.
    - ``sample(n)``: generate new plausible poses from the learned distribution.
    - ``save`` / ``load``: persist the fitted model.

The negative log-probability serves as a regularisation loss during
model fitting (analogous to the VPoser prior in human pose estimation).
"""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from torch import Tensor

class PosePriorGMM:
    """Gaussian Mixture Model prior over shark pose parameters.

    Wraps scikit-learn's GaussianMixture (or falls back to a scipy-based
    implementation) to model the distribution of valid shark poses in
    the 48-dimensional axis-angle space.

    Attributes:
        num_components: number of Gaussian mixture components.
        covariance_type: covariance parameterisation ('full', 'diag', 'tied', 'spherical').
        is_fitted: whether the model has been fit to data.
    """

    def __init__(self, num_components: int=DEFAULT_NUM_COMPONENTS, covariance_type: str=DEFAULT_COVARIANCE_TYPE, regularization: float=DEFAULT_REGULARIZATION, random_state: int=42):
        """Initialise the pose prior GMM.

        Args:
            num_components: number of mixture components.
            covariance_type: type of covariance matrix.
            regularization: regularisation added to covariance diagonal.
            random_state: random seed for reproducibility.
        """
        ...

    def _init_gmm(self) -> None:
        """Initialise the underlying GMM estimator."""
        ...

    def fit(self, poses: Union[np.ndarray, Tensor]) -> 'PosePriorGMM':
        """Fit the GMM to a collection of pose vectors.

        Args:
            poses: (N, 48) array or tensor of axis-angle pose parameters.
                Each row is a full 16-joint pose.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: if poses have wrong dimensionality.
        """
        ...

    def _fit_scipy(self, poses: np.ndarray) -> None:
        """Fit GMM using scipy (K-means init + EM-like approximation).

        This is a simplified fallback; for production use sklearn.
        """
        ...

    def log_prob(self, theta: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        """Compute log-probability of pose(s) under the fitted GMM.

        This value can be negated and used as a pose prior loss term:
            loss_prior = -pose_prior.log_prob(theta)

        Args:
            theta: (48,) or (N, 48) pose parameter(s). Accepts both
                numpy arrays and torch tensors.

        Returns:
            Log-probability with the same type as input.
            Shape (N,) for batched input, scalar for single input.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        ...

    def _log_prob_scipy(self, X: np.ndarray) -> np.ndarray:
        """Compute log-probability using scipy fallback parameters."""
        ...

    def sample(self, n: int=1, return_tensor: bool=True, device: Optional[torch.device]=None) -> Union[np.ndarray, Tensor]:
        """Sample new poses from the fitted GMM.

        Args:
            n: number of poses to sample.
            return_tensor: if True, return a torch.Tensor.
            device: torch device for the output tensor.

        Returns:
            (n, 48) poses sampled from the prior distribution.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        ...

    def _sample_scipy(self, n: int) -> np.ndarray:
        """Sample from the GMM using scipy fallback."""
        ...

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted GMM to a pickle file.

        Args:
            path: output file path.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        ...

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PosePriorGMM':
        """Load a fitted GMM from a pickle file.

        Args:
            path: path to the saved pickle file.

        Returns:
            Fitted PosePriorGMM instance.
        """
        ...

    def __repr__(self) -> str:
        ...
