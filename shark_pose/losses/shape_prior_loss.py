"""
Shape prior loss: L2 regularisation on PCA shape coefficients (betas).

Penalises deviation from the mean shape to prevent extreme or
implausible body shapes.
"""
import torch
import torch.nn as nn
from torch import Tensor
from shark_pose.core.constants import NUM_BETAS

class ShapePriorLoss(nn.Module):
    """L2 norm penalty on shape parameters (betas).

    Encourages betas to remain close to zero (the mean shape),
    acting as a Gaussian prior N(0, sigma^2 I) on the shape space.

    Args:
        weight: Global scalar weight for this loss term.
    """

    def __init__(self, weight: float=1.0) -> None:
        ...

    def forward(self, betas: Tensor) -> Tensor:
        """Compute L2 shape prior loss.

        Args:
            betas: Shape coefficients, shape (B, NUM_BETAS).

        Returns:
            Scalar loss tensor.
        """
        ...
