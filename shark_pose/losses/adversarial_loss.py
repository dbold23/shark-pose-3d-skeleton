"""
Adversarial losses for discriminator-based training.

Supports two GAN formulations:
    - **WGAN-GP** (Wasserstein GAN with gradient penalty): uses the
      Wasserstein distance as the critic loss with gradient penalty
      for Lipschitz constraint enforcement.
    - **Hinge loss**: a margin-based formulation that saturates for
      well-classified samples, providing stable gradients.

Gradient penalty is provided as a standalone function for flexibility.
"""
from typing import Literal, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import grad as torch_grad

class AdversarialLoss(nn.Module):
    """GAN loss supporting WGAN-GP and hinge formulations.

    Args:
        weight: Global scalar weight for this loss term.
        mode: 'wgan-gp' or 'hinge'.
    """

    def __init__(self, weight: float=1.0, mode: Literal['wgan-gp', 'hinge']='wgan-gp') -> None:
        ...

    def forward(self, disc_real: Tensor, disc_fake: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute generator and discriminator losses.

        Args:
            disc_real: Discriminator output for real samples, shape (B,) or (B, 1).
            disc_fake: Discriminator output for fake/generated samples, same shape.

        Returns:
            Tuple of (generator_loss, discriminator_loss), both scalar tensors.
        """
        ...

def gradient_penalty(discriminator: nn.Module, real: Tensor, fake: Tensor, lambda_gp: float=10.0) -> Tensor:
    """Compute gradient penalty for WGAN-GP.

    Interpolates between real and fake samples and penalises the
    discriminator's gradient norm deviating from 1.

    Args:
        discriminator: The discriminator network.
        real: Real samples, shape (B, ...).
        fake: Fake/generated samples, same shape as real.
        lambda_gp: Gradient penalty coefficient.

    Returns:
        Scalar gradient penalty tensor.
    """
    ...
