"""
GAN-style training loop for adversarial refinement.

Alternates between generator (encoder) and discriminator updates
using WGAN-GP training with gradient penalty.
"""
import torch
import torch.nn as nn
from torch import Tensor
from .discriminator import SharkMeshDiscriminator

class AdversarialTrainer:
    """Manages alternating generator/discriminator training.

    Supports WGAN-GP (gradient penalty) for stable training.
    """

    def __init__(self, discriminator: SharkMeshDiscriminator, disc_lr: float=0.0001, gen_lr_multiplier: float=0.1, gp_weight: float=10.0, n_critic: int=5):
        """
        Args:
            discriminator: the mesh discriminator
            disc_lr: discriminator learning rate
            gen_lr_multiplier: generator LR = disc_lr * this
            gp_weight: gradient penalty weight (lambda)
            n_critic: number of discriminator updates per generator update
        """
        ...

    def gradient_penalty(self, real_rotmats: Tensor, fake_rotmats: Tensor, real_betas: Tensor, fake_betas: Tensor) -> Tensor:
        """Compute WGAN-GP gradient penalty.

        Interpolates between real and fake samples and penalizes
        gradient norm deviating from 1.
        """
        ...

    def discriminator_step(self, real_rotmats: Tensor, real_betas: Tensor, fake_rotmats: Tensor, fake_betas: Tensor) -> dict:
        """Single discriminator update step.

        Returns:
            dict with disc_loss, gp_loss, real_score, fake_score
        """
        ...

    def generator_loss(self, fake_rotmats: Tensor, fake_betas: Tensor) -> Tensor:
        """Compute generator (encoder) adversarial loss.

        The generator wants to maximize the discriminator's score on its output.

        Returns:
            gen_loss: scalar tensor (for backprop through encoder)
        """
        ...

    def should_update_discriminator(self) -> bool:
        """Check if discriminator should be updated this step."""
        ...

    def should_update_generator(self) -> bool:
        """Check if generator should be updated this step."""
        ...
