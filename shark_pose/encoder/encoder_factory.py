"""
Factory function for creating encoder instances from config.

This is the single point where encoder selection happens. No base class,
no registry, no plugin system — just a function with an if/elif chain.
"""
from omegaconf import DictConfig
import torch.nn as nn

def create_encoder(encoder_cfg: DictConfig) -> nn.Module:
    """Create an encoder instance from an experiment config.

    Args:
        encoder_cfg: OmegaConf dict with at least a ``type`` key.
            Supported types: "proxy", "dinov2_frozen", "dinov2_lora", "hybrid".

    Returns:
        An nn.Module with:
            - forward(input) -> (B, output_dim) feature tensor
            - output_dim: int property
    """
    ...
