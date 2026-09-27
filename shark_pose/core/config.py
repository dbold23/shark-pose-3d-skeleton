"""
Configuration management using OmegaConf with Hydra-style composition.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from omegaconf import DictConfig, OmegaConf

def load_config(config_path: str | Path) -> DictConfig:
    """Load a single YAML config file."""
    ...

def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configs (later configs override earlier ones)."""
    ...

def load_experiment_config(experiment_name: str='default', overrides: dict[str, Any] | None=None) -> DictConfig:
    """Load a full experiment config with Hydra-style composition.

    Loads the experiment YAML which references sub-configs for model, data, train.
    Then applies any CLI overrides.

    Args:
        experiment_name: name of experiment config (without .yaml)
        overrides: dict of dot-path overrides, e.g. {"train.lr": 1e-4}

    Returns:
        Fully composed config.
    """
    ...

def load_model_config(model_name: str) -> DictConfig:
    """Shortcut to load a model config from configs/model/."""
    ...

def load_train_config(stage_name: str) -> DictConfig:
    """Shortcut to load a training stage config from configs/train/."""
    ...

def load_data_config(data_name: str) -> DictConfig:
    """Shortcut to load a data config from configs/data/."""
    ...

def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save config to YAML file."""
    ...

def config_to_dict(cfg: DictConfig) -> dict:
    """Convert OmegaConf config to plain dict."""
    ...
