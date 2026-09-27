"""
Pose sampling for synthetic shark data generation.

Samples biologically plausible shark poses across 6 locomotion modes,
each with mode-specific joint angle distributions. Uses truncated
Gaussians to enforce per-joint angle limits.

Modes and their probabilities:
    cruise  (40%): sinusoidal body wave, low amplitude
    turn    (20%): asymmetric body curvature with banking
    breach  (10%): extreme dorsal-ventral flex
    strike  (10%): rapid head extension
    rest    (10%): minimal articulation
    escape  (10%): sharp C-start escape response
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def _truncated_gaussian(rng: np.random.Generator, mean: float, std: float, low: float, high: float, max_attempts: int=100) -> float:
    """Sample from a truncated Gaussian via rejection sampling.

    Args:
        rng: numpy random generator instance.
        mean: mean of the Gaussian.
        std: standard deviation.
        low: lower bound for truncation.
        high: upper bound for truncation.
        max_attempts: maximum rejection iterations before clamping.

    Returns:
        Sampled value within [low, high].
    """
    ...

def _sample_joint_angles(rng: np.random.Generator, joint_name: str, lateral_scale: float, dorsoventral_scale: float, axial_scale: float, mean_lateral: float=0.0, mean_dorsoventral: float=0.0, mean_axial: float=0.0) -> np.ndarray:
    """Sample axis-angle rotation for a single joint.

    Samples each of the 3 rotation axes independently from a truncated
    Gaussian centered at the given mean, with standard deviation equal
    to scale * joint_limit.

    Args:
        rng: random generator.
        joint_name: name of the joint (for limit lookup).
        lateral_scale: fraction of max limit used as std for lateral axis.
        dorsoventral_scale: fraction for dorso-ventral axis.
        axial_scale: fraction for axial twist axis.
        mean_lateral: mean for lateral rotation (degrees).
        mean_dorsoventral: mean for dorso-ventral rotation (degrees).
        mean_axial: mean for axial twist rotation (degrees).

    Returns:
        (3,) array of axis-angle components in radians.
    """
    ...

def _apply_sinusoidal_wave(rng: np.random.Generator, joint_angles: np.ndarray, mode_cfg: Dict[str, Any]) -> np.ndarray:
    """Apply a sinusoidal travelling wave along the body chain.

    Models the undulatory locomotion pattern of carangiform/thunniform
    swimming where amplitude increases from head to tail and phase
    propagates posteriorly.

    Args:
        rng: random generator.
        joint_angles: (16, 3) current joint angles in radians.
        mode_cfg: mode configuration dict.

    Returns:
        Modified joint_angles array.
    """
    ...

def _apply_c_start(rng: np.random.Generator, joint_angles: np.ndarray, mode_cfg: Dict[str, Any]) -> np.ndarray:
    """Apply a C-start escape response pattern.

    The shark bends its body into a sharp C-shape by contracting muscles
    on one side. This produces large unilateral lateral curvature.

    Args:
        rng: random generator.
        joint_angles: (16, 3) current joint angles.
        mode_cfg: mode configuration dict.

    Returns:
        Modified joint_angles array.
    """
    ...

def _apply_head_strike(joint_angles: np.ndarray, rng: np.random.Generator, mode_cfg: Dict[str, Any]) -> np.ndarray:
    """Emphasize head joint angles for strike mode.

    Args:
        joint_angles: (16, 3) current joint angles.
        rng: random generator.
        mode_cfg: mode configuration dict.

    Returns:
        Modified joint_angles array.
    """
    ...

def _apply_turn_asymmetry(joint_angles: np.ndarray, rng: np.random.Generator, mode_cfg: Dict[str, Any]) -> np.ndarray:
    """Apply asymmetric curvature and banking for turn mode.

    Args:
        joint_angles: (16, 3) current joint angles.
        rng: random generator.
        mode_cfg: mode configuration dict.

    Returns:
        Modified joint_angles array.
    """
    ...

def sample_mode(rng: Optional[np.random.Generator]=None, mode_weights: Optional[Dict[str, float]]=None) -> str:
    """Sample a locomotion mode from the categorical distribution.

    Args:
        rng: optional numpy random generator. If None, creates a default one.
        mode_weights: optional dict mapping mode_name -> probability.
            If provided, overrides the hardcoded MODE_CONFIG probabilities.
            Must sum to ~1.0 and contain only valid mode names.

    Returns:
        Mode name string.
    """
    ...

def sample_pose(mode: Optional[str]=None, rng: Optional[np.random.Generator]=None, seed: Optional[int]=None, mode_weights: Optional[Dict[str, float]]=None) -> Dict[str, Any]:
    """Sample a complete shark pose for a given locomotion mode.

    This is the main entry point for pose generation. It produces a
    16-joint axis-angle pose vector (48-dimensional) following biologically
    informed distributions specific to the requested locomotion mode.

    Args:
        mode: locomotion mode name. If None, sampled from the categorical prior.
        rng: numpy random generator. If None, one is created from seed.
        seed: random seed (used only if rng is None).
        mode_weights: optional dict mapping mode_name -> probability.
            Only used when mode is None. Overrides MODE_CONFIG probabilities.

    Returns:
        Dictionary with:
            theta: (48,) float64 array of axis-angle rotations (16 joints x 3).
            joint_angles: dict mapping joint_name -> (3,) array in radians.
            mode_name: str, the locomotion mode used.
            joint_angle_limits_deg: dict mapping joint_name -> (lat, dv, ax) limits.
    """
    ...

def sample_pose_batch(n: int, mode: Optional[str]=None, seed: Optional[int]=None, mode_weights: Optional[Dict[str, float]]=None) -> List[Dict[str, Any]]:
    """Sample a batch of poses.

    Args:
        n: number of poses to sample.
        mode: if given, all poses use this mode; otherwise each is sampled independently.
        seed: random seed for reproducibility.
        mode_weights: optional dict mapping mode_name -> probability.
            Only used when mode is None. Overrides MODE_CONFIG probabilities.

    Returns:
        List of pose dictionaries (see sample_pose).
    """
    ...
