"""Synthetic known-truth dorsal fins for testing the trailing-edge matcher (S8 R3).

An individual is a falcate fin outline in canonical coordinates (anterior base
(0,0), posterior base (1,0), tip up) whose trailing edge carries 0-4 Gaussian
notches: the identity. A sighting renders that outline with the body below the
base line into a binary mask after a random view: rotation, scale, mirror
(left- or right-swimming), yaw foreshortening along the body axis, mild shear,
and noisy fin keypoints. Nothing about a sighting is visible to the matcher
except the mask and the three noisy keypoints, as on real frames.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class FinIdentity:
    tip: tuple[float, float]
    sweep: float
    notches: list[tuple[float, float, float]]

def random_identity(rng: np.random.Generator, n_notches: int | None=None) -> FinIdentity:
    ...

def outline(ident: FinIdentity, n: int=400) -> np.ndarray:
    """Closed canonical outline: anterior base -> leading edge -> tip -> trailing edge -> posterior base."""
    ...

@dataclass
class Sighting:
    mask: np.ndarray
    tip: np.ndarray
    base_front: np.ndarray
    base_back: np.ndarray

def render(ident: FinIdentity, rng: np.random.Generator, *, base_px: float | None=None, yaw_max_deg: float=35.0, kp_noise_frac: float=0.015, size: int=512) -> Sighting:
    ...
