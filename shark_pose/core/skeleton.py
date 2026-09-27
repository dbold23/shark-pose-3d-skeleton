"""
Shark skeleton definition: 16-joint kinematic tree with parent-child relationships.

Canonical keypoint sequence and morphometric pairs imported from SharkScarAnnotator.
Root = body_midpoint_dorsal (analogous to human pelvis in SMPL).
"""
from typing import Dict, List, Optional, Tuple
import torch

def get_parent_indices() -> List[int]:
    """Return parent index for each joint (-1 for root).

    Ordered by SHARK_KEYPOINT_SEQUENCE.
    """
    ...

def get_kinematic_chain(joint_name: str) -> List[str]:
    """Return chain from root to the given joint (inclusive)."""
    ...

def get_children(joint_name: str) -> List[str]:
    """Return direct children of a joint in the kinematic tree."""
    ...

def build_adjacency_matrix() -> torch.Tensor:
    """Build symmetric adjacency matrix (16x16) for structure-aware loss.

    A[i,j] = 1 if joints i and j are connected by a bone in the kinematic tree.
    """
    ...

def get_bone_pairs() -> List[Tuple[int, int]]:
    """Return list of (parent_idx, child_idx) bone pairs."""
    ...
