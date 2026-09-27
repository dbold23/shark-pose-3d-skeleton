"""
Mesh I/O utilities: load/save OBJ, PLY, and manage template mesh assets.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch import Tensor

def load_obj(path: str | Path) -> tuple[Tensor, Tensor]:
    """Load an OBJ file, returning vertices and faces.

    Args:
        path: path to .obj file

    Returns:
        vertices: (N, 3) float tensor
        faces: (F, 3) long tensor (0-indexed)
    """
    ...

def save_obj(path: str | Path, vertices: Tensor | np.ndarray, faces: Tensor | np.ndarray) -> None:
    """Save mesh as OBJ file.

    Args:
        path: output path
        vertices: (N, 3)
        faces: (F, 3) 0-indexed
    """
    ...

def load_ply(path: str | Path) -> tuple[Tensor, Tensor]:
    """Load a PLY file (ASCII format), returning vertices and faces.

    Args:
        path: path to .ply file

    Returns:
        vertices: (N, 3) float tensor
        faces: (F, 3) long tensor (0-indexed)
    """
    ...

def compute_vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """Compute per-vertex normals by averaging face normals.

    Args:
        vertices: (N, 3)
        faces: (F, 3)

    Returns:
        normals: (N, 3) unit normals
    """
    ...

def compute_edge_lengths(vertices: Tensor, faces: Tensor) -> Tensor:
    """Compute edge lengths for all edges in the mesh.

    Args:
        vertices: (N, 3)
        faces: (F, 3)

    Returns:
        edge_lengths: (num_edges,) unique edge lengths
    """
    ...
