"""Pure-numpy glTF 2.0 / GLB reader.

No trimesh / pygltflib / pytorch3d. Buffers are memory-mapped, so a 175 MB GLB
can be opened and its *skin* accessors read without ever materialising the mesh.

What is supported
-----------------
* ``.gltf`` (external ``.bin``, ``data:`` URIs) and ``.glb`` (binary chunks).
* Accessors: every component type, ``byteStride``, ``normalized`` integers, and
  sparse accessors.
* Node hierarchy: local ``matrix`` or TRS, world matrices by walking parents.
* Skins: joint node list, parents restricted to the skin, inverse bind matrices.
* Animations: sampler ``(times, values)`` with ``LINEAR`` / ``STEP`` /
  ``CUBICSPLINE`` interpolation and a :func:`resample` helper (quaternion
  outputs are slerped, everything else is lerped).

Rotation helpers are numpy (glTF quaternions are ``xyzw``); the project's
``shark_pose.core.transforms`` is the torch equivalent and is deliberately not
imported here so that asset extraction stays torch-free.
"""
from __future__ import annotations
import base64
import json
import mmap
import struct
from pathlib import Path
from typing import Any
import numpy as np
__all__ = ['GLTF', 'AnimationSampler', 'quaternion_to_matrix', 'matrix_to_quaternion', 'quaternion_slerp', 'matrix_to_axis_angle', 'axis_angle_to_matrix', 'trs_to_matrix', 'resample']
_GLB_MAGIC = 1179937895
_CHUNK_JSON = 1313821514
_CHUNK_BIN = 5130562

def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """``(..., 4)`` xyzw quaternion -> ``(..., 3, 3)`` rotation matrix."""
    ...

def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrix -> ``(..., 4)`` xyzw quaternion (w >= 0)."""
    ...

def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Shortest-arc slerp. ``q0``/``q1`` ``(..., 4)``, ``t`` broadcastable ``(...,)``."""
    ...

def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` -> ``(..., 3)`` axis-angle (rotation vector)."""
    ...

def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """``(..., 3)`` rotation vector -> ``(..., 3, 3)`` via Rodrigues."""
    ...

def trs_to_matrix(translation: np.ndarray | None, rotation: np.ndarray | None, scale: np.ndarray | None) -> np.ndarray:
    """Compose a glTF node matrix ``T * R * S`` (4x4, column-vector convention)."""
    ...

def resample(times: np.ndarray, values: np.ndarray, query: np.ndarray, interpolation: str='LINEAR', is_quaternion: bool | None=None) -> np.ndarray:
    """Evaluate a keyframe track at ``query`` times.

    ``times`` ``(K,)`` ascending, ``values`` ``(K, C)`` (``(3K, C)`` for
    CUBICSPLINE: in-tangent / value / out-tangent triples). A 1-D ``values``
    (a SCALAR track) is accepted and returns a ``(N,)`` result. Queries outside
    the key range are clamped, per the glTF spec.
    """
    ...

class AnimationSampler:
    """One animation channel: ``(times, values)`` plus :meth:`at`."""

    def __init__(self, node: int, path: str, times: np.ndarray, values: np.ndarray, interpolation: str='LINEAR') -> None:
        ...

    @property
    def duration(self) -> float:
        ...

    def at(self, t: np.ndarray) -> np.ndarray:
        """Sample the channel at scalar or array times ``t``."""
        ...

    def __repr__(self) -> str:
        ...

class GLTF:
    """A parsed glTF/GLB document with lazily decoded accessors."""

    def __init__(self, doc: dict[str, Any], buffers: list[Any], base_dir: Path | None=None):
        ...

    @classmethod
    def load(cls, path: str | Path) -> 'GLTF':
        """Open a ``.gltf`` or ``.glb``; external buffers are memory-mapped."""
        ...

    def _read_glb(self, path: Path):
        ...

    def _resolve_buffer(self, buf: dict[str, Any], bin_chunk):
        ...

    def close(self) -> None:
        """Release memory maps (accessor arrays taken before this are copies)."""
        ...

    def __enter__(self) -> 'GLTF':
        ...

    def __exit__(self, *exc) -> None:
        ...

    def buffer_view_bytes(self, index: int) -> memoryview:
        ...

    def accessor(self, index: int) -> np.ndarray:
        """Decode accessor ``index`` to ``(count, ncomp)`` (``(count,)`` for SCALAR)."""
        ...

    def node_local_matrix(self, index: int) -> np.ndarray:
        ...

    def node_parents(self) -> np.ndarray:
        """``(num_nodes,)`` parent node index, ``-1`` for roots."""
        ...

    def node_world_matrices(self) -> np.ndarray:
        """``(num_nodes, 4, 4)`` world matrices (roots use their local matrix)."""
        ...

    def node_name(self, index: int) -> str:
        ...

    def skin(self, index: int=0) -> dict[str, Any]:
        """Joint node ids, names, in-skin parents (-1 for roots), inverse bind matrices."""
        ...

    def animation_names(self) -> list[str]:
        ...

    def animation(self, index: int | str) -> list[AnimationSampler]:
        """All channels of an animation, by index or by name."""
        ...

    def animation_duration(self, index: int | str) -> float:
        ...

    def mesh_primitive_attribute(self, mesh: int, attribute: str, primitive: int=0) -> np.ndarray:
        ...

    def mesh_primitive_indices(self, mesh: int, primitive: int=0) -> np.ndarray | None:
        ...

    def mesh_node(self, mesh: int) -> int | None:
        """First node that instantiates ``mesh`` (``None`` if unused)."""
        ...
