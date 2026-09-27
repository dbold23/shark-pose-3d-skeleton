"""Midline spine skeleton for the v2 shark rig (31 joints).

The v1 rig in :mod:`shark_pose.core.skeleton` uses the 16 detector keypoints as
bones.  Those keypoints are *skin landmarks* (eye centre, gill slit, fin tips),
several of which sit off the midline, so the v1 kinematic tree is not an
anatomical skeleton and posing it contorts the mesh.

This module defines the replacement: a 13-segment midline vertebral chain plus
9 fin root/tip pairs.  The spine segment names and their order are copied
verbatim from the user's rigged sevengill scan
(``sevengill_rigged.glb``) so its four behaviour animations (cruise, turn,
escape, rest) retarget onto this rig one segment at a time.

Frame convention (same as the v1 template): ``x`` lateral with left negative,
``y`` longitudinal with the snout at +y, ``z`` vertical.

The numeric rig (joint positions, skinning weights) lives in
``data/model_assets_v2/`` and is produced by ``scripts/build_spine_rig.py``;
this module only owns the *topology* and the helpers for reading it back.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

def build_parents(root_name: str=ROOT_JOINT_NAME) -> list[int]:
    """Parent index per joint (-1 for the root).

    The vertebral chain is walkable both ways from ``root_name``: stations
    ahead of the root are parented *backwards* toward it (cranium -> branchial_1
    -> ... -> root), stations behind it forwards (root -> ... -> caudal_axis_2).
    This mirrors how the DigitalLife armature splits its two ``spine.*`` chains
    at ``main``.  Fin roots are attached by ``FIN_PARENT_STATION`` (filled in by
    the build script and stored in ``skeleton.json``); this function uses the
    defaults below and the build script overrides them with measured values.
    """
    ...

def bone_frames(bone_positions: np.ndarray, parents: list[int] | None=None, joint_names: list[str] | tuple[str, ...] | None=None) -> np.ndarray:
    """Per-joint orthonormal basis ``(J, 3, 3)``, columns (along-bone, lateral, vertical).

    Mirrors ``scripts/extract_rig_assets.py::spine_frames`` so that angles
    measured in the sevengill's joint-local bone frame can be re-expressed in
    THIS rig's model frame: with ``B = bone_frames(...)[j]``, a local axis-angle
    ``v`` becomes ``B @ v`` and a local rotation ``R`` becomes ``B R B.T``.

    ``along-bone`` always points tail-ward (station ``k`` -> station ``k+1``),
    independently of which way the v2 chain is parented, because that is the
    direction the sevengill's frames were built along.  Non-spine joints inherit
    the basis of their nearest vertebral ancestor, again as in the extractor.
    """
    ...

def chain_signs(parents: list[int] | None=None) -> np.ndarray:
    """``(J,)`` sign that turns a head-to-tail midline curvature into a local angle.

    The curvature at station ``k`` is the turning angle between the segments
    ``k-1 -> k`` and ``k -> k+1``, measured walking snout to tail.  A joint whose
    parent lies BEHIND it (the stations ahead of the root, parented backwards)
    is traversed the other way round, so its local rotation is the negated
    curvature.  The root itself gets 0: both of its segments are rigid in its own
    frame, so its curvature cannot be expressed at all and callers must move it
    onto the neighbouring stations.
    """
    ...

def children_of(parents: list[int]) -> list[list[int]]:
    """Child index lists for a parent array."""
    ...

def descendants_of(joint: int, parents: list[int] | None=None) -> list[int]:
    """Every joint in the subtree rooted at ``joint`` (excluding ``joint``)."""
    ...

def bone_pairs(parents: list[int] | None=None) -> list[tuple[int, int]]:
    """(parent, child) index pairs, one per bone."""
    ...

def is_spine(joint: int) -> bool:
    ...

def fin_group_of(joint: int) -> str | None:
    """Name of the fin group a joint belongs to, or ``None`` for the spine."""
    ...

@dataclass(frozen=True)
class SpineRig:
    """The numeric v2 rig loaded from ``data/model_assets_v2``."""
    joint_names: list[str]
    parents: list[int]
    root: int
    bone_positions: np.ndarray
    lbs_weights: np.ndarray
    template_vertices: np.ndarray
    faces: np.ndarray
    meta: dict

    @property
    def num_joints(self) -> int:
        ...

    def index(self, name: str) -> int:
        ...

def load_spine_rig(asset_dir: str | Path | None=None) -> SpineRig:
    """Read ``skeleton.json`` + the ``.npy`` arrays written by the build script."""
    ...
