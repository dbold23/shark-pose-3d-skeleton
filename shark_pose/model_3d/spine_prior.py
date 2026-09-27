"""Low-dimensional swimming prior ("rails") for the 31-joint v2 spine rig.

A real shark spine is a midline chain that mostly stays straight and bends as a
travelling wave whose amplitude grows towards the tail.  The sevengill scan
ships four behaviour animations (cruise / turn / escape / rest) that describe
*how* it bends; ``scripts/build_motion_prior.py`` fits them into gain tables and
this module turns a handful of interpretable scalars into per-joint rotations.

Parameters (``SpinePoseParams``, 25 numbers per frame)

===============  ======  ========================================================
amp              (B,)    cruise amplitude scale, 1.0 = the animation.  The
                         animation is exaggerated; the prior is a broad mixture
                         (swimming ~0.55, gliding ~0.05).
phase            (B,)    wave phase at t = 0, radians
freq             (B,)    tail-beat frequency, Hz (one value per clip; free,
                         with a log-normal prior centred at 0.45 Hz)
turn             (B,)    turning curvature kappa, radians of peak per-segment
                         yaw offset; Laplace prior, mostly 0
escape           (B,)    C-start scale, 1.0 = the animation's peak bend
pitch            (B,)    dorsoventral bend scale on the trunk profile, radians
pectoral         (B,2,2) [side, (pitch, dihedral)] radians
dorsal           (B,)    dorsal sway, radians
caudal           (B,)    caudal lobe yaw offset, radians
residual         (B,13)  per-segment free yaw, radians, strong L2 prior
===============  ======  ========================================================

Frames
------
The gain tables and the joint limits live in the joint-local bone frame of
``scripts/extract_rig_assets.py`` (x = along-bone, y = lateral, z = vertical);
the three angles compose as ``Rz(yaw) Ry(pitch) Rx(roll)``, so yaw is a lateral
bend, pitch is dorsoventral and roll spins about the bone.  Skinning, however,
consumes rotations in the MODEL frame (x lateral, y longitudinal with the snout
at +y, z vertical), so :meth:`SpinePrior.to_axis_angle` maps every joint through
its rest basis ``B`` (``basis`` in the asset, columns = the model-frame
directions of the local axes) before returning: ``aa_model = B @ aa_local``,
equivalently ``R_model = B R_local B^T``.  Without that change of basis "pitch"
would rotate about the body's long axis, i.e. it would be a roll.

Clamping and :meth:`limit_violation` happen BEFORE the basis change, in the
local (yaw, pitch, roll) frame the limits were measured in; ``B`` is a rotation,
so distances in axis-angle space are the same on either side of it.

The v2 chain is rooted mid-body (``spine_05_trunk_03``) and the stations ahead
of the root are parented backwards, so a joint's local rotation there is the
NEGATED midline curvature (``chain_sign`` in the asset).  The root's own
curvature is not expressible -- both of its segments are rigid in its own frame
-- and ``scripts/build_motion_prior.py`` splits it evenly onto the two
neighbouring stations when it writes the tables.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
YAW, PITCH, ROLL = (0, 1, 2)
SPINE_PRIOR_VERSION = 2
DEFAULT_AMP = 0.35
AMP_PRIOR_MODE = 0.55
AMP_PRIOR_LOG_SIGMA = 0.7
AMP_GLIDE_WEIGHT = 0.25
AMP_GLIDE_MODE = 0.05
AMP_GLIDE_LOG_SIGMA = 0.5
FREQ_PRIOR_HZ = 0.45
FREQ_PRIOR_LOG_SIGMA = 2.0
TURN_LAPLACE_SCALE = 0.05
ESCAPE_LAPLACE_SCALE = 0.25

def canonical_key(name: str) -> str | None:
    """Reduce a rig bone name to a canonical v2 slot key.

    Tolerates the usual spelling drift (``spine_04_trunk_02`` / ``trunk02`` /
    ``trunk_2``, ``dorsal_1_fin_root`` / ``first_dorsal_root`` / ``dorsal1_root``).
    Returns ``None`` if the name is not part of the v2 rig.
    """
    ...

def resolve_joint_order(joint_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """Permutation ``p`` with ``p[slot] = index into joint_names``.

    Raises ``ValueError`` if the given rig does not cover the 31 v2 slots.
    """
    ...

def ypr_to_axis_angle(ypr: Tensor) -> Tensor:
    """``(..., 3)`` (yaw, pitch, roll) -> ``(..., 3)`` axis-angle.

    Composes ``Rz(yaw) Ry(pitch) Rx(roll)`` through quaternions, which stays
    differentiable at the identity (the common case for this prior).
    """
    ...

@dataclass
class SpinePoseParams:
    """Swimming parameters.  Scalars are ``(B,)`` or ``(B, T)`` for a clip."""
    amp: Tensor
    phase: Tensor
    freq: Tensor
    turn: Tensor
    escape: Tensor
    pitch: Tensor
    pectoral: Tensor
    dorsal: Tensor
    caudal: Tensor
    residual: Tensor

    @property
    def batch_size(self) -> int:
        ...

    @classmethod
    def default(cls, batch_size: int=1, prior: 'SpinePrior | None'=None, device: torch.device | str | None=None, dtype: torch.dtype=torch.float32) -> 'SpinePoseParams':
        """Cruise defaults: amplitude 0.35, no turn/escape/pitch/fins/residual."""
        ...

    def requires_grad_(self, flag: bool=True) -> 'SpinePoseParams':
        ...

    def to(self, *args, **kwargs) -> 'SpinePoseParams':
        ...

    def to_vector(self) -> Tensor:
        """Pack into ``(B, 25)`` (per-frame axes are flattened alongside)."""
        ...

    @classmethod
    def from_vector(cls, vec: Tensor) -> 'SpinePoseParams':
        """Inverse of :meth:`to_vector`."""
        ...

class SpinePrior(nn.Module):
    """Maps :class:`SpinePoseParams` to per-joint axis-angle on the v2 rig."""

    def __init__(self, tables: dict[str, np.ndarray | Tensor], joint_names: list[str] | tuple[str, ...] | None=None):
        ...

    @classmethod
    def load(cls, path: str | Path=DEFAULT_ASSET, joint_names: list[str] | tuple[str, ...] | None=None) -> 'SpinePrior':
        """Load ``motion_prior.npz`` (optionally reordered onto a rig's names)."""
        ...

    def _angles(self, params: SpinePoseParams, t: Tensor) -> Tensor:
        """Per-joint (yaw, pitch, roll) in radians.

        ``t`` broadcasts against the parameter scalars; the returned shape is
        ``(*broadcast, 31, 3)``.
        """
        ...

    def to_model_frame(self, aa_local: Tensor) -> Tensor:
        """``(..., 31, 3)`` axis-angle: joint-local bone frame -> model frame."""
        ...

    def to_local_frame(self, aa_model: Tensor) -> Tensor:
        """Inverse of :meth:`to_model_frame` (the basis is a rotation)."""
        ...

    def to_axis_angle(self, params: SpinePoseParams, t: Tensor | float=0.0, clamp: bool=True) -> Tensor:
        """``(B, 31, 3)`` model-frame axis-angle at ``t`` (scalar or ``(B,)`` s)."""
        ...

    def to_axis_angle_clip(self, params: SpinePoseParams, times: Tensor, clamp: bool=True) -> Tensor:
        """``(B, T, 31, 3)`` model-frame axis-angle over a clip.

        The phase advances as ``2 pi freq t``, so one clip shares
        ``(amp, freq, phase0)``.  ``turn`` (and every other scalar) may be given
        per frame as ``(B, T)`` for a smooth manoeuvre.
        """
        ...

    def clamp_to_limits(self, angles: Tensor) -> Tensor:
        """Soft-free hard clamp of ``(..., 31, 3)`` yaw/pitch/roll to the limits."""
        ...

    def limit_violation(self, angles: Tensor) -> Tensor:
        """Hinge penalty on ``(..., 31, 3)`` angles outside the limits."""
        ...

    def limits_deg(self) -> dict[str, list[float]]:
        ...

    def configure_priors(self, **overrides) -> 'SpinePrior':
        """Override the amplitude / frequency prior shape in place.

        ``None`` values are ignored, so a config block can pass every key and
        leave the unset ones at the module defaults.
        """
        ...

    def amplitude_nll(self, amp: Tensor) -> Tensor:
        """Negative log density of the cruise amplitude, unshifted.

        A mixture of two log-normals -- a broad swimming mode at
        ``amp_mode`` and a narrow gliding mode at ``glide_mode`` -- evaluated as
        a log-sum-exp so the fit can sit in either well without a barrier
        between them.  ``glide_weight = 0`` collapses it to a single log-normal.
        """
        ...

    def frequency_nll(self, freq: Tensor) -> Tensor:
        """Negative log density of the tail-beat frequency, unshifted."""
        ...

    def prior_loss(self, params: SpinePoseParams, reduce: bool=True) -> Tensor:
        """Negative log prior, SHIFTED so the cruise defaults score exactly 0.

        A mixture log-normal on ``amp`` (swimming mode 0.55, gliding mode 0.05),
        a log-normal on ``freq`` (0.45 Hz), Laplace on ``turn`` / ``escape``, L2
        on ``pitch``, the fin scalars and the per-segment residual yaw.

        The amplitude and frequency terms are offset by their value at the
        cruise defaults (``DEFAULT_AMP``, ``self.freq_hz``).  The offsets are
        constants -- they move no minimum -- but they keep the "zero at the
        defaults" contract, so the loss may dip slightly below zero between the
        default and the prior's true mode.
        """
        ...

    def cruise_cycle(self) -> Tensor:
        """``(T, 31, 3)`` axis-angle poses over one exported cruise period."""
        ...

    def spine_midline(self, angles_or_aa: Tensor, lengths: Tensor | None=None) -> Tensor:
        """Top-view midline ``(..., 14, 2)`` from per-joint lateral angle ``(..., 13)``.

        The input is the model-frame lateral rotation of the 13 spine joints in
        SLOT order (snout first).  ``chain_sign`` turns each one back into the
        snout-to-tail midline curvature before the headings are accumulated, so
        the polyline is a real midline whichever way the chain is parented.
        """
        ...

    def summary(self) -> dict:
        ...

def _base_ndim(field: str) -> int:
    """Rank of a parameter field when it is per-frame constant (batch only)."""
    ...

def load_prior_report(path: str | Path | None=None) -> dict:
    """Read the human-readable ``motion_prior.json`` next to the asset."""
    ...
