"""TASK F1 (shape-observe) -- letting the MASK vote on the shape.

Three changes, each INERT at its default, so a fit that sets none of them is
bit-identical to the one that shipped:

* ``shape_freeze_components="auto"`` -- the freeze list resolved AGAINST THE
  ASSET.  The shipped literal ``[4, 5, 6, 7, 8, 9]`` was calibrated on the v2
  basis, where 4-9 are the generator's isotropic noise floor; on the v3
  anatomical basis those same indices are ``dorsal_position``,
  ``dorsal_height``, ``caudal_span``, ``pectoral_length`` and two inert pads,
  so the literal silently froze FOUR NAMED MODES at zero.
* ``girth_delay_frac`` -- the girth-vs-shape disambiguation rule.  Measured on
  both clips (``outputs/demo_2026-09-03/work/shape_observe/grad/``), the girth
  field's width Jacobian norm is 441.1 (g_x) / 20.5 (g_z) px per unit on the
  demo against 0.030-4.51 for the eight named modes, and 236.1 / 73.7 against
  0.034-3.15 on clip 2 -- 4.6x to 14500x depending on the pair -- so with both
  leaves free from step one the width error is always paid by girth.
* ``width_align_grad`` -- the station alignment carrying gradient into the
  betas, so a mode that slides a station along the body is charged for the
  width it slides onto.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import pytest
import torch
from shark_pose.encoder.smplify_fitting import SHAPE_NOISE_FLOOR, SpineSMPLify
from shark_pose.model_3d import SharkSMPL
from shark_pose.model_3d.shark_smpl import select_visible_landmarks

@pytest.fixture(scope='module')
def model_v2() -> SharkSMPL:
    ...

@pytest.fixture(scope='module')
def model_v3() -> SharkSMPL:
    ...

def _frames(model, fitter, n=3, seed=0):
    ...

def test_auto_freeze_on_a_tableless_basis_is_the_shipped_literal(model_v2):
    """v2 declares no modes, so "auto" is the measured noise floor 4-9."""
    ...

@needs_v3
def test_auto_freeze_on_the_named_basis_frees_every_named_mode(model_v3):
    """The whole point: "auto" must NOT freeze dorsal_*/caudal_*/pectoral_*."""
    ...

@needs_v3
def test_auto_freeze_leaves_the_named_modes_with_gradient(model_v3):
    """A frozen leaf gets a zeroed gradient; a freed one does not."""
    ...

@needs_v3
def test_a_literal_freeze_on_the_named_basis_warns_and_names_the_modes(model_v3):
    """The silent failure this pass found: the shipped literal on v3 assets.

    F1 did not flip the shipped default (SHIP NOTHING), so the documented v3
    entry point -- ``--assets-dir data/model_assets_v3`` with the shipped
    config -- still runs ``[4..9]``.  It must say so at build time, not only in
    ``summary.shape.freeze_resolved`` after the fit.
    """
    ...

@needs_v3
def test_auto_and_an_inert_only_literal_do_not_warn(model_v3):
    ...

def test_a_literal_freeze_on_a_tableless_basis_does_not_warn(model_v2):
    """v1/v2 declare no modes, so the shipped literal is exactly right there."""
    ...

def test_auto_is_the_only_string_accepted(model_v2):
    ...

def test_girth_delay_defaults_to_one_stage_b_optimiser(model_v2, monkeypatch):
    """delay_frac 0 must run stage B as the SINGLE Adam it always was."""
    ...

def test_girth_delay_splits_stage_b_and_pins_the_field(model_v2, monkeypatch):
    ...

def test_a_stage_b_that_is_all_prefix_never_moves_the_girth(model_v2):
    """The rule's limiting case: shape answers, thickness does not move."""
    ...

def test_girth_delay_validates(model_v2):
    ...

def _ramp_align(fitter, model, n=2, seed=1):
    ...

def _aligned(model, n, root, trans, pose, betas, fitter):
    ...

@needs_v3
def test_align_grad_defaults_off_and_gives_the_same_numbers(model_v2, model_v3):
    ...

@needs_v3
def test_align_grad_carries_gradient_into_the_betas(model_v3):
    ...

def test_shape_lr_in_sd_defaults_off_and_needs_sds(model_v2):
    ...

def test_betas_of_is_the_leaf_when_unscaled(model_v2):
    ...

@needs_v3
def test_shape_lr_in_sd_lifts_the_per_mode_reach(model_v3):
    """The measured defect: a fixed raw step reaches less of a wide mode.

    With the leaf in raw units, ``lr * beta_lr_scale * stage_b_steps`` is a
    hard ceiling in RAW units, so a mode whose prior sd is 4.0 raw units can
    only be reached to 0.375 sd.  In sd units the ceiling is the same number of
    SD on every mode.  This asserts the reach the two parametrisations give,
    which is the quantity the config comment quotes.
    """
    ...

@needs_v3
def test_shape_lr_in_sd_actually_moves_the_shape_further(model_v3):
    """End to end: the scaled leaf travels further in sd on the wide modes."""
    ...
