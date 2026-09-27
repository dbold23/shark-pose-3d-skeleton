"""``scripts/eval_detector_marginal.py --real-focal-px``: the focal-aware v2 row.

Two things have to be true at once and both are checked here.

1. With no flag the harness is BIT-IDENTICAL to what it was: the regressor is
   called with no focal argument at all, the v2 warm start's depth is untouched,
   and the row keeps its old name.  Every published harness number was measured
   that way.
2. With a flag the v2 row does what ``scripts/fit_video.py --camera`` does at
   that focal -- the same ``fit_video.warm_start_focal_plan`` decides it -- and
   the v1 rows (control / detector / spin / spin+smplify) do NOT move, because a
   conditioned warm start is taken from a SECOND forward rather than by
   re-conditioning the one the v1 rows are scored from.

The inference stages need no checkpoint here: SPIN, the proxy builder and the
v2 fitter are all stood in for, so what is under test is the wiring.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import eval_detector_marginal as edm
import fit_video
GOPRO_PROXY_FOCAL = 204.288

def test_no_flag_means_no_plan_at_all():
    ...

def test_an_unconditioned_checkpoint_rescales_its_warm_start_depth():
    ...

def test_a_conditioned_checkpoint_takes_the_focal_and_keeps_its_metric_depth():
    ...

@pytest.mark.parametrize('conditioned', [False, True])
def test_the_plan_is_fit_videos_plan(conditioned):
    """Not a re-implementation: the harness must return fit_video's own answer."""
    ...

def test_the_default_row_keeps_its_published_name():
    ...

def test_a_focal_aware_row_is_tagged_so_both_can_sit_in_one_report():
    ...

class _RecordingSpin:
    """Stands in for SPINModel: records how it was called, answers per focal."""

    def __init__(self, n_joints: int=16):
        ...

    def __call__(self, batch, **kwargs):
        ...

def _proxies(n: int, res: int=8):
    ...

def test_spin_on_proxies_passes_no_focal_argument_by_default():
    """The default path must not even mention `focal` -- that is the bit-identity."""
    ...

def test_spin_on_proxies_conditions_every_sample_on_the_given_focal():
    ...

class _RecordingV2Fitter:

    def __init__(self):
        ...

    def fit(self, root_init, trans_init, kp, conf):
        ...

def _v2_ctx(fitter, spin_focal=None, warm_z_scale=1.0):
    ...

@pytest.fixture()
def stub_pipeline(monkeypatch):
    """Replace the proxy builder and the v1->v2 retarget with identities."""
    ...

def _run(spin, v2_ctx, n=3):
    ...

def test_default_path_runs_one_forward_and_leaves_the_depth_alone(stub_pipeline):
    ...

def test_an_unconditioned_checkpoint_scales_the_v2_depth_and_nothing_else(stub_pipeline):
    """No second forward (there is no focal input); the depth carries the focal."""
    ...

def test_a_conditioned_checkpoint_warm_starts_from_a_second_forward(stub_pipeline):
    ...

def test_real_focal_px_defaults_to_none(monkeypatch):
    ...

def test_real_focal_px_is_parsed_as_a_float(monkeypatch):
    ...
