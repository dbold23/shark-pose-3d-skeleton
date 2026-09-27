"""
Focal-aware SPIN: per-sample projection, focal conditioning, and the padded load.

The v3 retrain projected every sample's 2D loss through one fixed focal (1000 px
@256) while its renders used U(110, 230) and its 3D loss supervised the metric
translation those renders were made at.  The two terms then asked for depths
differing by 5.90x on average and the run converged to a compromise (Stage 1
plateaued at j2d 53.7 / j3d 36.2 against v2's 24.8 / 4.0).

The fix has three halves and each is checked here:

  * the 2D loss projects each sample through the focal it was RENDERED at, so
    the two supervision terms agree by construction;
  * the regressor is TOLD that focal, so the depth it regresses can be metric
    instead of expressed in one projection focal's units;
  * a pre-focal checkpoint loads into the new architecture by zero-padding the
    single new input column, which reproduces it exactly at focal 1000 -- the
    property that makes the v4 run an experiment about the focal and not about
    a re-initialised network.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.core.camera import PerspectiveCamera
from shark_pose.data.synthetic_dataset import DEFAULT_FOCAL_PX, SyntheticDataset
from shark_pose.encoder.iterative_regressor import FOCAL_REFERENCE_PX, IterativeRegressor, focal_to_feature
from shark_pose.encoder.spin_model import NEAR_PLANE_M, SPINModel
from shark_pose.model_3d.shark_smpl import SharkSMPL

@pytest.fixture
def tiny_model():
    """A placeholder SharkSMPL: this file tests projection and conditioning."""
    ...
SAME_FUNCTION_RTOL = 1e-05

def assert_same_function(got: dict, want: dict, keys: tuple[str, ...]) -> None:
    """Two SPIN outputs come from the same function, bit-identically if it can."""
    ...

def _real_proxies(n: int=4) -> torch.Tensor:
    """The first n real proxies from data/raw_v2, as a (n, 19, 256, 256) batch."""
    ...

@pytest.mark.skipif(not V2_CHECKPOINT.exists(), reason=f'needs {V2_CHECKPOINT}')
@pytest.mark.skipif(not (V1_ASSETS / 'bone_positions.npy').exists(), reason=f'needs {V1_ASSETS}')
def test_padded_v2_weights_reproduce_the_v2_model():
    """The shipped v2 Stage 1 checkpoint, padded, is the same network at 1000 px.

    This is what licenses `init_from` in the v4 config: the run starts from the
    v2 model itself, not from the v2 model plus a randomly-initialised input.
    The focal feature of the reference focal is log(1) = 0, and the padded
    column is zero, so the extra term in the matmul is exactly 0.0.

    The WEIGHTS are checked bit for bit, which no BLAS can perturb; the OUTPUTS
    are checked through ``assert_same_function``, which takes bit-identity where
    the gemm gives it and bounds the summation-order noise where it does not.
    """
    ...

def test_padded_load_reproduces_outputs_for_a_freshly_built_pair(tiny_model):
    """The same property without the 360 MB checkpoint, so CI always checks it."""
    ...

def test_padded_load_refuses_an_unrelated_shape(tiny_model):
    ...

def test_projection_uses_one_focal_per_sample(tiny_model):
    """`SPINModel.project` equals a PerspectiveCamera built for each sample."""
    ...

def test_projection_without_a_focal_is_the_model_default(tiny_model):
    ...

@pytest.fixture(scope='module')
def v3_set(tmp_path_factory):
    """A small v3 set: focal sampled per sample, 2D and 3D GT stored."""
    ...

@pytest.mark.skipif(not (V1_ASSETS / 'bone_positions.npy').exists(), reason=f'needs {V1_ASSETS}')
def test_v3_batch_gt_reprojects_through_its_own_focals(v3_set, tiny_model):
    """The check that failed on the v3 run.

    Project the batch's GT 3D joints through the batch's OWN per-sample focals
    and you get the batch's GT 2D joints back.  That is the whole content of
    "the 2D and 3D terms agree": the translation the 3D loss supervises is the
    one the 2D loss's projection implies.  Through a single focal of 1000 the
    same joints land hundreds of pixels away, which is what the run was
    optimising against.
    """
    ...

def test_a_v2_style_set_reports_the_default_focal(tmp_path, tiny_model):
    """A set that records no focal at all reads as the pipeline pinhole."""
    ...

def test_real_and_mixed_rows_carry_the_pipeline_focal():
    """Real rows carry the pipeline default absent a per-clip calibration.

    That label is a placeholder, not a measurement -- see the note in
    ``RealDataset.__getitem__`` and ``test_real_rows_take_a_calibrated_focal``.
    """
    ...

def test_focal_feature_is_log_relative_to_the_reference():
    ...

def test_the_regressor_output_moves_with_the_focal():
    """Same features, different focal -> different parameters, with gradient.

    A zero-initialised focal column would make this pass trivially at init, so
    the column is given a value first: what is being checked is that the input
    is wired to the output at all, not that an untrained net has an opinion.
    """
    ...

def test_an_unconditioned_regressor_refuses_a_focal():
    ...

def test_spin_forward_rejects_a_focal_of_the_wrong_length(tiny_model):
    ...

def test_focal_agreement_guard_is_skipped_for_a_conditioned_model(caplog):
    """The guard exists for a model with ONE projection focal; this has none."""
    ...

def test_checkpoint_meta_stamps_a_conditioned_model_with_no_train_focal(tmp_path):
    """`train_focal: null` is the point: there is no number to divide by."""
    ...

def test_projection_clamps_depth_at_the_near_plane(tiny_model):
    """A joint at or behind the camera must not project to ~1e10 px.

    ``PerspectiveCamera.project`` clamps z at 1e-8, so an unguarded divide
    turns one overshot depth into a ~1e10 px residual that eats the whole
    clipped gradient of its batch.
    """
    ...

def test_projection_in_front_of_the_near_plane_is_untouched(tiny_model):
    """The guard is a no-op for every physical sample, to the last bit."""
    ...

@pytest.mark.parametrize('bad', [0.0, -1.0, -204.29])
def test_focal_to_feature_rejects_a_non_positive_focal(bad):
    """log(0) is -inf and log(<0) is nan; both would reach every parameter."""
    ...

@pytest.mark.parametrize('bad', [0.0, -1.0])
def test_projection_rejects_a_non_positive_focal(tiny_model, bad):
    ...

def test_row_focal_prefers_the_annotation_then_the_set_then_the_default():
    ...

@pytest.mark.skipif(not (RAW_V2 / 'annotations').exists(), reason='data/raw_v2 not present')
def test_real_rows_take_a_calibrated_focal():
    """A real set shot on a known lens can be labelled with its own focal.

    Labelling real rows 1000 px is a placeholder: letterboxing does not change
    a lens, and the demo clip's proxy focal is 204.29 px @256.  The default is
    unchanged, so no shipped recipe moves.
    """
    ...

def test_near_plane_clamp_is_counted_and_reset():
    """A joint at or behind the near plane is clamped AND counted, then reset.

    The clamp keeps the 2D loss finite; the counter is what makes a collapsing
    depth head visible in the training log (review of the v4 retrain).
    """
    ...

def test_projection_in_eval_mode_has_no_side_effects():
    """eval() projection clamps but counts nothing: no state on a forward pass.

    Only the training loops call pop_clamp_stats(); counting in eval would let
    the counters grow for the life of a fit_video / harness process.
    """
    ...

def test_trainer_warns_on_clamp_and_spike(caplog):
    ...
