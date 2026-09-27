"""TASK B1 -- the joint (per-individual) fit: one girth field, N clips.

THE DEFECT.  The girth field is a ``(K, 2)`` leaf, ``(g_x, g_z)`` per control
station, and it is fitted PER CLIP against that clip's mask width series.  A
projected body width is measured normal to the projected spine, so it reads the
cross-section's extent in the IMAGE PLANE: on a broadside clip that is the
dorsoventral ``g_z`` alone, and on an oblique one it is a MIX of ``g_z`` and the
lateral ``g_x``.  A per-clip fit has no way to tell the mix apart from a fatter
``g_z`` -- the lateral axis rides the isotropy tie prior -- so it books all of it
to ``g_z``, and the same animal comes back with trunk ``g_z`` 1.084 on its
broadside window and 1.257 on its rolling one (AN15092101; +16.0 %, and
+21.7 % / +13.5 % on the other two shipped pairs).  The lateral axis is then
PRIOR on every animal, which is why volume, cross-section and mass are only
partially delivered.

THE FIX is an estimator, not a fusion rule: fit an individual's clips at once
with the girth field and the betas SHARED and everything else per clip.  Two
clips whose per-frame width Jacobian rows are not parallel make both axes
identifiable; how *well* is a measured property of the pair, and
:func:`girth_jacobians` publishes it with the fit so the morphometrics layer can
report the lateral girth with honest uncertainty instead of a prior dressed as a
measurement.

WHY BETAS ARE SHARED TOO.  Per unit of the largest shape component the mid-trunk
moves 3-4 % laterally, so a per-clip betas leaf would re-absorb exactly the
anisotropy the shared girth is trying to identify.  Nothing else is shared: the
root, the translation, every rail, the wave and the whole per-clip camera / fps /
time base / ego context stay per clip, because they describe a moment, not an
animal.

THE OBJECTIVE is

    L = L_0(include_shared=True) + sum_{c>0} L_c(include_shared=False)

where ``L_c`` is :meth:`SpineSMPLify._loss` verbatim and the three shared terms
(the shape prior, the girth prior and the girth tie) are billed ONCE, inside
clip 0's expression, at the position they already occupy there.  That position
is deliberate: float addition is not associative and gradient accumulation into
a shared leaf follows the graph's topology, so billing them after the clip sum
would differ in the last ulp and Adam would walk away from it over 500 steps.
Billing them to clip 0 makes the N = 1 objective the same statement sequence as
the shipped one -- which is what lets this class replace ``SpineSMPLify.fit``'s
inline driver outright, leaving ONE optimisation driver in the codebase instead
of a single-clip one and a joint one that can drift apart.

Clip weighting is a plain sum: every clip term is already a mean over frames, so
the sum is length-balanced, and summing N likelihoods against one prior is the
honest joint MAP.  It halves the prior's per-clip pull at N = 2, which is the
point of the exercise.  ``clip_weight="mean"`` keeps the per-clip prior strength
for the ablation; the two are identical at N = 1.
"""
from __future__ import annotations
import itertools
import math
from typing import Optional, Sequence
import torch
from torch import Tensor
from .smplify_fitting import ClipContext
DEFAULT_MAX_JOINT_FRAMES = 1200

def _both(a, b):
    """Compose two ``on_step`` hooks; ``None`` when both are off."""
    ...

class JointSpineFit:
    """Drive one Adam over N clips with the girth field and betas shared.

    ``contexts`` are :class:`~shark_pose.encoder.smplify_fitting.ClipContext`
    objects, one per clip, each carrying its OWN
    :class:`~shark_pose.encoder.smplify_fitting.SpineSMPLify` -- fps, focal
    length, principal point and ego track differ inside a single individual.

    At ``N == 1`` this executes the statement sequence ``SpineSMPLify.fit``
    used to execute inline, so a single-clip fit is float-for-float the shipped
    one; ``tests/test_joint_fit.py`` pins that against a golden captured before
    the refactor.
    """

    def __init__(self, contexts: Sequence[ClipContext], *, clip_weight: str='sum', wells: str='auto', max_joint_frames: int=DEFAULT_MAX_JOINT_FRAMES, shared: Sequence[str]=SHARED_LEAVES, tracer=None):
        ...

    def _betas(self, shape_leaf: Tensor) -> Tensor:
        """The betas the shared shape leaf stands for.

        TASK F1.  Unscaled -- the default -- this IS the leaf, so every call
        site reads as the shipped one.  It is deliberately not hoisted into a
        local at those call sites: with ``shape_lr_in_sd`` on the betas are a
        FUNCTION of the leaf, Adam updates the leaf in place and each backward
        frees the product's graph, so the multiply has to be re-made every
        step.
        """
        ...

    def _shared_girth_init(self) -> Optional[Tensor]:
        """The girth START value the shared leaf is cloned from.

        At N = 1 the clip's own value, returned UNCHANGED -- a weighted mean of
        one element is not bitwise that element, and this short circuit is what
        keeps the single-clip fit identical.  Above that, the candidates are
        combined weighted by how many usable width frames each clip brings,
        since a clip with no width observation has nothing to say about girth.
        """
        ...

    def _shared_betas_init(self) -> Tensor:
        """The betas the shared shape prior pulls to; N = 1 returns the clip's."""
        ...

    def joint_loss(self, clips, betas, girth, *, stage=None, silhouette: bool=False, score: bool=False) -> Tensor:
        """``L_0(shared) + sum_{c>0} L_c(no shared)``, in clip order.

        ``silhouette`` draws each clip's next stochastic mask slice; ``score``
        uses each clip's FIXED scoring slice instead, which is how the wells are
        ranked on the same masks.
        """
        ...

    @staticmethod
    def _run(param_groups, steps: int, loss_fn, on_step=None, on_trace=None) -> None:
        """Adam over ``param_groups`` for ``steps`` steps.  The shipped loop.

        TASK D3: a group with no parameters is dropped, and a call with nothing
        left to optimise returns.  Every shipped group is non-empty, so this is
        inert there; it exists because the frozen-pose debug arm
        (``SpineSMPLify.freeze_clip_params``) empties the per-clip groups, and
        Adam refuses an empty parameter list.

        TASK F1: ``on_step``, when given, is called with the 0-based step index
        BEFORE each loss and may change anything the loss reads that is not a
        parameter -- it is what drives the silhouette sharpness anneal.  It is
        ``None`` in every shipped call but the stage-B ones, and those pass
        ``None`` too while the anneal is off, so the loop below is the loop it
        always was.

        ``on_trace`` -- ``--trace-json``, ``None`` unless a tracer was given --
        is called with the step index and the loss AFTER it is computed and
        BEFORE the backward, so what it reads (the loss, the leaves) is the
        state that loss was measured at.  It must not touch a parameter.
        """
        ...

    def _silhouette_anneal_hook(self, stage_b_steps: int):
        """The TASK F1 sharpness schedule, or ``None`` while it is off.

        Every clip's fitter is stepped from the SAME stage-B progress, because
        a joint stage B is one optimiser loop over one objective: two clips
        rasterised at two different blurs would be two different terms summed.
        Returns ``None`` -- and so leaves ``_run`` the loop it always was --
        unless at least one fitter has the anneal on.

        Why it is annealed rather than simply set: the sharp blur is what makes
        the silhouette a roll observation (its minimum moves from 15 deg off
        truth to the truth, and its curvature there from negative to positive
        -- ``outputs/demo_2026-09-03/work/roll/roll_jacobian/``), but a 0.2
        proxy-px blur has a gradient reach of a fraction of a raster pixel, so
        a fit STARTING there is pulled by almost nothing.  The blur begins at
        the shipped 0.75, which reaches, and is tightened onto the answer.
        """
        ...

    def _ref_snapshot(self, clips, shape_leaf) -> dict:
        """Clip 0's MIDDLE frame: enough to re-pose the mesh off a trace.

        ``--trace-json`` only.  The pose is the CLAMPED one -- what the record
        writes and what a viewer would re-pose -- and the betas are in beta
        units whatever the shape leaf's own units are.
        """
        ...

    def _trace_hooks(self, stage: str, clips, shape_leaf, girth, pass_index: int, step_offset: int=0):
        """``(arm, emit)`` for one :meth:`_run`, or ``(None, None)``.

        ``--trace-json``, DEFAULT OFF: with no tracer this returns a pair of
        ``None`` and the optimiser loop is the shipped one.  ``arm`` puts a
        fresh term dict on clip 0's fitter for the steps the tracer wants (the
        shared terms are billed to clip 0, so that is the fitter that sees the
        whole objective); ``emit`` takes it back off and writes the record.
        """
        ...

    def _optimise(self, clips, shape_leaf, girth, pass_index: int=0):
        """Stage A per clip, then ONE joint stage B.  Returns the scores."""
        ...

    @staticmethod
    def _residual(ctx, raw, root, trans, betas, girth) -> float:
        ...

    def _combinations(self, starts) -> list[tuple[int, ...]]:
        """Which (well per clip) combinations the joint schedule runs.

        ``fit()`` runs the whole schedule once per amplitude well and picks on
        the converged objective, so with N clips the full product is W^N.  At
        N <= 2 that is at most 4 joint stage-B runs and is run in full -- at
        N = 1 it is exactly the shipped two runs.  Above that it goes greedy:
        each clip's own winner (``starts`` is already ranked, winner first),
        plus the N single-clip swaps.
        """
        ...

    def run(self) -> list[dict]:
        """Fit every clip.  Returns one result dict per clip, in clip order."""
        ...

    def schedule_block(self) -> dict:
        """What the schedule did, for the record."""
        ...
JACOBIAN_MAX_FRAMES = 120

def _decimate(n: int, cap: int) -> list[int]:
    """``cap`` frame indices SPREAD over ``[0, n)`` -- never a prefix.

    ``step = n // cap`` then ``[:cap]`` truncates: at n = 300, cap = 120 it
    scores frames 0..238, the first 79.7 % of the clip, and at n = 163 it scores
    the first 73.6 %.  The two clips of one individual differ in length by up to
    1.8x, so a prefix rule scores DIFFERENT fractions of each -- and the
    Jacobian rows it averages are exactly the quantity the lateral-axis verdict
    is read off.  Measured on the shipped pairs the prefix cost FAR20110101
    2.7 deg of separation (7.054 -> 4.387), which moves its amplification from
    8.14x to 13.07x, i.e. across the gate.  A spread sample is the same cost.
    """
    ...

def pairwise_separation(mean_rows: Sequence[tuple[str, 'np.ndarray']]) -> dict:
    """Every pair's angle between two clips' mean unit width-Jacobian rows.

    ONE pair is one linear-algebra fact about the girth field: two clips whose
    rows are ``theta`` apart identify both axes at ``1 / sin(theta)`` noise
    amplification.  With N clips there are ``N (N-1) / 2`` such facts, and the
    statistic the gate needs is the BEST of them -- the widest-separated pair is
    what makes ``g_x`` identifiable, and the rest ride along.

    Until 2026-09-13 only ``angle(clip_0, clip_i)`` was computed, so the
    published figure depended on which clip the caller passed first: the E3
    N = 3 record's three real angles are 43.616 / 36.433 / 7.183 deg and it
    published 43.616 or 36.433 according to ``--clip`` order, never seeing the
    7.183 deg broadside-intermediate pair at all.  At N = 2 there is exactly one
    pair, so ``max == min == `` the old value and every N = 2 record ever
    written is unchanged to the digit (spec section 4.3 clause 2, R3).

    Args:
        mean_rows: ``(clip_id, mean row (2,))`` per clip, in clip order.  The
            rows are normalised here, so a per-frame mean of unit rows (which
            is shorter than 1) may be passed straight in.

    Returns:
        ``separation_pairwise_max`` / ``_min`` (deg, or None below two clips),
        ``separation_matrix`` -- the symmetric ``(N, N)`` table with its clip
        order, zero on the diagonal -- and ``separation_pairs``, the
        ``N (N-1) / 2`` upper-triangle entries named by clip id.
    """
    ...

def girth_jacobians(contexts: Sequence[ClipContext], results: Sequence[dict], max_frames: int=JACOBIAN_MAX_FRAMES) -> dict:
    """SIGNED per-clip, per-station width Jacobians, and their conditioning.

    Each clip contributes, per girth control station ``k`` and axis ``a``, the
    derivative of the SUM of its projected trunk-station widths with respect to
    ``g[k, a]``, through THAT clip's own width operator and camera, at the
    fitted pose.  Signed, and per station -- unlike ``fit_video``'s
    ``girth_observed_axis``, which sums ``|grad|`` over frames and so destroys
    exactly the sign a conditioning number needs.

    Why this and not the summed ratio.  A single clip gives one linear
    constraint on the pair ``(g_x, g_z)``: a row ``r_c``.  One clip is rank 1 --
    only ``r_c . g`` is seen and the tie prior supplies the missing direction.
    Two clips are rank 2 IF their rows are not parallel, and the number that
    governs the answer is the ANGLE BETWEEN THE ROWS, not how big either of
    them is.  Measured on the three shipped JOINT fits (AN15092101, FAR20110101,
    PR16100901; ``multiclip/fits/`` and ``multiclip/eval/fits/``) the summed
    ``g_z / g_x`` ratio is 4.63 / 8.19 / 5.08 -- 4.71 / 8.65 / 5.40 once the
    frame sample below is spread rather than a prefix -- all above ``fuse_girth``'s
    ``GIRTH_OBSERVED_RATIO_MAX`` of 2.0.  Summing also makes the pair look
    WORSE than the oblique clip alone (AN station 4: 1.90 alone, 3.19 summed),
    because the broadside frames add pure ``g_z``.  So the conditioning of the
    STACKED unit rows is reported too, in ``identifiability``.

    ``identifiability`` is a DIAGNOSTIC, not a verdict.  Nothing here says the
    lateral axis is measured: the separation angle is an analytic statement
    about the rows, and the one pair with a known anisotropy (the synthetic
    aniso/iso pair) recovers 10 % of it, with ``d g_x`` of the wrong sign, at a
    separation of 4.1 deg.  TASK F1a calibrated the verdict on that pair, and
    ``fuse_girth`` now gates the lateral axis on this separation, on the width
    OPERATOR and on the girth PRIOR BLOCK, and no longer on the summed ratio --
    which is anti-informative in the joint regime (the single rolled view reads
    1.51 and recovers 2 % of the known anisotropy; the pair that recovers 86 %
    reads 3.76).  The ratio is still published, and still gates the per-clip
    path.  See ``GIRTH_JOINT_AMPLIFICATION_MAX``'s calibration table.

    Returns a block ready to drop into ``summary.girth.joint_fit``.
    """
    ...
