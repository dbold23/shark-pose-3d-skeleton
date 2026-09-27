"""Per-clip and per-individual aggregation, with the uncertainty model.

The rule this module exists to enforce: **a CI needs a distribution, a
sensitivity band needs only a perturbation, and calling the second one the first
is the failure mode.**

Which is which is decided by the fit, not by taste.  The rig's bones are rigid
and the shape is shared across a clip, so every model-space length has exactly
zero per-frame spread -- quoting a frame MAD on one is a category error.  The
only genuine per-frame distribution lives in the projected landmarks, so:

* **(P) projected** quantities get a real CI: median + MAD over gated frames,
  with a percentile bootstrap over frames.  ``ci_kind = "spread"``.
* **(M) model-space** quantities get a sensitivity band composed
  multiplicatively from three named perturbations -- the girth envelope, +/-1
  PCA sd of shape, and the clip-level depth-jitter scale bias.
  ``ci_kind = "sensitivity"``.
* ``ci_kind = "posterior"`` is in the vocabulary and is never emitted, because
  nothing in this pipeline produces a posterior over shape.

Absolutes are absent, not zero.  A clip whose scale block is unmet reports
``state = "unmet"``, ``value = None`` and the reason; a consumer that reads that
null as zero has a bug.

Units of the bands: ``ci_low`` / ``ci_high`` are in the row's ``unit`` and carry
girth x betas x depth only.  The metric-scale band is built by
``world_frame.scale_resolve`` (D1): ``sigma_cm`` -- May et al.'s observer error,
a LENGTH IN CENTIMETRES, not a fraction -- divided by the individual's own
resolved label and taken at the declared ``ci_sigma_multiple``, which is why the
same sigma is 20.5 % on a 2.44 m animal and 8.6 % on a 5.79 m one (8.6-20.5 %
across this corpus).  ``scale.label_resolution`` carries the named scalars.
That band enters ``value_m`` linearly on a length, squared on an area and cubed
on a volume, and it is folded into ``value_m_ci_low`` / ``value_m_ci_high``.
Those two are the only interval that brackets ``value_m``; joining
``(value_m, ci_low, ci_high)`` is a unit error.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
from ..sim2real import world_frame as wf
from ..sim2real.proxy_geometry import proxy_to_image_xy
from .extract import girth_field_from_record, BODY_DENSITY_KG_M3, GENERIC_CLIP_STEMS, ClipExtraction, DEPTH_JITTER_FRAC, GIRTH_BAND_G_X, GIRTH_BAND_G_Z, GIRTH_FIELD_CORPUS_REACH, GIRTH_OBSERVED_LEVEL_ERROR_FRAC, GIRTH_OBSERVED_LEVEL_ERROR_WORDS, MorphAssets, extract_frame, girth_field_from_record, rebuild_girth_band
from .schema import CSV_COLUMNS, EXPECTED_RANGES, MorphometricRow, PER_STATION_NEVER_OBSERVED, QUANTITIES, TRUNK_ONLY_UNMEASURED, TWOD_COLUMNS, quality_grade, range_verdict_applies, resolve_observed_or_prior
BROADSIDE_MAX_DEG = 20.0
RESIDUAL_MAD_MAX = 3.0
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260907
MAD_TO_SIGMA = 1.4826

def median_mad(values: np.ndarray) -> tuple[float, float]:
    """Median and median absolute deviation, NaNs dropped."""
    ...

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median -- order-invariant by construction (it sorts)."""
    ...

def bootstrap_median_ci(values: np.ndarray, alpha: float=0.05, n_boot: int=N_BOOTSTRAP) -> tuple[float, float]:
    """Percentile bootstrap CI of the median over frames (deterministic seed)."""
    ...

def boxcar_smooth(series: np.ndarray, span_frames: int) -> np.ndarray:
    """Zero-phase boxcar, truncated SYMMETRICALLY at the ends.

    The records' own convention (``attitude_quality``): a 0.5 s window, no
    padding and no extrapolation, so a sample near an end averages a shorter,
    still-centred window.  Reused rather than replaced so the smoothed series
    here and in the record mean the same thing.
    """
    ...

def smoothing_span_frames(fps: float, seconds: float=0.5) -> int:
    """Odd frame count spanning ``seconds`` at ``fps`` (the records' 0.5 s)."""
    ...

@dataclass
class GateResult:
    keep: np.ndarray
    broadside: np.ndarray
    obliqueness_deg: np.ndarray
    rejections: dict[str, int]
    n_frames: int
    n_gated: int
    params: dict[str, Any]

def proxy_joints_to_native(camera: dict, xy_proxy: np.ndarray) -> np.ndarray:
    """Map proxy-canvas (x, y) back to native image pixels.

    ``joints_2d_proxy`` lives on the SQUARE ``proxy_resolution`` canvas that
    ``sim2real.proxy_geometry.letterbox_image`` produced: a uniform scale plus a
    centred pad.  Multiplying by ``width / proxy_resolution`` recovers the scale
    and drops the pad ``(W - H) / 2``, which on a 1920x1080 frame pushes every
    y down by 420 px -- 38.9 % of the frame height.  The gate then reads a
    centred animal as sitting on the bottom border, and discards the frame.

    This routes through the one shared inverse, so the gate, the fitter and the
    centreline extractor cannot disagree about where a proxy joint is.

    Returns the input unchanged when the camera block does not carry a width,
    height and proxy_resolution (a caller that has no letterbox to undo).
    """
    ...

def gate_frames(extraction: ClipExtraction, broadside_max_deg: float=BROADSIDE_MAX_DEG) -> GateResult:
    """The four gates of the D1 spec, with the count each one rejects.

    1. **Detection** -- the record's own ``detected`` flag.
    2. **Residual** -- reject ``residual_px`` above
       ``median + 3 * 1.4826 * MAD`` of the clip, i.e. three ROBUST sigma
       (``RESIDUAL_MAD_MAX * MAD_TO_SIGMA``, the rule ``GateResult.params``
       states), which is 48% wider than three raw MADs.
    3. **Truncation / edge contact** -- the record's own centreline gate
       parameters (``bend_series.gate.edge_end_margin_frac``), applied to the
       two ENDS of the projected spine chain, which is what the record's
       ``edge_end`` rejection means: a body whose snout or tail station reaches
       into the frame margin is truncated and its lengths are short.  This is a
       stricter analogue than the record's own, because the record gates the SAM
       mask centreline on a stride-2 subset while this gates the fitted chain on
       every frame; both counts are reported, neither is assumed equal.
    4. **Obliqueness** -- the angle between the body's longitudinal axis and the
       image plane.  This is the true ``is_perpendicular``; the 2D tool has to
       proxy it from keypoint spread.  Reported as an ANGLE, and used as a
       weight, not only as a boolean: a broadside gate and a girth gate want
       opposite frames.
    """
    ...
INSUFFICIENT_SUPPORT_FRAC = 0.1
THIN_SUPPORT_FRAC = 0.5
OK_SUPPORT_FRAMES = 30
REFUSE_ROWS_ON_INSUFFICIENT_SUPPORT = False

def support_state(frames_measured: int, frac_measured: float) -> str:
    """``ok`` / ``thin`` / ``insufficient`` -- :data:`MEASUREMENT_SUPPORT_STATE_RULE`."""
    ...

def measurement_support(clips: Sequence[tuple[str, GateResult]]) -> dict:
    """The first-class ``frames measured of frames fitted`` block (S3 B).

    One clip's pair for a clip record; every clip of the animal for an
    individual record, in which case the totals are the SUM over clips and the
    per-clip breakdown travels with them -- a two-clip individual carrying
    290/300 and 5/300 must not be readable as one 295/600.
    """
    ...

def apply_support_contract(rows: Sequence[MorphometricRow], support: dict, enabled: bool | None=None) -> None:
    """S3 B rule 1: an ``insufficient`` record publishes reasons, not numbers.

    In place, and a no-op unless the record's state is ``insufficient`` AND the
    flag is on (:data:`REFUSE_ROWS_ON_INSUFFICIENT_SUPPORT`, default OFF).
    """
    ...

def scale_propagate(value: float | None, metres_per_model_unit: float | None, power: int) -> float | None:
    """Model units -> metres.  Linear on a length, CUBED on a volume."""
    ...

def compose_band(factors: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """Multiply independent multiplicative bands."""
    ...

def sensitivity_factors(name: str, extraction: ClipExtraction) -> dict[str, tuple[float, float]]:
    """The named perturbations behind a model-space band, per quantity.

    * ``girth`` -- measured by re-evaluating the quantity at both ends of the
      girth envelope.  Per axis, and stated in the row's own
      ``girth_band_basis``: an axis that rode the PRIOR is banded at the D1 5c
      ``honest_band`` (g_x in ``GIRTH_BAND_G_X``, g_z in ``GIRTH_BAND_G_Z``) --
      how far the field can move without changing what a broadside camera sees
      -- while an axis the joint fit OBSERVED is banded at its own fitted field
      +- ``GIRTH_OBSERVED_LEVEL_ERROR_FRAC``, the delivering configuration's
      level error at synthetic truth (spec 5.3, ``resolve_girth_observation``).
      Neither is a scatter and neither is a posterior.
    * ``betas`` -- rms fractional change over +/-1 PCA sd on each of the 10
      components, measured on this clip's own pose.
    * ``bins`` -- the station integral's cell count, the construction's one free
      constant, re-evaluated at half and double ``PROFILE_BINS``.  Since D4
      settled the cross-section on an exact plane slice it is a QUADRATURE
      constant: it moves no envelope, and the term is worth a fraction of a
      percent on the demo's volume -- against a factor of 1.64 for the
      superseded vertex-binned envelope, and against `section_shape` beside it,
      which is the construction's largest term by an order of magnitude.  It is kept on every girth, area, volume and mass row -- a term
      that is small should be shown to be small -- and absent from lengths and
      their ratios, which do not move with it.  The full sweep travels in
      ``provenance.profile_bins_sweep``.
    * ``section_shape`` -- the AREA rule inside the exact plane slice.  The
      slice is exact; the area is that of the contour points' CONVEX HULL,
      which bounds the area they enclose, and the band's low end is the
      shoelace over the same points sorted by angle, which chords the caudal
      keel and the gaps the fin gate leaves and reads low.  One-sided DOWNWARD
      by construction, worth -11.5% on the rest template's volume, i.e. an
      order of magnitude more than ``bins``.  The superseded ``pi * a * b``
      envelope is NOT an end of this band: it sits above the hull at 86% of the
      template's stations, so it is not a candidate value.  Perimeters do not
      move with the rule and carry no term.
    * ``depth`` -- the clip-level scale bias from root-depth jitter, +/-1.8% on
      a model-unit length, raised to the quantity's dimension.  Ratios do not
      inherit it: numerator and denominator move together.
    """
    ...

def girth_band_stamp(extraction: ClipExtraction) -> dict[str, list[float] | str]:
    """What the ``girth`` term of this clip's band was built from (spec R2).

    Empty on the prior path, so a band that did not move carries no new key and
    the shipped corners stay corner-for-corner what they were.  On the observed
    path it states the floor and the words that must travel with it: the number
    is a level error measured at synthetic truth, not a posterior and not the
    fit's marginal (which is ~0.4 % and is pixel noise).
    """
    ...

def truncation_block(extractions: Sequence[ClipExtraction]) -> dict:
    """Which window LANE each clip was admitted in (spec 3.4, R10).

    The record carries the test, not this layer: ``summary.truncation`` with a
    boolean ``truncated`` (and, optionally, a ``lane``, a ``reason`` and the
    measurement behind it).  A record that carries none is reported
    ``untested`` -- not ``whole``.  The distinction is the point: no mask-based
    truncation test has ever been run over this corpus, and a reader must be
    able to see that rather than infer a clean lane from silence.

    On a clip whose test says TRUNK_ONLY the ends of the animal are outside the
    frame, so ``TRUNK_ONLY_UNMEASURED`` is emitted ``state: "unmeasured"`` with
    a reason and no number.  Girth, cross-section, pose and betas are still
    reported: the trunk is what a close-up sees, and it is the reason these
    windows are worth having.
    """
    ...

@dataclass
class ClipResult:
    individual: str
    clip: str
    rows: list[MorphometricRow]
    gates: GateResult
    scale: dict
    provenance: dict
    keep: np.ndarray | None = None
    broadside: np.ndarray | None = None

    def row(self, name: str) -> MorphometricRow:
        ...

    def as_dict(self) -> dict:
        ...

def label_rows_of(raw: dict) -> list[dict]:
    """The length-label rows a fit record carries, for :func:`wf.scale_resolve`.

    A record written before the label ladder landed carries only the parsed
    filename token, so the rows are rebuilt from it -- which is what makes the
    band RETROACTIVE: an existing record is re-banded by re-aggregating it, with
    no refit.  A record written after it carries the resolution itself and that
    is used verbatim.
    """
    ...

def label_ladder_knobs(resolutions, who: str='this individual') -> dict:
    """The tie-break and the band a set of label resolutions DECLARE (boxfix).

    ``wf.scale_resolve`` takes two optional knobs, ``prefer_source`` and
    ``sigma_cm_override``, and a block resolved WITH them must be re-resolved
    WITH them: rebuilding the same rows without the preference silently returns
    the group median -- on two disagreeing rows, a number neither source said --
    at the May class band instead of the one the clip published.  Every rebuild
    path (the clip's own, the individual's, the fusion) reads them from here.

    Raises when the blocks disagree: one animal has one length and one band.
    """
    ...

def _scale_headline(resolved: dict) -> dict:
    """The four fields a published absolute must be greppable by (test P5).

    They also live in ``label_resolution``; they are hoisted because a reader
    checking "does this record state whose estimate the scale is?" must not
    have to know the nested key, and because ``scale.observer_class`` is the
    name the spec's own acceptance test greps for.
    """
    ...

def clip_label_resolution(raw: dict, ci_sigma_multiple: float | None=None) -> dict:
    """One clip's own label resolution: the record's, or rebuilt from its rows.

    ``ci_sigma_multiple`` is the COVERAGE the band is quoted at, and it is a
    reporting choice over the same rows -- so a record that already carries a
    resolution at another coverage is re-resolved from its own ``rows_in``
    rather than returned as it stands.  Without it the CLI flag would be
    silently overridden by whatever the records were written with, and
    ``_fuse_scale`` (which reads the coverage off the clips) would rebuild the
    individual's band at that value.
    """
    ...

def resolve_individual_scale(extractions: Sequence[ClipExtraction], ci_sigma_multiple: float | None=None) -> dict:
    """The INDIVIDUAL's length label, resolved once, over every clip's rows.

    The two-pass shape ``resolve_girth_observation`` already has, for the same
    reason: **whether a label is contested is a statement about the individual's
    sources SEEN TOGETHER** (spec 4.6 clause 4), so a clip resolving its own
    single filename token can never see the disagreement and always resolves.
    Before this, ``_fuse_scale`` refused the individual while every
    ``clip_<individual>_<clip>.json`` beside it in the SAME output directory
    published a metric scale and every ``value_m`` -- two mutually contradictory
    scales, x6.16 apart in volume on the APT22062207 geometry (14.0 vs 9.5 ft),
    shipped together.

    Hand the result to :func:`aggregate_clip`; a clip whose individual is
    contested then refuses with the individual's reason.
    """
    ...

def _scale_block(extraction: ClipExtraction, individual_resolution: dict | None=None, ci_sigma_multiple: float | None=None) -> dict:
    ...

def _stamp_tl_convention(scale: dict, rows: Sequence['MorphometricRow']) -> None:
    """Say WHICH total length the label is assumed to measure, and by how much
    the other one would move the answer.

    Three lengths live in one record: the fitted mesh's snout-to-caudal-tip
    extent (the scale DENOMINATOR), the posed landmark chord, and the spine arc
    plus rostrum cap and caudal lobe.  ``arc/chord`` is 1.070-1.078 on every
    record on disk -- x1.23 on volume -- so the convention is not a detail, and
    leaving it implicit is a one-sided +23 % volume risk corpus-wide.
    """
    ...

def _fit_stamps(record) -> dict:
    """What the FIT was: its bars, its commit and its config digest (R6, P5).

    Test P5 -- "0 records carry a ``value_m`` without ``bars_stamp``,
    ``git_sha``, ``config_digest``" -- is scored on the MORPHOMETRICS record,
    which is where a ``value_m`` lives, while the stamps are written by
    ``scripts/fit_individual.py`` into the fit record's joint block.  This is
    the one hop between them.  Every field is ``None`` on a record written
    before 2026-09-13, and saying so is the point: an absolute read off a
    record whose gate bars are unknown cannot be checked, because the same
    20.1 deg of separation is ``observed`` under the bars of one commit and
    ``prior`` under the next.
    """
    ...

def aggregate_clip(extraction: ClipExtraction, broadside_max_deg: float=BROADSIDE_MAX_DEG, density_kg_m3: float=BODY_DENSITY_KG_M3, girth: dict | None=None, scale_resolution: dict | None=None, ci_sigma_multiple: float | None=None, refuse_on_insufficient_support: bool | None=None) -> ClipResult:
    """One clip -> one row per quantity in the vocabulary.

    ``girth`` is the INDIVIDUAL's fused girth block (``fuse_girth``, normally
    via ``resolve_girth_observation``).  It decides ``observed_or_prior`` on
    every girth-field row and therefore the grade -- the verdict belongs to the
    clips seen together, so a clip aggregated on its own cannot produce it and,
    with ``None``, every such row reports ``prior`` exactly as it shipped.

    ``scale_resolution`` is the INDIVIDUAL's label resolution
    (``resolve_individual_scale``), for the same reason: whether the label is
    CONTESTED is a statement about the individual's sources seen together, and a
    clip resolving its own filename token can never see the disagreement.  With
    ``None`` the clip resolves alone, which is right for a one-clip call and
    wrong for a clip of a contested individual.

    ``ci_sigma_multiple`` is the coverage the metric-scale band is quoted at
    (1 sigma by default, and what every record on disk was banded at).  It is
    stamped in ``scale.coverage_reported`` and must be the same on every clip of
    one individual -- ``_fuse_scale`` refuses a disagreement rather than
    averaging two coverages.
    """
    ...

def _girth_band_provenance(extraction: ClipExtraction) -> dict:
    """``girth_band_g_x`` / ``_g_z`` / ``_scope`` -- of the band ACTUALLY USED.

    On the prior path these are the D1 5c ``honest_band`` corners and the
    sentence that names them, byte-for-byte what shipped.  Once A2 rebuilds an
    axis at its own fitted field x (1 -+ 8 %), the honest_band corners are no
    longer what that axis was banded at, and publishing them under the older,
    more familiar key name three keys from ``girth_band_basis`` (which says
    otherwise) is a false statement in the more legible place.
    """
    ...

def _build_row(name, spec, extraction, gates, tl_med, mpu, scale_unc, scale, density, girth=None, truncation=None) -> MorphometricRow:
    ...

def _absolute_row(name, spec, extraction, gates, mpu, scale_unc, scale, density, common, label=None) -> MorphometricRow:
    """volume_m3 / mass_kg -- present with a cubed band, or absent with a reason."""
    ...

def _clip_weight(result: ClipResult) -> float:
    ...

def fuse_betas(results: Sequence[ClipResult], extractions: Sequence[ClipExtraction]):
    """The individual's shape: one tie when the fit tied it, else a FALLBACK.

    The correct estimator for a multi-clip individual is a joint refit with the
    betas tied across clips (one shape, per-clip cameras and poses).  Where the
    records did NOT come from such a fit, this layer cannot produce one, so it
    reports a precision-weighted mean with ``w = n_gated /
    median_native_residual^2`` and says ``FALLBACK``.

    TASK D4 / spec R13.  Where they DID -- every clip carries the same
    ``girth.joint_fit`` id and that fit lists ``betas`` among its shared
    tensors -- the correct estimator was already applied, at fit time, and the
    per-clip betas are copies of one tensor (measured spread 0 to 1.7e-18 on
    the records on disk).  Calling that a FALLBACK and printing the estimator
    it used beside it told a reader the shape was fused after the fact when it
    was tied by construction.  The weighting is still reported, and is
    immaterial: a weighted mean of identical vectors is that vector.

    The tie is CHECKED, not assumed: a spread above ``SHARED_BETAS_TOL`` means
    the records disagree about a tensor they claim to share, and the status
    falls back with the measured spread named.
    """
    ...

def _clip_video(record) -> tuple[str, str]:
    """``(video, how it was identified)`` for the clip.

    ``summary.scale.source_name`` is the only field that names the SOURCE video:
    every window in this corpus is cut to a file called ``window.mp4``, so
    ``raw["video"]`` identifies the window, not the video it came from, and two
    clips of one animal would look like one file. When the source video was
    never recorded the clip falls back to its own id -- which makes clause 6
    fall back with it, recorded as such and never passed off as a pass test.
    """
    ...

def _creation_time_s(record) -> float | None:
    """The container ``creation_time`` of the clip's video, in seconds.

    Written into ``summary.time`` by the metadata sweep (ledger ``videos.csv``).
    Absent on every record fitted before that sweep, which is what the clause's
    stated fallback exists for.
    """
    ...

def _clip_span_s(record) -> tuple[float | None, float | None]:
    """(start, end) of the window within its own video, in seconds."""
    ...

def _different_pass(a: dict, b: dict) -> tuple:
    """``(different_pass, which test)`` for one pair of clips.

    ``(None, "unavailable")`` when NEITHER route -- a ledger ``pass_id`` on both
    clips, or a container ``creation_time`` on both -- is there.  That is not
    "same pass": a test that could not be run has no verdict, and the caller
    publishes the absence as ``different_video_only`` rather than as a pass
    test.

    ``(None, "creation_time_conventions_differ")`` when both clips carry a
    timestamp but the two clocks are stamped on DIFFERENT conventions.
    ``world_frame.clip_meta_pass`` refuses a ``creation_time_utc`` that carries
    no convention precisely because "a timestamp whose convention is unknown
    cannot be differenced against another video's" -- and differencing two
    KNOWN but unlike conventions is the same error with the refusal skipped: a
    start-stamped and an end-stamped video of ONE pass differ by the first
    video's own duration, which clears :data:`CLIP_PASS_GAP_S` for any video
    over two minutes and invents independence (review, 2026-09-14).
    """
    ...

def _clip_independence(extractions, rule: str=CLAUSE6_RULE_DEFAULT) -> dict:
    """GATE CLAUSE 6, per pair, published as a matrix (spec 4.3).

    Independence is a property of a PAIR, so it is scored over every (i, j) and
    the clause passes when at least one pair is independent -- that pair is the
    one that identifies the lateral axis; a third, correlated window adds
    information to the shared field without invalidating it.  The pair-level
    verdicts are published so the separation statistic can be restricted to
    independent pairs when the pairwise-max gate lands.

    **WHERE THE PASS COMES FROM** (boxfix, 2026-09-14).  ``pass_id`` and
    ``creation_time`` are read off ``record.summary.time``, and the path that
    puts them there is: ledger ``videos.csv`` -> the window prep writes
    ``clip_meta.json`` (``pass_id``, ``creation_time_utc``,
    ``creation_time_convention``) -> ``fit_video.load_clip_meta`` ->
    ``world_frame.clip_meta_pass`` -> ``world_frame.time_block`` ->
    ``summary.time`` -> here.  **No prep wrote any of them until the boxfix**,
    so this clause fell back to "different video" on every record in the corpus,
    and three individuals whose videos the ledger puts in ONE pass published
    ``g_x`` as observed.  When the pass is missing the record says
    ``clause6_basis = "different_video_only"`` and never calls the verdict a
    pass test.  (The VIDEO is a different field and a different sidecar key:
    ``clip_meta.source_video`` -> ``summary.scale.source_name`` ->
    :func:`_clip_video`.)

    ``rule`` names which reading the record's ``independent`` verdict follows.
    Both are always computed and published -- ``clause6_and`` and ``clause6_or``
    -- so a report can print the pair of numbers instead of one.
    """
    ...

def _station_visibility(extractions, stations: int) -> dict:
    """GATE CLAUSE 7 (spec 4.3): was each station actually inside the frame?

    Reads the FIT record's ``summary.girth.stations_in_frame_frac`` -- a
    per-station fraction of the clip's contributing frames on which that station
    was in frame, or ``{"n_frames": N, "fraction": [...]}`` -- and pools it over
    the clips by frame count.

    **That is the name task F1's producer must write.**  This field and the
    FUSED block's ``girth["stations_observed"]`` are two different payloads:
    the fused one is this function's own clause-7 verdict (or, historically, a
    bare list of per-station BOOLEANS), and
    ``schema.resolve_observed_or_prior`` raises on anything else.  One key
    carrying both shapes at two levels is what forced merge repair cdf2f64,
    where reading the ``not_evaluated`` block as a list made every girth row of
    every real record raise.  ``stations_observed`` is still read here as the
    pre-rename spelling, so a record written against the old name is not
    silently ignored.  **No record on disk carries either field**
    (the per-station in-frame flags are one of the five TRUNK_ONLY fitting edits
    deferred to phase 6), so the clause reports ``not_evaluated`` on them rather
    than passing them: a WHOLE-lane clip is admitted by gates that require both
    ends detected, which is an argument for the clause, not a substitute for it.
    """
    ...

def _apply_station_clause(out: dict, visibility: dict) -> None:
    """Mark the stations clause 7 refuses -- that station only."""
    ...

def _joint_blocks(extractions) -> list[dict] | None:
    """The clips' ``summary.girth.joint_fit`` blocks -- if they are ONE fit.

    Returns ``None`` unless every extraction carries the block and they all
    stamp the same ``id``, which is a hash of (individual, sorted clip ids,
    config digest).  That check is the point of the id: N single-clip fits that
    happen to sit in one directory must NOT be read as a joint fit.
    """
    ...

def _joint_field(blocks: list[dict], stations: int) -> np.ndarray | None:
    """The shared (stations, 2) field, after checking the clips agree on it.

    There is nothing to fuse -- the clips were fitted against ONE tensor -- so
    this passes the field through.  Silently averaging two fields that should be
    identical would hide exactly the bug this check exists to catch.
    """
    ...

def _independent_separation(ident: dict, fit_names: Sequence[str], independence: dict) -> dict:
    """The separation clause 2 is gated on: the max over INDEPENDENT pairs.

    Clause 2 took the max over ALL (i, j) while clause 6 passed on ANY ONE
    independent pair, and nothing tied them together -- so ``g_x`` could be
    called observed on the geometry of a pair the record itself marks
    ``independent: false``.  Reproduced on three synthetic clips: two windows of
    ONE video 40 deg apart plus a third video 8 deg from each passed both
    clauses, with every independent pair at a third of the 25 deg bar.

    Needs the pairwise matrix (written since 2026-09-13, task C2).  A record
    fitted before it publishes one scalar and no pairing: at N = 2 that scalar
    IS the single pair clause 6 scored, and above it the pair is unknown and
    said to be unknown rather than assumed.
    """
    ...

def _attribution_sentence(head: dict) -> str:
    """The non-broadside clips' resolution and 2D fit quality, in one sentence.

    Spec 3.6 / R9.  A ``prior`` verdict on the lateral axis has two very
    different causes -- the animal was not rolled (GEOMETRY) or the mask was too
    small to see the roll it had (RESOLUTION) -- and over 511 trusted windows
    those two are correlated: Spearman(dvL, ``L_px_native_median``) -0.487 and
    Spearman(dvL, ``fit_rms_over_L``) +0.712, so the windows that could observe
    ``g_x`` are systematically the smallest and worst-fitting.  The refusal
    therefore names the numbers rather than leaving the reader to guess.

    Empty when the fit record carries no ``refusal_attribution`` block (every
    record written before 2026-09-13), which is itself the honest answer: the
    attribution cannot be made for those.
    """
    ...

def _clause6_reading(independence: dict, rule: str) -> dict:
    """One clause-6 READING of a published independence block (boxfix).

    ``_clip_independence`` scores every pair under both rules and publishes the
    adopted one at the top level.  This re-points ``matrix`` / ``independent`` /
    ``pairs[*].independent`` at the OTHER reading, so the whole downstream chain
    -- ``_independent_separation``, the gated pair, the amplification, the g_x
    verdict -- can be re-run against it without a second pass over the records.
    """
    ...

def _g_x_stage(independence: dict, ident: dict, fit_names: Sequence[str], *, n_clips: int, any_fitted: bool, operator_unbiased: bool, tie_off: bool, sep_published, amp_published) -> tuple:
    """``(stage, sep, amp, gated)`` -- which clause decides ``g_x``, and on what.

    ``stage`` is a member of :data:`G_X_STAGES`; ``"observed"`` means every
    clause passed.  ``sep`` and ``amp`` are the separation and the noise
    amplification the verdict was read at, restricted to the pairs THIS
    clause-6 reading admits, and ``gated`` is that restriction's own block.
    """
    ...

def _fuse_girth_joint(extractions, blocks: list[dict], stations: int, clause6_rule: str=CLAUSE6_RULE_DEFAULT) -> dict:
    """The joint-record path: pass the shared field through, gate on the pair.

    ``clause6_rule`` names WHICH reading of clause 6 the record adopts; both
    are computed and published either way (see :func:`_clip_independence`).
    """
    ...

def _record_girth_field(record, stations: int) -> np.ndarray:
    """(stations, 2) girth field of one record: the per-frame median when the
    frames carry ``girth``, else the summary's ``fitted`` (dict or array), else
    ones (field disabled)."""
    ...

def fuse_girth(extractions: Sequence[ClipExtraction], clause6_rule: str=CLAUSE6_RULE_DEFAULT) -> dict:
    """Per station, per axis: who observed it, and is it still on the prior?

    Each clip constrains the axis lying in its image plane and its own record
    names which one and how strongly (``girth.observed_axis``).  Weights are
    those Jacobian norms.  A station whose every contributing clip was broadside
    has its LATERAL axis unobserved and must never be reported as a measured
    girth.

    ``clause6_rule`` is the reading of gate clause 6 a JOINT record's verdict
    follows -- ``"and"`` (the shipped default: a different video AND a different
    recording pass) or ``"or"``.  Both are always computed and published; this
    only names which one the record adopts, and it is plumbed to here so the
    weaker reading can be asked for rather than only read about.
    """
    ...

def resolve_girth_observation(extractions: Sequence[ClipExtraction], assets: MorphAssets | None=None) -> tuple[list[ClipExtraction], dict]:
    """Decide the girth verdict, then rebuild the bands it earns.

    The two passes spec R2 requires, in one place.  Pass one is
    ``fuse_girth``: the verdict is a statement about the individual's clips
    SEEN TOGETHER, so no clip can produce it alone.  Pass two hands that verdict
    back to each clip's sensitivity band, so an OBSERVED axis stops being
    banded at what an unobserved field could have reached
    (``GIRTH_BAND_G_X`` / ``GIRTH_BAND_G_Z``, the D1 5c ``honest_band``) and is
    banded at its own fitted field +- 8 % instead (spec 5.3).

    Two conditions, both stated in the returned ``band_basis``:

    * ``assets`` -- the rebuild re-evaluates the mesh, so a caller with no
      assets keeps the prior band.
    * **at least two clips.**  The +-8 % floor was measured on the TWO-VIEW arm
      (``roll/gate/gate.md`` 1).  On a single-clip record the lateral axis
      rides the prior by construction and the level error of ``g_z`` ALONE has
      never been scored, so a single-clip record keeps the ``honest_band`` --
      spec 5.3's stated dependency, and task D5 is what lifts it.

    Returns ``(extractions, girth)``.  The extractions are the ones passed in
    where nothing was rebuilt.
    """
    ...

@dataclass
class IndividualResult:
    individual: str
    clips: list[str]
    rows: list[MorphometricRow]
    shape: dict
    girth: dict
    scale: dict
    provenance: dict

    def row(self, name: str) -> MorphometricRow:
        ...

    def as_dict(self) -> dict:
        ...

def aggregate_individual(results: Sequence[ClipResult], extractions: Sequence[ClipExtraction], assets: MorphAssets | None=None, density_kg_m3: float=BODY_DENSITY_KG_M3, girth: dict | None=None, refuse_on_insufficient_support: bool | None=None) -> IndividualResult:
    """Fuse every clip of one animal.

    Order-invariant: clips are sorted by id before anything is combined, and
    every combining step (weighted median, precision-weighted mean, sum) is
    symmetric.

    Model-space quantities are recomputed at the FUSED shape rather than
    averaged over per-clip morphometrics -- averaging non-linear functions of
    the shape would give a different, wrong answer -- and the statistic is the
    weighted median over the POOLED gated frames, the same statistic the
    per-clip rows report over their own frames (see ``_fused_model_values``).
    Projected quantities pool the gated frames of every clip too, and the
    between-clip spread is reported SEPARATELY because with more than one clip
    it is expected to dominate the within-clip one.

    ``girth`` is the fused girth block from ``resolve_girth_observation``.  Pass
    the SAME block the clip rows were built with, or the individual's labels and
    its clips' will disagree; with ``None`` it is recomputed here, which is
    correct but cannot rebuild the clips' bands (see
    ``resolve_girth_observation``).
    """
    ...

def _fused_model_values(results: Sequence[ClipResult], extractions: Sequence[ClipExtraction], fused_betas: np.ndarray, assets: MorphAssets) -> tuple[dict[str, float], dict]:
    """Model-space quantities at the FUSED betas, over the POOLED gated frames.

    A model-space quantity is not a constant of the shape alone: it is measured
    on the POSED mesh, so ``TL`` on a hard-beat frame is shorter than on a
    straight one and every ratio built on it moves with the pose.  The per-clip
    path therefore reports the (weighted) median over that clip's gated frames,
    and this is the same statistic over the pooled gated frames of every clip,
    each frame carrying its own clip's weight.

    It replaces an earlier single-frame evaluation at the heaviest clip's
    reference frame, which made the fused row a FRAME statistic quoted beside a
    clip statistic: on FAR18101701 that put the fused ``TL`` at 4.5032 model u
    below BOTH clips (4.5330 and 4.5524), and the ratio of the fused values
    ``max_girth / TL`` at 0.5186 outside the per-clip 0.5149 .. 0.5154 and
    outside the row's own ``value_bl``.
    """
    ...

def _record_summary_girth(record, n_stations: int):
    """The record's clip-level girth field, or ``None`` -- as ``extract_clip``."""
    ...

def _fuse_scale(results: Sequence[ClipResult], extractions: Sequence[ClipExtraction]) -> dict:
    """One label for the individual, reconciled -- not "the first clip's".

    ``scale_resolve`` (spec 4.6) replaces what this used to do, which was to
    take ``bounded[0]``: the first clip **in directory order** that carried a
    length field.  That is not a rule, and on an individual whose own sources
    disagree it silently published one of them.  The rule now is: quarantine
    ``catalog_size_ft``, group the surviving rows by ``independence_group``
    (five in-corpus sources are ONE observation), take the median of the
    distinct values in the best-ranked group, and refuse -- ``contested``, no
    ``value_m``, ratio lane only -- when the distinct values spread by more
    than 5 %.

    The mesh span stays clip 0's, because the clips of a joint fit are fitted
    against one shared betas tensor and their spans are the same number; the
    disagreement between them is published as ``mesh_span_disagreement_frac``
    and added to the band.
    """
    ...

def _fuse_row(name, spec, results, extractions, model_values, mpu, scale_unc, scale, density, girth=None, truncation=None) -> MorphometricRow:
    ...

def rows_to_csv(rows: Sequence[MorphometricRow]) -> str:
    ...

def rows_to_markdown(rows: Sequence[MorphometricRow], title: str) -> str:
    ...
