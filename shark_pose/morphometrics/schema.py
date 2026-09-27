"""Output vocabulary for the 3D morphometrics layer.

The names, definitions and observability flags come from the D1 spec
(``outputs/demo_2026-09-03/work/morphometrics/spec/spec.{md,json}``).  Where the
2D tool (``shark-morphometrics/shark_pose_project``) ships a column with the
same meaning, the name here is that column's, so the two tables can be joined.

Three vocabulary rules the rest of the package enforces:

1. ``ci_kind`` is one of ``posterior`` / ``sensitivity`` / ``spread``.  Nothing
   in this pipeline produces a POSTERIOR -- the fit returns a point estimate of
   the shape with no covariance, no Hessian and no sampling -- so the value is
   declared and never emitted.  A model-space number gets ``sensitivity``
   (a named perturbation), a projected number gets ``spread`` (a real
   distribution over gated frames).  Calling the second one the first is the
   failure mode this vocabulary exists to prevent.
2. An absolute (metric) value on a clip with no metric scale is ABSENT:
   ``value_m = None`` with ``value_m_reason`` set.  Never zero.
3. ``value_bl`` is the body-length-normalised value: ``value / TL ** bl_power``.
   It is the only column that is comparable between clips today, because the
   metric scale exists on one clip out of two.
4. ``ci_low`` / ``ci_high`` are ALWAYS in ``unit`` -- the model-unit, pixel or
   ratio band -- and never carry the metric-scale uncertainty.  The band that
   belongs beside ``value_m`` is ``value_m_ci_low`` / ``value_m_ci_high``, which
   is the same band with the metric-scale factor
   ``((1 - u) ** scale_power, (1 + u) ** scale_power)`` folded in.  Joining
   ``(value_m, ci_low, ci_high)`` is a unit error and an over-confident one:
   the demo's scale band alone is +/-10-20%.
5. A quantity whose 3D construction is not the 2D column's construction carries
   ``twod_comparable = False``.  Its ``twod_column`` is still named so a reader
   can find the counterpart, but no ``in_expected_range`` verdict is emitted
   against a range the two definitions do not share.  The one exception is
   ``RANGE_ORIENTATION_EXCEPTIONS``: a row whose only mismatch is that the 2D
   product ships the RECIPROCAL, where ``EXPECTED_RANGES`` is stated in this
   module's orientation and therefore does apply.
6. ``QuantitySpec.observed_or_prior`` is the vocabulary's DEFAULT, not the
   reported label.  For the rows of ``GIRTH_FIELD_QUANTITIES`` the reported
   label is the fit's own girth verdict, resolved per record by
   ``resolve_observed_or_prior`` -- a static ``prior`` there is what made the
   one record in this corpus with an observed lateral girth report ``prior``
   and grade ``low`` in the same file that says the axis was observed.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass(frozen=True)
class QuantitySpec:
    """One row of the output vocabulary."""
    name: str
    definition: str
    observed_or_prior: str
    ci_kind: str
    domain: str
    unit: str
    bl_power: int = 0
    scale_power: int = 0
    twod_column: str | None = None
    scale_free: bool = False
    twod_comparable: bool = True

    def __post_init__(self) -> None:
        ...

def _q(*args: Any, **kw: Any) -> QuantitySpec:
    ...

def resolve_observed_or_prior(name: str, spec: QuantitySpec, girth: dict | None=None) -> str:
    """The row's observability, COMPUTED from what the fit observed.

    ``spec.observed_or_prior`` is the vocabulary's *default*.  For a row in
    ``GIRTH_FIELD_QUANTITIES`` it is only a default: the answer is the fused
    girth block's own verdict, which is the gate of ``aggregate.fuse_girth``
    (spec 4.3) and travels with the record in ``girth.axes.{g_x,g_z}.observed``
    and its ``reason``.

    Both axes are required.  A perimeter, an area and a volume are all
    functions of BOTH semi-axes, so an individual whose lateral axis rides the
    tie prior -- every broadside-only fit, and every pair the separation clause
    refuses -- reports ``prior``, which is the shipped behaviour and stays it.

    Only the WHOLE-PROFILE rows follow the verdict.  ``girth_station_*``,
    ``area_station_*``, ``girth_at_*`` and ``area_at_*`` are per-station reads
    of a field spec 4.4 says is never observed, so they keep ``prior`` on every
    record (``PER_STATION_NEVER_OBSERVED`` carries the reason).

    ``girth["stations_observed"]``, when the record carries it (spec 4.3
    clause 7), is a boolean per GIRTH STATION saying the station was in frame on
    enough contributing frames.  A row that integrates or maximises over the
    profile -- which, after the restriction above, is every row this function
    can promote -- needs them all.
    A length that disagrees with ``girth["n_stations"]`` is an error and is
    raised, not silently ignored: reading a 13-joint in-frame list as a 7-station
    one would hand out an ``observed`` label for a station nobody tested.

    The clause's producer (``aggregate._station_visibility``, task D2) publishes
    the whole clause-7 block there rather than a bare list, so both shapes are
    read: a mapping is the clause's own verdict and its ``reported`` list is the
    per-station answer, while ``state != "evaluated"`` means the clause could
    NOT be run and refuses no station -- the same position as a record that
    carries no clause at all.  A mapping that claims to be evaluated without a
    ``reported`` list, or any other type, is an error and is raised: silence
    about the shape is how a station nobody tested gets labelled ``observed``.

    Returns one of ``OBSERVED_OR_PRIOR``.  Never widens a row: a row that is
    ``observed`` or ``observed_broadside_only`` in the vocabulary is returned
    unchanged, because those labels also decide WHICH FRAMES the row is measured
    over and are not this function's to move.
    """
    ...

def range_verdict_applies(name: str) -> bool:
    """Whether ``in_expected_range`` may be emitted for ``name``.

    True where the 3D construction IS the 2D column's construction, and for the
    orientation exceptions above, whose band is stated in the 3D orientation.
    """
    ...

@dataclass
class MorphometricRow:
    """One reported quantity, with everything needed to read it honestly."""
    name: str
    value: float | None
    unit: str
    value_bl: float | None = None
    value_m: float | None = None
    value_m_ci_low: float | None = None
    value_m_ci_high: float | None = None
    value_m_reason: str | None = None
    value_arc: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_kind: str | None = None
    n_frames_used: int = 0
    state_reason: str | None = None
    twod_column: str | None = None
    in_expected_range: bool | None = None

    def __post_init__(self) -> None:
        ...

    def to_dict(self) -> dict[str, Any]:
        ...

def quality_grade(spec: QuantitySpec, n_frames_used: int, template_driven: bool, state: str='ok', broadside_frac: float | None=None, observed_or_prior: str | None=None) -> str:
    """Grade a row.

    The rule, stated so a reader can disagree with it:

    * ``none``   -- the row carries no number (unmet scale, not computable).
    * ``low``    -- the quantity is prior-driven, OR the clip is
      template-driven (girth off and the fitted betas within 1% of one PCA sd),
      in which case a model-space number is a constant of the ASSET and says
      nothing about this animal, OR the row is IMAGE-domain on a clip where NO
      gated frame is broadside.
    * ``medium`` -- observable only on a broadside frame, fewer than 30 gated
      frames went into it, or the row is image-domain and fewer than half the
      gated frames are broadside.
    * ``high``   -- observed, on 30 or more gated frames.

    ``broadside_frac`` is the fraction of the clip's GATED frames that pass the
    obliqueness gate.  It only touches image-domain rows: a projected length is
    a foreshortened length, and a clip with no broadside frame cannot support a
    high grade on one however many frames it has.  Pass ``None`` (the default)
    where the caller has no gate -- the grade is then the pre-existing one.

    ``observed_or_prior`` is the row's RESOLVED label
    (``resolve_observed_or_prior``), which for a girth-field row is the fit's
    own verdict rather than the vocabulary's default.  The grade follows it, so
    a girth the pair actually observed stops being graded ``low`` on the
    strength of a schema string.  Pass ``None`` to grade against the default.
    """
    ...
