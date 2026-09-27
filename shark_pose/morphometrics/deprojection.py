"""De-projected companions to the broadside-only projected rows (S5 lane
``broadside_clause``, 2026-09-16).

**What the broadside clause is.**  ``aggregate.gate_frames`` splits the frames
the other three gates keep into a ``broadside`` subset -- ``obliqueness_deg <=
broadside_max_deg`` (20 deg by default) -- and ``aggregate._build_row`` takes
the median of a quantity over that subset instead of over all kept frames
whenever the quantity's ``observed_or_prior`` is ``observed_broadside_only``.
``obliqueness_deg`` is the fit's OWN quantity: ``extract_frame`` computes it as
``arcsin(|axis_z| / |axis|)`` of the posed spine's end-to-end vector in the
camera frame.  It is not the ledger's dvL-derived ``obl_deg_est`` and no ledger
column reaches this gate.

**Why a correction is possible.**  A projected landmark chord is foreshortened
by the cosine of the angle ITS OWN 3D chord makes with the image plane, which
is not the body axis's obliqueness: a dorsal fin standing perpendicular to the
spine is barely foreshortened at the yaw that halves the projected total
length.  ``extract_frame`` now records that per-chord cosine (``*_oop_cos``),
so the de-projected pixel length is simply ``px_proj / cos``.

**Why the corrected rows are companions and not replacements.**  De-projecting
a landmark chord returns the MODEL-SPACE chord times the local pixel scale --
the projected row carries no information the model-space row does not already
carry, because both are readings of the same fitted mesh.  The corrected rows
exist to make that identity auditable on real records, and the conclusion the
lane draws from them is a vocabulary statement, not a new measurement: the
``*_px_proj`` rows are image-domain COMPARISON rows for the 2D tool, rightly
broadside-gated, while the model-space rows that used to carry the same
``observed_broadside_only`` label (``DH``, ``PL``, ``CS`` and their three
ratios) are 3D distances on the posed mesh and are not foreshortened at all --
S6 lane ``morph_v4`` acted on that conclusion and moved those six rows to
``observed``, so they are now medians over ``gates.keep`` like ``TL``.

Everything here is OFF by default: ``build_deprojected_rows`` is called only
when a caller passes ``--deprojected-rows``.  No shipped record changes.
"""
from __future__ import annotations
import numpy as np
from .schema import MorphometricRow, QuantitySpec
MIN_OOP_COS = 0.2

def _spec(name: str, src: str, unit: str, twod: str | None, definition: str | None=None) -> QuantitySpec:
    ...

def deprojected_series(per_frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The de-projected per-frame series, NaN where the cosine is too small."""
    ...

def build_deprojected_rows(extraction, gates, median_mad, bootstrap_median_ci):
    """One ``MorphometricRow`` per de-projected companion, over ``gates.keep``.

    ``median_mad`` and ``bootstrap_median_ci`` are passed in so this module
    does not import ``aggregate`` (which imports the schema this module also
    imports); the caller hands it the same two functions the shipped rows use.
    """
    ...
