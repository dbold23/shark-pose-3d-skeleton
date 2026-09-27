"""Per-frame 3D morphometrics from a v2 fit record.

This module READS fit records.  It changes no default, touches no asset and
never re-fits: given a record written by ``scripts/fit_video.py --model v2`` and
the asset directory the fit used, it re-runs ``SharkSMPL`` forward on the stored
parameters and measures the posed mesh.

The constructions that decide every number here are stated up front, because
the number moves with the construction:

**The volume is a station integral, not a divergence integral.**  The shipped
``faces.npy`` is manifold-WITH-BOUNDARY (1954 of 9419 unique edges have one
incident face; 1943 boundary vertices in 142 open loops; V - E + F = 121), so
the divergence theorem does not apply to it -- run naively it overstates by
~70%.  ``divergence_volume`` is provided and is exact on a closed mesh, but
``section_profile``/``volume`` is what this layer reports, and
``mesh_watertightness`` is checked before the divergence number is ever quoted.

**Cross-sections are EXACT PLANE SLICES perpendicular to the posed spine.**
Axis = the posed 13-joint spine polyline, the same polyline the girth field
uses.  At each station the body surface is cut by the plane through the
polyline point, normal to the polyline direction there; ``a`` and ``b`` are
HALF the lateral and dorsoventral extents of that contour, measured in the BODY
frame of the spine joint the station hangs off, so a bent animal is measured
across its body and not across the world.  Half-extents rather than distances
from the axis, because this rig's spine joints do not sit on the body midline
-- at spine fraction 1/3 every body vertex is on one side of the axis -- and a
semi-axis anchored on the axis reads that rig error as shape.

**The section's AREA is read off the contour, not off ``pi * a * b`` (D4b).**
``a`` and ``b`` BOUND the section, so the ellipse through them is an envelope
and reading it as an area over-states the body.  The fact that settles it is
not the size of the gap but its sign: ``pi * a * b`` sits 5.31% above the
CONVEX HULL of the very same contour points -- an upper bound on the area that
contour can enclose -- and above it at 86% of the rest template's stations.  It
is therefore superseded, and it is NOT an end of the replacement's band.

What ships is that hull: 0.8671 model u3 on the rest template against the
envelope's 0.9131.  The hull rather than the contour's own shoelace area,
because the contour this asset produces is neither closed nor single-loop --
the fin gate removes body wall, so 13% of the section's perimeter is missing at
the median station and 72% at the worst, and forward of spine fraction 0.3 the
plane cuts the mouth and branchial cavities as well as the body wall.  The hull
depends on the point SET and not on an ordering of it, so it survives both; on
a convex section (every trunk station here) it IS the section, up to the
polyhedral inscription of the mesh, which is one-sided low and worth 0.16% at
64 contour points.  The angularly sorted shoelace is kept as the LOW end of a
named ``section_shape`` band on every area, volume and mass row -- worth -11.5%
on the rest template's volume, more than an order of magnitude wider than the
``bins`` term beside it.  The PERIMETER half of the construction is untouched:
``girth`` is Ramanujan of the same semi-axes, sits 1.6% above the hull's own
perimeter at the median station, and no girth row carries the term.

**Fins are out.**  A triangle is cut only when all three of its vertices are
body (summed non-spine LBS weight <= ``GIRTH_FIN_WEIGHT_MAX``).  The fins are
open sheets in this asset and enclose no volume at all, the girth gate already
excludes them, and body-condition indices are body-only in the literature.
``fin_volume`` is reported absent with that reason, never as zero.

**Why the slice and not a vertex bin (D4).**  Through 2026-09-06 the profile
was built by binning BODY VERTICES into 64 arc bins and taking the q95 of each
bin's offsets.  That rule reads the mesher, not the animal: 18 of the 64 trunk
bins hold 1-11 body vertices and 10 hold fewer than four, so those bins needed a
population gate and a fill rule that the D1 spec never wrote down, and the
volume tracked the bin count instead of the geometry -- 0.9498 model u3 at 16
bins falling to 0.5774 at 256, a factor of 1.64, with 48% of the integral
coming from bins holding fewer than ten vertices.  Resampling the surface
area-uniformly before binning fixes the sample DENSITY (the volume then settles
at 0.834 for 50k samples and up) but not the bin dependence (still x1.20 over
16..128), because pooling a q95 over a slab is biased by however much the body
tapers across the slab.  The plane slice removes both: the cross-section at a
station is a geometric object, no station can be under-populated (the sparsest
contour on this asset has 32 points), the gate and the fill rule stop existing,
and ``bins`` becomes a pure quadrature knob.  Integrated by composite Simpson
over the cell edges and centres, the rest-template volume is 0.8671 model u3
and moves 0.95% over 16/32/64/128 cells and 2.22% over 16..256.  The superseded
numbers -- 0.9131 for this same slice read with the elliptical area rule, 0.7082
for the vertex bin with linear fill, 0.6250 published in the D1 spec -- are kept
in ``SUPERSEDED_TEMPLATE_VOLUMES`` with the rules that made them; the D1 spec's own 0.6250 is NOT reproducible from its literal text under
any combination of population gate, fill rule, station centring and integrator
(160 variants span 0.34 to 0.71, the closest landing 3.0% away), which is one
more reason the construction had to be settled rather than matched.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from ..model_3d.shark_smpl import GIRTH_FIN_WEIGHT_MAX, GIRTH_GATE_START, GIRTH_STATION_FRACTIONS, SharkSMPL, select_visible_landmarks
PROFILE_BINS = 64
PROFILE_BINS_SWEEP = (16, 24, 32, 48, 64, 96, 128, 192, 256)
PROFILE_ARC_GUARD = 0.25
PEDUNCLE_SEARCH_START = 0.8
RING_WEIGHT_MIN = 0.5
BODY_DENSITY_KG_M3 = 1060.0
LM_SNOUT_TIP = 0
LM_GILL_SLIT = 2
LM_PECTORAL_BASE = 3
LM_PECTORAL_TIP = 4
LM_FIRST_DORSAL_BASE = 5
LM_FIRST_DORSAL_TIP = 6
LM_PELVIC_FIN_TIP = 9
LM_CAUDAL_NOTCH = 11
LM_CAUDAL_UPPER_TIP = 12
LM_CAUDAL_LOWER_TIP = 13

def mesh_watertightness(faces: np.ndarray, num_vertices: int | None=None) -> dict:
    """Decide whether the divergence theorem applies to a triangle soup.

    A closed, consistently oriented manifold has every undirected edge shared by
    exactly two faces and every directed edge appearing exactly once.  Anything
    else -- a boundary edge, a duplicated directed edge -- and the enclosed
    volume is not defined by the surface integral.

    Returns:
        dict with ``watertight``, the edge and boundary-loop counts, the Euler
        characteristic and, when not watertight, ``reason``.
    """
    ...

def divergence_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume enclosed by a CLOSED triangle mesh (divergence theorem).

    ``V = 1/6 * sum over faces of (v0 x v1) . v2`` for an outward-oriented,
    watertight surface.  Exact for a box; converges to the true volume of a
    sphere as the tessellation refines.

    This is a cross-check only.  Call ``mesh_watertightness`` first: on an open
    surface this number is meaningless (it overstates the shipped shark asset by
    ~70%), and ``station_volume`` is the primary construction.
    """
    ...

def polyline_arc(polyline: np.ndarray) -> np.ndarray:
    """Cumulative arc length along a polyline, starting at 0."""
    ...

def assign_to_polyline(vertices: np.ndarray, polyline: np.ndarray) -> dict:
    """Nearest point on a polyline for every vertex.

    Mirrors ``build_girth_operator``'s assignment exactly (clamped projection on
    every segment, nearest wins), so girth, area and volume are measured on the
    same parameterisation the girth field can actually move.

    Returns:
        dict with ``foot`` (N, 3), ``fraction`` (N,), ``segment`` (N,),
        ``arc`` (N,) absolute arc position, and ``total`` arc length.
    """
    ...

def global_joint_rotations(local_rotmats: np.ndarray, parents: Sequence[int]) -> np.ndarray:
    """Compose per-joint local rotations into world rotations along the tree."""
    ...

def _topological_order(parents: Sequence[int]) -> list[int]:
    ...

def ramanujan_perimeter(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Ramanujan's second approximation to the perimeter of an ellipse."""
    ...

def _station_frames(polyline: np.ndarray, fractions: np.ndarray, joint_rotations: np.ndarray | None=None, spine_indices: Sequence[int] | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cutting plane of each station: point, normal, and in-plane axes.

    The normal is the spine polyline's own direction there, so the plane is
    perpendicular to the axis the girth field is parameterised on.  The
    in-plane axes are the lateral (x) and dorsoventral (z) axes of the spine
    joint the station hangs off, orthogonalised against the normal, so a bent
    animal is cut across its own body and not across the world.
    """
    ...

def _contour_area(x: np.ndarray, z: np.ndarray) -> float:
    """Shoelace area of a station's contour points, sorted by angle about their
    centroid.

    The LOW end of the ``section_shape`` band, not the shipped rule.  It equals
    ``_hull_area`` wherever the points are in convex position -- every trunk
    station of this asset -- and reads low everywhere else: it chords across the
    concavities and across the gaps the fin gate leaves, and where the plane
    cuts an interior surface as well as the body wall (the mouth and branchial
    cavities of this asset, forward of spine fraction 0.3) the angular sort is
    not a traversal of the boundary at all and the area it returns is not one.
    That is why it bounds rather than ships.
    """
    ...

def _hull_area(x: np.ndarray, z: np.ndarray) -> float:
    """Area of the convex hull of a station's contour points (monotone chain).

    The SHIPPED area rule.  A shark's trunk section is convex, and on a convex
    section the hull of the cut points is the section itself up to the
    polyhedral inscription of the mesh (one-sided LOW, 0.16% at 64 contour
    points, 0.6% at 32).  It is also the only rule here that survives what this
    asset's slices actually look like -- open across part of their perimeter
    where the fin gate removed body wall, and multi-loop in the head -- because
    it depends on the point SET and not on any ordering of it.  Where the true
    section is concave (the caudal keel) it reads high, and the band's low end
    carries that.
    """
    ...

def _slice_extents(tri_points: np.ndarray, tri_lo: np.ndarray, tri_hi: np.ndarray, fractions: np.ndarray, centre: np.ndarray, normal: np.ndarray, ex: np.ndarray, ez: np.ndarray, arc_guard: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Semi-axes AND enclosed areas of each station's exact mesh cross-section.

    Every body triangle that straddles the plane contributes the two points
    where the plane cuts its edges; ``a`` and ``b`` are HALF THE EXTENT of that
    contour along the in-plane lateral and dorsoventral axes, and the two areas
    returned beside them are what that same contour encloses (``_contour_area``)
    and what its convex hull encloses (``_hull_area``).

    Extents, not a quantile: on an exact contour the envelope IS the extent,
    and a quantile over contour points would depend on where the mesher
    happened to put its edges -- the very dependence this construction exists
    to remove.  Half the full extent, not the largest distance from the axis:
    this rig's spine joints do not sit on the body midline (at spine fraction
    1/3 every body vertex of this asset is on one side of the axis), and a
    semi-axis measured from the axis reads that rig error as shape, while a
    half-extent is invariant to where the axis sits inside the section.

    The extents describe the section's ENVELOPE and are the right thing for a
    perimeter; they are NOT an area rule, which is why the area is integrated
    from the contour rather than from ``pi * a * b`` (see ``AREA_RULE``).
    """
    ...

def section_profile(vertices: np.ndarray, polyline: np.ndarray, body_mask: np.ndarray, faces: np.ndarray, joint_rotations: np.ndarray | None=None, spine_indices: Sequence[int] | None=None, bins: int=PROFILE_BINS, arc_guard: float=PROFILE_ARC_GUARD, area_rule: str=AREA_RULE) -> dict:
    """Cross-section profile along the spine polyline.

    The cross-section at an arc fraction is the EXACT intersection of the
    body surface with the plane perpendicular to the spine there (D4, spec
    3.1).  Nothing in it is binned: the semi-axes are the extents of that
    contour, so no station can be under-populated, there is no population gate
    and no fill rule, and refining ``bins`` refines only the quadrature.

    The section's AREA is read off that contour (``AREA_RULE``), not off
    ``pi * a * b``: the semi-axes BOUND the section, so the elliptical envelope
    is not an area but an over-estimate of one -- on this asset it exceeds even
    the contour's own CONVEX HULL, which bounds the area the contour can
    enclose, at 86% of the rest template's stations.  The hull is what ships.
    All three areas are returned per station, and ``volume_by_area_rule``
    carries the volume under each.

    Args:
        vertices: (N, 3) posed mesh vertices.
        polyline: (S, 3) posed spine polyline, snout -> tail.
        body_mask: (N,) True on body vertices (fins excluded).
        faces: (F, 3) triangles; a triangle is cut only when all three of its
            vertices are body, which is how the fins stay out.
        joint_rotations: (J, 3, 3) world rotations per joint; when given, the
            in-plane axes are the body frame of the spine joint the station
            hangs off.  None means the mesh is already in the body frame.
        spine_indices: joint index of each polyline point; needed with
            ``joint_rotations``.
        bins: integration cells of the station integral.
        arc_guard: half-width, in arc fraction, of the band of triangles a
            station's plane is cut against.
        area_rule: which area the ``area`` profile and the volume are built
            from -- one of ``AREA_RULES``.  ``"hull"`` ships; ``"contour"`` is
            the shoelace over the same points and is the low end the
            ``section_shape`` band is measured against; ``"ellipse"`` is the
            superseded ``pi * a * b``.

    Returns:
        dict with the per-bin ``a`` / ``b`` / ``area`` / ``girth`` profiles at
        the bin centres, the same bins' areas under every rule, the Simpson
        nodes they were integrated on, the arc length, the volume under the
        chosen rule and under every rule, and the per-vertex assignment.
    """
    ...

def station_population(profile: dict) -> dict:
    """How well resolved each station's contour is.

    The construction cannot be starved of vertices the way the superseded
    vertex-binned envelope was, but a station cut from very few triangles is
    still worth seeing, so the contour counts travel with the number.
    """
    ...

def profile_at(profile: dict, fraction: float, key: str) -> float:
    """Linear interpolation of a profile series at a spine arc fraction."""
    ...

@dataclass
class MorphAssets:
    """Everything about the rig that the extraction needs, loaded once."""
    assets_dir: Path
    model: SharkSMPL
    joint_names: list[str]
    parents: list[int]
    spine_indices: list[int]
    body_mask: np.ndarray
    pectoral_ring: np.ndarray
    dorsal_ring: np.ndarray
    faces: np.ndarray
    watertight: dict
    branchial_span: tuple[int, int]

    @classmethod
    def load(cls, assets_dir: str | Path) -> 'MorphAssets':
        ...

    def forward(self, pose: np.ndarray, betas: np.ndarray, translation: np.ndarray | None=None, girth: np.ndarray | None=None) -> dict:
        """SharkSMPL forward on stored parameters -> numpy arrays."""
        ...

@dataclass
class Record:
    """A v2 fit record, read-only."""
    path: Path
    raw: dict

    @classmethod
    def load(cls, path: str | Path) -> 'Record':
        ...

    @property
    def clip_id(self) -> str:
        """The video stem, unless it is a generic cut name (``window``, ``fit``):
        the corpus windows are all ``window.mp4``, so two clips of one animal
        would collide on disk (per-clip tables overwrote each other,
        2026-09-07); the record's parent directory carries the window tag then."""
        ...

    @property
    def source_name(self) -> str | None:
        ...

    @property
    def individual(self) -> str:
        """Encounter code -- the 2D tool's key for one animal.

        A JOINT fit declares the individual it tied its shared girth field over
        (``fit_individual.py --individual``, cross-checked against every clip's
        ``quality.json`` ``encounter`` by ``check_meta``), and stamps it in
        every per-clip record at ``summary.girth.joint_fit.individual``.  That
        declaration wins over parsing the code out of a file name, because the
        corpus names some videos ``<encounter><clip index>_...``: AN22121502's
        two windows come from ``AN2212150202_M_12.mp4`` and
        ``AN2212150206_M_12.mp4``, and ``_ENCOUNTER_RE`` reads those as two
        different 12-digit encounters -- splitting one animal into two
        single-clip aggregates, each of which then refuses ``g_x`` for want of
        a second view that is in fact sitting right beside it.
        """
        ...

    @property
    def native_px_per_proxy_px(self) -> float:
        ...

    @property
    def fps(self) -> float:
        ...

    @property
    def scale_block(self) -> dict:
        ...

    def metres_per_model_unit(self) -> tuple[float | None, str | None, list[float]]:
        """(value, reason-when-absent, uncertainty fraction band)."""
        ...

def _pinhole(points: np.ndarray, focal: float, principal: Sequence[float]) -> np.ndarray:
    """The fit's own projection: proxy pinhole, focal + principal from the record.

    Reproduces the record's stored ``landmarks_2d_proxy`` to ~5e-5 proxy px, so
    a quantity measured here and one read off the record are the same number.
    """
    ...

def extract_frame(assets: MorphAssets, pose: np.ndarray, betas: np.ndarray, translation: np.ndarray | None=None, girth: np.ndarray | None=None, landmarks_2d_native: np.ndarray | None=None, camera: dict | None=None, native_scale: float=1.0, bins: int=PROFILE_BINS, area_rule: str=AREA_RULE) -> dict:
    """Every quantity of one frame, in model units and (when a camera is given)
    in native pixels.

    Args:
        assets: loaded rig.
        pose: (J*3,) axis-angle.
        betas: (10,) shape coefficients.
        translation: (3,) camera-frame translation, or None.
        girth: (K, 2) girth field, or None for the identity field.
        landmarks_2d_native: (16, 2) stored projected landmarks in NATIVE px;
            when None they are recomputed from ``camera``.
        camera: the record's ``summary.camera`` block (proxy focal + principal).
        native_scale: native px per proxy px.
        bins: integration cells of the station integral.  The cross-section
            at a station is an exact plane slice, so no envelope moves with it
            and only the quadrature does (see ``_bins_sensitivity``); the
            projected station width keeps its own ``PROFILE_BINS``-wide window,
            so it is not swept by this argument.
        area_rule: which area rule the section areas, the volume and the mass
            are built from (``AREA_RULES``); the shipped rule is the contour
            points' convex hull and the alternative is the shoelace over the
            same points, which is what ``_section_shape_sensitivity`` measures
            the band against.

    Returns:
        flat dict of quantity name -> float, plus ``_profile`` diagnostics.
    """
    ...

def _anatomical_fractions(landmarks: np.ndarray, polyline: np.ndarray, profile: dict) -> dict[str, float]:
    """Spine arc fractions of the anatomical stations.

    A landmark's station is the arc fraction of its own nearest point on the
    spine polyline -- the same assignment every vertex gets.  That removes the
    fraction-convention conversion (landmark-axis fraction vs spine fraction)
    that the spec warns is the easiest error to make here: there is only one
    convention in this module, the spine polyline's own.

    The peduncle is not a fixed fraction: it is the profile's own minimum-area
    station aft of ``PEDUNCLE_SEARCH_START``, and the fraction found is
    reported.  On this asset the profile decreases monotonically aft of that
    point (the caudal fin is excluded as a fin, so nothing flares again), so the
    minimum lands on the LAST spine station -- the caudal-axis end, aft of the
    anatomical peduncle.  The rule is kept because it is the spec's, and the
    fraction it found travels with the number so a reader can see where it went.
    """
    ...

def _projected_station_width(vertices: np.ndarray, fraction: np.ndarray, body_mask: np.ndarray, station_fraction: float, landmarks_2d: np.ndarray, camera: dict, native_scale: float, half_window: float=1.0 / PROFILE_BINS) -> float:
    """Projected body width at a station, perpendicular to the projected axis.

    The 2D tool scans a SAM mask along the perpendicular to the snout ->
    caudal-upper axis through the gill midpoint; this is the same measurement
    taken on the fitted body instead of on the mask, so the two are directly
    comparable.  It is a PROJECTED WIDTH, not a girth.
    """
    ...

@dataclass
class ClipExtraction:
    """Per-frame quantities of one clip, plus the sensitivity evaluations."""
    record: Record
    assets_dir: str
    per_frame: dict[str, np.ndarray]
    frame_index: np.ndarray
    detected: np.ndarray
    residual_native_px: np.ndarray
    betas: np.ndarray
    girth_enabled: bool
    template_driven: bool
    reference_frame: int
    girth_band: dict[str, tuple[float, float]]
    betas_band: dict[str, float]
    bins_band: dict[str, tuple[float, float]]
    section_shape_band: dict[str, tuple[float, float]]
    profile: dict
    watertight: dict
    provenance: dict
    seconds_per_frame: float
    spine_end_joints: tuple[int, int] | None = None
GIRTH_BAND_G_X = (0.93, 1.12)
GIRTH_BAND_G_Z = (0.77, 1.25)
DEPTH_JITTER_FRAC = 0.018
GIRTH_OBSERVED_LEVEL_ERROR_FRAC = 0.08

class GirthFieldShapeError(ValueError):
    """A record's girth field is not the ``(K, 2)`` field ``SharkSMPL`` takes."""

def girth_field_from_record(value: Any, num_stations: int, where: str) -> np.ndarray:
    """Coerce a record's stored girth field to the ``(K, 2)`` model input.

    Records write the field in two shapes and both are in the corpus: the
    per-frame ``frames[i].girth`` is already the ``(K, 2)`` array, while
    ``summary.girth.fitted`` is a MAPPING (``g_x`` / ``g_z`` plus the two
    percent-change diagnostics).  Reading the summary as an array raises deep
    inside torch; this converts it and names the record and key when it cannot.
    """
    ...

def extract_clip(record: Record, assets: MorphAssets, max_frames: int | None=None) -> ClipExtraction:
    """Run the extraction over every frame of a record.

    Model-space quantities are recomputed per frame even though the rig's rigid
    bones and the clip's single shared shape make them constant: measuring the
    spread is how we show it is zero rather than assuming it.
    """
    ...

def girth_band_corners(n_stations: int, ref_girth: np.ndarray | None=None, observed: Mapping[str, bool] | None=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """The two ``(K, 2)`` girth fields the sensitivity band is measured at.

    Per axis, and independently (spec 5.3):

    * **prior** -- the D1 spec's 5c ``honest_band`` as an ABSOLUTE uniform
      field, ``GIRTH_BAND_G_X`` / ``GIRTH_BAND_G_Z``.  This is the shipped
      behaviour and is what ``observed = None`` returns, corner for corner.
    * **observed** -- the FITTED field on that axis, scaled by
      ``1 -+ GIRTH_OBSERVED_LEVEL_ERROR_FRAC``.  It re-anchors the axis as well
      as narrowing it, and it is a level error, not a posterior.

    An axis can only take the observed corner when there IS a fitted field
    (``ref_girth``); with none, ``g == 1`` and the axis rides the prior whatever
    a gate says.  Returns ``(lo, hi, basis)``, where ``basis`` names per axis
    what was used and why, for the record's own provenance.
    """
    ...

def _girth_sensitivity(assets: MorphAssets, args: tuple, base: dict, ref_girth: np.ndarray | None=None, observed: Mapping[str, bool] | None=None) -> dict[str, tuple[float, float]]:
    """Re-evaluate every quantity at the two ends of the girth envelope.

    This is not a posterior and it is not reproducibility.  With no observation
    (the default, and what clip extraction can know on its own) the ends are the
    D1 spec's 5c ``honest_band``, i.e. what the fitted field reached at the
    TRUNK stations of the two reference clips, applied uniformly to all seven
    stations.  The field's realised reach over the whole girth pass is wider
    (``GIRTH_FIELD_CORPUS_REACH``).  The lateral axis rides the prior on a
    broadside clip because that axis points at the camera.

    ``observed`` -- the fused girth block's per-axis verdict, which is only
    known once the individual's clips are seen together -- swaps that axis's
    corners for the fitted field scaled by ``1 -+ 8 %``; see
    ``girth_band_corners`` and ``aggregate.resolve_girth_observation``, which
    is what supplies it.

    One measured bias to read the HIGH factor against: the girth field's own
    fin gate blends fin vertices back into the body envelope as the field
    inflates, so a uniform field ``g = s`` scales the recovered max girth by
    slightly MORE than ``s`` and the volume by more than ``s ** 2`` -- +4.7% and
    +2.2% at ``s = 1.25`` (asserted in
    ``tests/test_morphometrics.py::test_uniform_girth_field_scales_the_envelope``).
    The HIGH end of this band is therefore a few percent larger than the girth
    field alone justifies; the LOW end is unaffected.
    """
    ...

def reference_evaluation(extraction: ClipExtraction, assets: MorphAssets) -> tuple[tuple, np.ndarray | None, dict]:
    """Re-run the extraction's REFERENCE frame, exactly as ``extract_clip`` did.

    Returns ``(args, ref_girth, base)`` -- the three things every sensitivity
    is measured against.  ``ClipExtraction`` keeps the reference frame's INDEX
    rather than its evaluation (the row carries a mesh profile and is not worth
    the memory), so this reconstructs it from the record; ``max_frames`` slices
    a PREFIX, so the index is valid on the record's own frame list.
    """
    ...

def rebuild_girth_band(extraction: ClipExtraction, assets: MorphAssets, observed: Mapping[str, bool]) -> ClipExtraction:
    """Re-measure this clip's girth band against the axes the fit OBSERVED.

    TASK A2 / spec 5.3.  ``extract_clip`` cannot know the verdict -- it is a
    property of the individual's clips seen together -- so the band it builds
    is the prior one and this replaces it once the verdict exists.  Returns a
    NEW extraction (the arrays are shared, not copied); with no observed axis
    it returns the one it was given, unchanged, so the prior path stays
    corner-for-corner what it shipped.
    """
    ...

def _bins_sensitivity(assets: MorphAssets, args: tuple, base: dict, girth: np.ndarray | None) -> dict[str, tuple[float, float]]:
    """Re-evaluate every quantity at half and double the profile's cell count.

    ``PROFILE_BINS`` is the one free constant of the volume construction, and
    since D4 settled the cross-section on an exact plane slice it is a
    QUADRATURE constant: refining it moves no envelope, so the band term is the
    integration error and nothing else.  It is kept in the band rather than
    assumed away, because a term that is small should be shown to be small.

    The factor is measured at ``PROFILE_BINS // 2`` and ``PROFILE_BINS * 2``;
    the full sweep is recorded in provenance.  Lengths do not move with the cell
    count at all and are dropped by the caller, so no length band inherits a
    spurious term.
    """
    ...

def _section_shape_sensitivity(assets: MorphAssets, args: tuple, base: dict, girth: np.ndarray | None) -> dict[str, tuple[float, float]]:
    """Re-evaluate every quantity under the alternative AREA rule.

    The cross-section is exact; its AREA is not a single-valued reading of it,
    because the fin gate leaves the contour open (13% of the section's perimeter
    is missing at the median station of the rest template) and a shark's section
    is not convex at the caudal keel.  The shipped rule is the contour's CONVEX
    HULL, which is an upper bound on the enclosed area; the alternative is the
    shoelace over the same points sorted by angle, which chords the gaps and the
    concavities and reads LOW.  The band is the interval between them --
    one-sided downward by construction, and worth -11.5% on the rest template's
    volume.

    The SUPERSEDED ``pi * a * b`` envelope is deliberately NOT an end of this
    band: it exceeds the hull, i.e. the upper bound, at 86% of the rest
    template's stations, so it is not a candidate value.  Its measured ratio
    travels in ``provenance.section_shape`` so a number quoted from an older
    record can still be placed.

    Perimeters do not move with the rule (they are Ramanujan of the same
    semi-axes) and come back at exactly 1.0, so the caller drops them and no
    girth row inherits a spurious term.
    """
    ...

def _section_shape_report(base_profile: dict) -> dict:
    """What the three area rules say about the reference frame, per station.

    Reported whether or not the band is wide, because the rule is a MODELLING
    choice inside an otherwise exact construction and the reader is entitled to
    see its size next to the quadrature term it dwarfs.
    """
    ...

def _bins_sweep(assets: MorphAssets, args: tuple, girth: np.ndarray | None) -> dict:
    """The reference frame's profile quantities at every cell count swept.

    Recorded so the reader can see that the volume now HAS a plateau: the
    station integral is flat in the cell count to under a percent, where the
    superseded vertex-binned envelope fell by a factor of 1.64 over the same
    sweep.  ``monotone_decreasing`` is kept as the shape statement it always
    was; on a converged integrand it is expected to be False.
    """
    ...

def _betas_sensitivity(assets: MorphAssets, args: tuple, base: dict, girth: np.ndarray | None) -> dict[str, float]:
    """RMS fractional change per +/-1 PCA standard deviation of shape.

    There is NO posterior over betas -- the fit returns a point estimate with no
    covariance, no Hessian and no sampling -- so this is a sensitivity band and
    the perturbation is printed with it.  Because the fitted coefficients sit at
    0.3-0.4% of one sd, the realised contribution today is ~0.02%: the band says
    what would happen if the shape pass ever moved, not what this animal's shape
    uncertainty is.
    """
    ...
TEMPLATE_VOLUME_MODEL_U3 = 0.8670925253682137
TEMPLATE_VOLUME_CONTOUR_MODEL_U3 = 0.7673249895508986
TEMPLATE_STATIONS = ((0.0, 0.2130805864410406, 0.13716409000543658, 1.1132883181585222), (0.16666666666666666, 0.38691317729134206, 0.3324085153095295, 2.2631553670401336), (0.3333333333333333, 0.3482211224492741, 0.17553661539746937, 1.690473583584823), (0.5, 0.3901073321652119, 0.338709565797666, 2.2924993443893875), (0.6666666666666666, 0.3286675681749153, 0.30034987826834536, 1.9771227116270604), (0.8333333333333334, 0.23435580651531554, 0.15379083337322877, 1.2326154352637897), (1.0, 0.1279418316997108, 0.05988674510809838, 0.6096117009504648))

def _template_station_check(profile: dict) -> dict:
    """The rest stations against the spec's normative table.

    The spec publishes the table this module produces, so this is a regression
    guard on the construction, not a reconciliation of two different ones: a
    non-zero deviation means the code and the spec have drifted apart.
    """
    ...

def template_reference(assets: MorphAssets) -> dict:
    """The asset's own morphometrics at rest, betas = 0, girth = 1.

    Reported so a reader can see what part of any clip's number is the template.
    """
    ...
