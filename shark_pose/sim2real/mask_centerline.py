"""Body-axis extraction from a binary silhouette mask, plus the canonical
body-bend metric used to score masks and fits identically.

There are three axis estimators (:data:`AXIS_CHOICES`, dispatched by
:func:`extract_axis`).  The one to use is :func:`extract_trunk_axis` -- see
"Three body axes" below; :func:`extract_centerline`, described first because
everything else is defined against it, is the historical one and is kept for
comparison.

Pipeline (``extract_centerline``, the historical axis)
------------------------------------------------------
1. Largest connected component, holes filled; Euclidean distance transform
   (kept at full resolution -- it supplies the widths).
2. Body axis = principal axis of the component's pixels (optionally weighted
   by ``distance_transform ** dt_power``).
3. The pixels are binned into ``n_bins`` slices along that axis; each slice
   contributes one centreline point at its mean perpendicular offset.
4. Light Savitzky-Golay smoothing, arc-length resampling to ``n_points``,
   widths = 2x the distance transform on the line, then orientation
   snout -> tail (the snout hint if there is one, else :func:`orientation_score`).

Why not thinning / medial axis (measured, 2026-09-04)
-----------------------------------------------------
A Zhang-Suen skeleton plus longest-geodesic-path was implemented first and is
*worse on real masks*: a shark's caudal fin forks, so the longest path runs up
whichever lobe happens to be longer in that frame and the tail chord flips
between lobes.  On the 150 demo SAM masks that gives Pearson r = 0.55 against
the reference bend series (r = 0.99 for the slice mean here), with p2p 67 deg
against a true 43 deg.  Averaging the two lobe branches at the last skeleton
junction did not fix it (r = 0.55, p2p 109 deg -- the "last junction" is often a
pelvic/anal fin).  The slice mean has no topology to get wrong: it averages the
two lobes by construction and is structurally immune to fin spurs.  (The
distance-transform path of :func:`extract_trunk_axis` is not a skeleton either:
it has no branches to choose between, and it is cut at the peduncle before the
fin can be reached at all.)

Known biases (measured on synthetic bodies with a known centreline):
* smoothing/binning shrink the reported bend by ~1.0 deg at a true 15 deg and
  ~2.2 deg at 30 deg (the straight axis under-reads strongly bent bodies);
* a large dorsal + pectoral fin blob biases the bend by -1.4 to -2.9 deg,
  because fin area pulls the slice mean off the medial axis.  ``dt_power > 0``
  down-weights thin fin blades (fin bias -0.4 deg at ``dt_power=4``) at the
  cost of agreement with the stored reference series (r 0.985 -> 0.89), so it
  defaults to 0.

Coordinate and sign conventions
-------------------------------
All 2D points are ``(x, y)`` image pixels with **y pointing DOWN** (the usual
image convention).  Polylines are ordered snout -> tail.

The bend metric compares two chords of the polyline:

* head chord: total-least-squares direction of the points covering the front
  40% of the arc length, signed to point snout -> tail;
* tail chord: the same over the back 40%.

``chord_bend_deg`` returns ``atan2(hx*ty - hy*tx, hx*tx + hy*ty)`` in degrees,
where ``h`` is the head chord direction and ``t`` the tail chord direction.
Because y points DOWN, a **positive** bend means the tail chord is rotated
**clockwise on screen** relative to the head chord.  Concretely: if the head
points along image ``+x`` (the animal faces right on screen), a positive bend
means the tail bends toward image ``+y``, i.e. downward on screen.

``tail_deflection_pct`` carries the same sign: the perpendicular offset of the
tail tip from the infinite head-chord line, positive on the side the head chord
rotates toward for a positive bend, as a percentage of the polyline arc length.

This is the convention of the stored reference series
``outputs/demo_2026-09-03/work/spine/demo_centerline_bend.json``; no sign flip
is applied (verified by ``scripts/extract_mask_centerline.py`` on the 150 demo
masks: r = 0.985, sign agreement 0.99 over the 144 frames with |bend| > 1 deg,
rmse 2.04 deg, median |bend| 10.4 deg vs the stored 10.2 deg).

What the metric is measured ON (measured, 2026-09-04)
-----------------------------------------------------
The mask centreline spans the whole animal, snout TIP to caudal fin TIP, but
the 13 v2 spine stations only cover arc :data:`RIG_ARC_SPAN` = 0.049-0.881 of
it.  Scored end to end the demo mask bends 10.4 deg median / 44.8 p2p; scored
on the spine's own span it bends 3.5 / 29.1 -- two thirds of the "mask bend"
was caudal fin and rostrum, neither of which the rig has.  So a fit must be
compared against a fin-excluded series:

* :func:`trim_to_span` cuts the centreline to that span ("trunk"), and
* :func:`station_bend` goes further and samples it at the stations' own arc
  fractions, which is exactly what the fitter projects -- one function, one set
  of samples for mask and fit alike.

On the demo the trunk and station series agree at r = 0.971 (medians 3.48 vs
3.63 deg, p2p 29.1 vs 30.7), while the full-body series correlates with them at
only r = 0.49 / 0.63: the fin dominates the untrimmed number.

That is the arithmetic for a snout-tip-to-caudal-tip centreline.  The
peduncle-terminated axis needs a different span, :data:`RIG_ARC_SPAN_PEDUNCLE`
= 0.058-1.044, and its own polyline is already almost all trunk: on the demo
its full and station series agree at r = 0.96.

Three body axes (measured, 2026-09-04)
--------------------------------------
:func:`extract_centerline` above is the historical axis, selected by
``axis="thinning"`` (the name is historical too -- see :data:`AXIS_CHOICES`).
It is a slice mean of the WHOLE silhouette, fins included, so on real masks it
climbs into the dorsal fin and runs out along the caudal upper lobe (look at
``outputs/demo_2026-09-03/work/trunk/demo_trunk_sheet.png``, where it is the
red line), and the bend series it produces is partly fin geometry.

:func:`extract_trunk_axis` (``axis="dt_path"``) replaces the slice mean with a
minimum-cost path through the distance transform's core -- cost
``1 / (DT + eps) ** power`` on the filled body, 8-connected, Dijkstra on a
downsampled grid -- from the snout to the far end of the body, cut at the
caudal peduncle.  Thin structures (fin blades) cost 1/DT^2 more to cross, so
the path stays on the thick medial core.  Measured against the slice mean on
the same masks with the same snout hints (150 demo / 65 clip-2 frames,
2026-09-04): over arc 0.2-0.6, the stretch that carries the dorsal fin, the
silhouette is 1.32x (demo) and 1.07x (clip 2) WIDER under the new axis than
under the old one -- the old line was standing off the core; the two lines part
company by 80 px (demo) and 57 px (clip 2) at the median; and
:func:`fin_contamination` falls 0.273 -> 0.203 and 0.266 -> 0.195.
That last number moves least because the two polylines do not cover the same
anatomy -- the new one ends at the peduncle, so its own thin taper is inside
the window while the old one's caudal fin is outside the trunk span -- which is
why the width ratio and the separation, not the contamination, are the numbers
that show the defect.  The new polyline spans snout tip -> PEDUNCLE (median arc
1478 px on the demo against the slice mean's 2020), so the span that maps the
rig's stations onto it is :data:`RIG_ARC_SPAN_PEDUNCLE`, not
:data:`RIG_ARC_SPAN`.

Removing the fins CHANGES THE SERIES, it does not merely clean it: the station
bend correlates with the old axis's at r 0.54 (demo) and 0.65 (clip 2), with
median |bend| 3.63 -> 4.24 deg and p2p 30.7 -> 18.4 deg on the demo, 8.92 ->
5.51 and 88.2 -> 77.8 on clip 2.  Any fit tuned against the old series has to
be re-measured against this one.

WHERE THE AXIS BEGINS is the head-end counterpart of the cut below, and it is
the silhouette's business, not the detector's.  The snout hint is a v5 keypoint
and lands anywhere from the rostrum to the gills, while the 13 station
fractions are fractions of the snout -> cut arc, so an anchor that wanders
slides all 13 exactly as a wandering cut does.  Measured 2026-09-04 on the
records this module used to ship, whose polyline started AT the snapped hint:
the silhouette still reached a median 20.9 px (demo) / 29.6 px (clip 2) forward
of the first sample, p90 100.1 px, max 307.4 px (demo frame 182, 24.5% of that
frame's arc), with adjacent-frame jumps up to 295.7 px -- an order of magnitude
more station-sliding than the cut jitter the same pass went to lengths to
remove.  ``extend_head`` walks the axis forward off the anchor onto the
outline (:func:`extract_trunk_axis`, step 3).  The residual is
``head_extent_px``, reported on every record: over the frames the gate keeps,
median 8.25 px on a 1478 px axis (p90 12.4, max 16.9, adjacent-frame step p90
6.6, max 12.1) on the demo, and median 3.74 px on a 835 px axis (p90 7.9, max
9.4, step p90 2.8) on clip 2 -- a smoothing shrink at the tip, not a wandering
anchor.  It moved the axis's first sample by a median 17.6 px (p90 99.5, max
309.8) on the demo and 33.2 px (p90 135.9, max 275.8) on clip 2, changed the
station-bend series by RMS 0.38 deg (demo, r 0.9963) and 0.99 deg over clip 2's
kept span (r 0.9844), and -- because the endpoint is now the animal's head --
made the end-of-frame test honest: 11 further demo records and 3 of clip 2's
are rejected as truncated, their rostrum having reached the border while the
detector-anchored endpoint sat clear of it -- 95.4-127.5 px inside on the demo,
against a 93.1 px margin, and 69.6-87.2 px on clip 2 against a 66.1 px one.
Moving the anchor onto the silhouette takes those same 14 endpoints to
10.0-92.6 px (demo) and 1.9-42.5 px (clip 2), i.e. inside the margin.

WHERE THE AXIS IS CUT is a second observation the series depends on, and it is
held still for the whole sequence rather than chosen per frame.  The 13
stations are fractions of the snout -> cut arc, so a cut that wanders slides
all 13 along the body and the chord bend moves with them.  Measured on the
current records (all figures over the 150 / 65 OK records unless said
otherwise): the per-frame width minimum puts the cut at arc 0.746 (demo) /
0.808 (clip 2) of the uncut path with a scatter of cv 6.82% / 6.77% and
adjacent-frame jumps of p90 7.0% / 11.2% (max 28.1% / 24.3%), and the resulting
series moves with it -- r(|d bend|, |d cut|) +0.016 / +0.463, r(bend, cut)
-0.288 / -0.322.  Cutting every frame at the sequence MEDIAN of that ratio
(:func:`trunk_cut_ratio`, :func:`extract_trunk_axis`'s ``cut_ratio``) leaves
cv 0.166% / 0.155% and r(|d bend|, |d cut|) -0.002 / -0.131.  It is a
1.21 deg (demo) / 4.38 deg (clip 2) RMS change to a 18.4 / 77.8 deg
series, correlating with the free one at r 0.966 / 0.970, and the
frame-to-frame step falls from 0.72 to 0.50 deg (demo) and from
1.46 to 1.04 deg (clip 2).

WHICH FRAMES SET THAT MEDIAN matters on a clip with truncation, and the pool is
the frames the gate will KEEP: the two border tests (:func:`edge_tests`) need
only the mask and the first pass's own polyline, so the extractor can ask them
before the cut exists.  Pooling over every extracted frame instead let clip 2's
32 truncated masks vote: on the pre-head-anchor records that put the applied
cut at 0.78926 against the kept frames' 0.82780, 3.85% of the path arc, worth
RMS 2.12 deg and p2p 14.15 -> 21.22 deg of station bend on the frames that are
actually fitted.  On the current records the two pools agree to 0.01% on clip 2
(0.80822 all, 0.80813 kept) and 0.11% on the demo (0.74586 / 0.74697), because
the head fix changed which frames are rejected; the guard stays because it is
the correct population and it is free.

:func:`extract_opening_axis` (``axis="opening"``) is the cross-check: open the
mask with a disc of 0.4x the trunk's median half-width to erase the fin blades,
then run the slice mean on what is left.  MEASURED ON THE DEMO, IT IS NOT A
VIABLE AXIS: the opening does cut fin contamination (0.102), but it also eats
the rostrum and the peduncle -- an opening cannot preserve a taper -- leaving a
median arc of 1292 px against 1569 for the path and 2020 for the slice mean,
and the residual blob at the dorsal fin's base still bends what is left, so its
station bend reads 28.1 deg median against 4.47.  It is a cross-check on the
path, not an alternative to it.

Which frames may be believed
----------------------------
:func:`centerline_quality` scores each record against the rest of its sequence
(frame-edge contact, arc length relative to the sequence median, robust z of
|station bend|, and how much of the line lies in fin-thin pixels) and sets
``ok_quality``.  ``ok`` only says the extraction ran; ``ok_quality`` says the
record cleared the quality gate at :data:`QUALITY_GATE_DEFAULTS` -- the same
thresholds the fitter's ``bend_series.gate`` uses, so the two verdicts agree as
measured; the fitter re-derives its flags from the record's fields rather than
reading ``ok_quality``, and its gate is ON by default
(``fit_video.DEFAULT_BEND_GATE["use_quality"]`` is True, as is
``bend_series.gate.use_quality`` in ``configs/inference/fit_video_v2.yaml``).
Re-measured 2026-09-05 on the shipped records, the two sides agree frame for
frame: this module keeps 128 of the demo's 150 and 30 of clip 2's 65, and so
does the fitter's gate.

It is not a "whole animal" test, and must not be read as one.  Its teeth used
to be a boolean edge test, and frame-edge CONTACT is not truncation: measured
2026-09-04 with that test, ``ok_quality`` was False on 95 of the demo clip's
150 records and on 31 of clip 2's 65 (four of clip 2's 31 also exceed
``abs_bend_z``; every other rejection on either clip was edge contact alone),
but the two clips mean opposite things by it.  On the demo the animal merely
grazes the right border while the silhouette stays complete: the fit run with
that gate ON tracked the 55 records it kept at r 0.999 but the 95 it never saw
at r -0.13, while the gate-OFF fit reproduces those same 95 at r 0.9986.  On
clip 2 the 22 contact frames at the end are the animal genuinely leaving the
shot, and their "bend" reaches 57 deg against 30.9 deg over the clean span.

Hence the edge test is now two graded numbers -- ``edge_near_end_px`` (how
close the polyline's OWN ENDS come to the border) and ``edge_contact_frac_span``
(border contact WITHIN the axis's own arc, as a fraction of the silhouette's
perimeter) -- and ``touches_edge`` survives as a flag that no longer rejects on
its own.  On the dt_path axis (2026-09-04, head anchored on the silhouette)
that passes 128 of the demo's 150 records against 55 under the boolean test,
and 30 of clip 2's 65, rejecting its frames 0-4 and 66-128 -- the stretch where
the animal leaves the shot, plus three frames at the start where the SAM mask
is half an animal.  The end test is what does the work on both clips: 22 of the
demo's 22 rejections and 31 of clip 2's 35 (of the other four, three exceed
``abs_bend_z_max`` -- those three start frames -- and one is both).  The
contact test rejects nothing on the demo, and one record on clip 2 that the end
test does not: frame 102, whose ``edge_contact_frac_span`` is 0.3601 while its
nearer axis end is 100.4 px inside the 66.1 px margin.  It exceeds
``abs_bend_z_max`` too (z 16.4), so contact is not the only thing holding it
out, but it is the one frame where contact is doing work of its own.

On an UNDISTORTED frame neither test means anything measured against the canvas
rectangle: ``alpha = 1`` undistortion maps the whole source image into the same
canvas and leaves black wedges along its edges, so the real border -- the image
of the SOURCE frame's edge -- is a curve inside the rectangle.  Both tests
therefore take an optional :class:`ValidRegion` and measure against its
boundary.  It is not a detail: on the demo's undistorted masks the rectangle
puts the nearer axis end 44-124 px further from "the border" than it is, and
sees 0-38 border-contact pixels on frames where the region sees 456-1493.
``None`` (a raw video frame) keeps the rectangle, bit for bit.

Both counts moved when the head anchor did (139 -> 128 on the demo, 33 -> 30 on
clip 2): the endpoint the test measures is now the animal's rostrum rather than
a detector keypoint 21 px behind it, so frames whose head reaches the border
are no longer passed.  The price is supervision: the demo's longest gap between
kept records grows from 3 records (6 frames, 0.10 s) to 20 (40 frames, 0.67 s
on a ~2.5 s beat), all of it one stretch where the animal swims out of the left
of frame.

Scoping the contact count to the axis's arc is what changed the demo from 81
to 128.  The unscoped count rejects 47 records whose trunk is whole and whose
nearer axis END is a median 320 px inside the 93 px margin -- i.e. the frame is
clipping the caudal FIN, which a snout -> peduncle axis does not travel.
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
__all__ = ['extract_centerline', 'extract_trunk_axis', 'extract_opening_axis', 'extract_axis', 'fin_contamination', 'peduncle_cut_index', 'polyline_arc_px', 'trunk_cut_ratio', 'cut_index_for_ratio', 'mask_perimeter_px', 'AXIS_CHOICES', 'RIG_ARC_SPAN_PEDUNCLE', 'chord_bend_deg', 'tail_deflection_pct', 'bend_metrics', 'bend_of_projected_stations', 'resample_polyline', 'orientation_score', 'trim_to_span', 'peduncle_index', 'station_arc_fractions', 'sample_polyline_at_fractions', 'station_bend', 'centerline_quality', 'mask_edge_contact_px', 'edge_tests', 'mask_edge_contact_span_px', 'polyline_edge_distance_px', 'ValidRegion', 'RIG_ARC_SPAN', 'measure_rig_arc_span', 'rig_arc_span', 'PEDUNCLE_SEARCH_STEP', 'ORIENT_BAND', 'QUALITY_GATE_DEFAULTS']
RIG_ARC_SPAN = (0.048889, 0.881144)
RIG_ARC_SPAN_PEDUNCLE = (0.057932, 1.044269)
PEDUNCLE_SEARCH_STEP = 0.005

def _section_extent_product(vertices: np.ndarray, faces: np.ndarray, y: float) -> float:
    """sqrt(lateral x dorsoventral) of the mesh's section at the plane ``y``.

    The FACES are intersected, not the vertices binned: whole trunk bins of
    this 3912-vertex mesh hold 3-17 vertices, so a vertex-binned extent
    measures where the artist put a loop rather than where the surface is.
    """
    ...

def measure_rig_arc_span(assets_dir) -> Dict[str, object]:
    """Measure both rig arc spans on a v2-style asset directory.

    :data:`RIG_ARC_SPAN` and :data:`RIG_ARC_SPAN_PEDUNCLE` are properties of
    the TEMPLATE MESH and the rig on it, not constants of the pipeline: they are
    where the first and last spine station sit along the body's length.  Editing
    the template -- ``scripts/reshape_template.py`` extends the rostrum, which
    moves both the snout tip and the cranium -- moves them, so the extractor and
    the fitter measure the asset they are using instead of pasting a number.

    The recipe is the one the two constants' docstrings record:

    * snout tip = the template's y max, tail tip = its y min;
    * cranium and ``caudal_axis_2`` = the first and last ``spine_joint_names``
      entry of ``bone_positions.npy``;
    * the peduncle = the minimum of sqrt(lateral x dorsoventral) over section
      planes between ``dorsal2_root`` and ``caudal_upper_root``, on a
      :data:`PEDUNCLE_SEARCH_STEP` grid with a parabolic refinement.  The window
      matters: searched further back the profile falls lower still, but that is
      inside the caudal fin, where the section is two thin blades.

    Args:
        assets_dir: directory holding ``template_vertices.npy``, ``faces.npy``,
            ``bone_positions.npy`` and ``skeleton.json``.

    Returns:
        ``{"span": (head, tail), "span_peduncle": (head, tail), "snout_y": ...,
        "tail_y": ..., "cranium_y": ..., "caudal_axis_2_y": ...,
        "peduncle_y": ..., "peduncle_extent": ...}``.
    """
    ...

def rig_arc_span(assets_dir=None, axis: str='thinning') -> Tuple[float, float]:
    """The arc span that goes with ``axis``, measured off ``assets_dir``.

    ``assets_dir=None`` returns the shipped PRE-RESHAPE constants, which is what
    a caller with no asset directory (a synthetic test, an old record) needs.
    """
    ...
SPAN_TAIL_MAX = 1.25
ORIENT_BAND = (0.15, 0.4)

def resample_polyline(points_xy: np.ndarray, n_points: int) -> np.ndarray:
    """Arc-length resample an ordered polyline to ``n_points`` samples."""
    ...

def _path_length(pts: np.ndarray) -> float:
    ...

def _smooth(points_xy: np.ndarray, window_frac: float=0.12) -> np.ndarray:
    """Savitzky-Golay smoothing of a polyline (quadratic, edge-preserving)."""
    ...

def _chord_direction(points_xy: np.ndarray) -> np.ndarray:
    """Total-least-squares direction of a point set, signed first -> last."""
    ...

def _chords(polyline: np.ndarray, front_frac: float, back_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...

def chord_bend_deg(polyline: np.ndarray, front_frac: float=0.4, back_frac: float=0.4) -> float:
    """Signed head-chord to tail-chord angle in degrees (image plane, y DOWN).

    Positive = the tail chord is rotated clockwise on screen relative to the
    head chord; with the head pointing along image +x the tail bends toward
    image +y.  ``polyline`` must be ordered snout -> tail and should be
    arc-length resampled first (see ``resample_polyline``).
    """
    ...

def tail_deflection_pct(polyline: np.ndarray, front_frac: float=0.4) -> float:
    """Signed perpendicular offset of the tail tip from the head-chord line,
    as a percentage of the polyline arc length.  Sign matches
    ``chord_bend_deg``."""
    ...

def bend_metrics(polyline: np.ndarray, front_frac: float=0.4, back_frac: float=0.4) -> Dict[str, float]:
    """Both bend numbers for one snout -> tail polyline."""
    ...

def bend_of_projected_stations(points_2d: np.ndarray, n_points: int=32, front_frac: float=0.4, back_frac: float=0.4) -> Dict[str, object]:
    """Apply the mask bend metric to projected 3D spine stations.

    ``points_2d`` is (K, 2) image-pixel stations ordered snout -> tail (the 13
    v2 spine stations, say).  They are arc-length resampled to ``n_points``
    first, so a fit and a mask centreline are scored by exactly the same
    measurement.  Returns ``bend_deg``, ``tail_defl_pct``, ``length_px`` and
    the resampled ``polyline``.
    """
    ...

def trim_to_span(polyline: np.ndarray, widths: Optional[np.ndarray]=None, head_frac: float=RIG_ARC_SPAN[0], tail_frac: float=RIG_ARC_SPAN[1], n_points: Optional[int]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Cut a snout -> tail polyline down to an arc-fraction span and resample.

    ``head_frac`` / ``tail_frac`` are fractions of the *input* polyline's arc
    length (0 = snout tip, 1 = tail tip).  The output is ``n_points`` samples
    spaced evenly in the input's arc length between those two fractions
    (default: as many samples as came in), i.e. the sub-curve is exact for a
    straight body and correct to the input's own polygonal resolution
    otherwise.  ``widths`` (same length as the polyline) is interpolated with
    the same parameterisation, so widths keep tracking their points.

    The default span is :data:`RIG_ARC_SPAN`, the piece of the animal the 13 v2
    spine stations cover -- so ``trim_to_span`` produces the fin-excluded
    ("trunk") centreline that is comparable with a fitted spine.  Everything
    outside it is rostrum at the front and caudal fin at the back.
    """
    ...

def peduncle_index(widths: np.ndarray, back_frac: float=0.3, tip_frac: float=0.05) -> int:
    """Index of the caudal peduncle: the width minimum in the back ``back_frac``.

    A shark's body tapers monotonically behind the trunk down to the caudal
    peduncle -- the narrow wrist just ahead of the tail fin -- and then widens
    again into the fin.  So the *minimum* of the width profile over the
    posterior ``back_frac`` of the arc length marks where the trunk ends and
    the caudal fin begins.  Its arc fraction is the per-frame, data-driven
    counterpart of the fixed ``RIG_ARC_SPAN[1]`` tail cut; it is reported for
    diagnosis rather than used as the default cut, because on a mask whose fin
    is folded edge-on the minimum can slide forward onto the anal-fin notch.

    The last ``tip_frac`` of the line is excluded from the search: widths come
    from a distance transform, which necessarily collapses toward zero at ANY
    silhouette tip, so the final samples are always the narrowest and would win
    the minimum on every frame (measured on the demo: without the exclusion
    a fifth of the frames report the peduncle at arc 1.0, the fin tip itself).

    Returns an index into ``widths`` (so ``index / (len(widths) - 1)`` is its
    arc fraction on an arc-length resampled polyline).
    """
    ...

def station_arc_fractions(spine_fraction: Sequence[float], span: Tuple[float, float]=RIG_ARC_SPAN) -> np.ndarray:
    """Map spine-relative station fractions onto the MASK centreline's arc.

    ``spine_fraction`` is the prior's table (``motion_prior.npz``): where each
    of the 13 stations sits along the SPINE, cranium = 0 to caudal_axis_2 = 1.
    The mask centreline instead spans the whole animal, snout tip to caudal fin
    tip, and the spine only covers ``span`` of it.  The mapping is affine,
    ``head + (tail - head) * spine_fraction`` -- identical to
    ``SpineSMPLify._station_arc_fractions``, so the fitter's midline term and
    this module sample the very same places on the centreline.

    ``span[1]`` may exceed 1 (up to :data:`SPAN_TAIL_MAX`): with the
    peduncle-terminated axis the rig's last station sits 4.4% of the trunk
    behind the end of the centreline, so :data:`RIG_ARC_SPAN_PEDUNCLE` ends at
    1.0443.  Fractions past 1 are clamped onto the last sample by
    :func:`sample_polyline_at_fractions` unless it is asked to extrapolate.
    """
    ...

def sample_polyline_at_fractions(polyline: np.ndarray, fractions: Sequence[float], extrapolate: bool=False) -> np.ndarray:
    """(K, 2) points at given arc-length fractions of an ordered polyline.

    Fractions outside [0, 1] are clamped onto the polyline's ends (numpy's
    ``interp`` behaviour).  With ``extrapolate=True`` they instead continue
    along the first/last segment's direction, which is what a station sitting
    past the end of the line wants: with :data:`RIG_ARC_SPAN_PEDUNCLE` the
    rig's last station is 4.4% of the trunk behind the mask's peduncle, and the
    body axis genuinely continues that way into the caudal fin.
    """
    ...
CHORD_STEP_PX = 0.5
CHORD_LEN_SCALE = 3.0
CHORD_MIN_LEN_PX = 60.0

def _bilinear_on(mask_bool: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of a binary mask at float coordinates; 0 outside."""
    ...

def _half_crossing(mask_bool: np.ndarray, p: np.ndarray, n: np.ndarray, t_in: float, t_out: float) -> float:
    """Where the bilinear mask crosses 0.5 between an inside and an outside t."""
    ...

def chord_width(mask: np.ndarray, point_xy: Sequence[float], normal_xy: Sequence[float], max_len: float, step: float=CHORD_STEP_PX, subpixel: bool=True) -> float:
    """Width of ``mask`` along the chord through ``point_xy``, in pixels.

    THE one width operator.  March along ``point +/- normal`` out to
    ``max_len`` on either side, take the CONTIGUOUS run of mask nearest the
    point (the run containing it when there is one, else the closest run), and
    return its length.  ``mask`` may be a boolean array or any array whose
    positive entries are the silhouette; ``normal_xy`` need not be unit.

    Both sides of a girth measurement have to be read with this same function
    or the ratio between them does not mean what it says -- which is what went
    wrong in the previous round, where the fitted target was ``2 x`` the
    distance transform (the INSCRIBED diameter, bounded by the nearer wall)
    while the score was a chord.  Measured at the 13 ``stations_xy`` the DT
    diameter is 0.925 of the chord on the demo and 0.968 on clip 2 (trunk
    stations 3-9, bend-gated records; 0.927 / 0.966 over all records --
    ``outputs/demo_2026-09-03/work/girth2/fix1/dt_over_chord.json``).  The
    0.888 / 0.981 pair quoted here before 2026-09-05 was that ratio at
    ``e2_score.py``'s stations, which sit 13-250 native px headward.

    The march is byte-for-byte the algorithm of
    ``outputs/demo_2026-09-03/work/shape_head/e2_score.py::_chord_width``
    (which should import this function): the same ``step``, the same
    nearest-neighbour lookup, the same run selection.  With ``subpixel=False``
    the return value is that function's, ``(run length in samples) * step``.
    With ``subpixel=True`` (the default) the two ENDPOINTS are refined by
    linear interpolation of the bilinearly sampled mask to its 0.5 level, which
    removes the +/- ``step`` quantisation without changing which run is
    measured.  Measured on the shipped records (40 frames x 13 stations each):
    the refinement is unbiased -- mean +0.002% (demo) and +0.000% (clip 2) --
    and moves any single reading by at most 0.56% and 0.92%, so it does not
    open a gap between this and a caller still marching at ``subpixel=False``.

    Returns 0.0 when the ray never touches the mask, when the normal is
    degenerate, or when the point is not finite.
    """
    ...

def chord_widths_along_polyline(mask: np.ndarray, points_xy: Sequence[Sequence[float]], widths_px: Optional[Sequence[float]]=None, step: float=CHORD_STEP_PX, subpixel: bool=True, len_scale: float=CHORD_LEN_SCALE, min_len_px: float=CHORD_MIN_LEN_PX) -> np.ndarray:
    """(P,) :func:`chord_width` at every point of an ordered polyline.

    The chord's direction is the normal to the LOCAL axis, taken as the central
    difference ``points[i + 1] - points[i - 1]`` (one-sided at the ends) --
    the same rule the girth scorer uses on the 13 stations.  ``widths_px``, when
    given, is the local distance-transform width and only sets how far the
    march looks (``max(len_scale * w, min_len_px)``); it never enters the answer.

    A point whose local direction is degenerate gets ``nan``, not 0: "the
    operator could not measure here" is not "the body is zero wide here".
    """
    ...

def polyline_span_length_px(points_xy: Sequence[Sequence[float]], head_frac: float, tail_frac: float) -> float:
    """Arc length (px) of the sub-curve between two arc fractions of a polyline.

    Fractions are of the polyline's own arc length, the parameterisation
    :func:`sample_polyline_at_fractions` samples in, so the answer is exactly
    ``(tail - head) * total`` -- the two endpoints are the very points that
    function returns.  Fractions are clamped to ``[0, 1]``: a station sitting
    past the end of a peduncle-terminated axis has no mask arc behind it to
    measure, and extrapolating one would be inventing the animal's length.
    """
    ...

def station_bend(polyline: np.ndarray, spine_fraction: Sequence[float], span: Tuple[float, float]=RIG_ARC_SPAN, n_points: int=32, front_frac: float=0.4, back_frac: float=0.4, extrapolate: bool=False) -> Dict[str, object]:
    """Bend of a mask centreline sampled at the rig's 13 station fractions.

    The centreline is sampled at :func:`station_arc_fractions`, giving the same
    13 points a fitted spine projects to, and those points then go through
    :func:`bend_of_projected_stations`.  Mask and fit are therefore scored by
    one function on one set of samples: any difference in the number is body
    curvature, not a difference of measurement.

    Returns ``bend_deg``, ``tail_defl_pct``, ``length_px``, the resampled
    ``polyline`` and the 13 ``stations`` that were sampled.
    """
    ...

def orientation_score(widths: Sequence[float], band: Tuple[float, float]=ORIENT_BAND) -> float:
    """Signed evidence that a polyline's FIRST end is the snout.

    ``band`` is an arc-length window measured inward from each end; the score
    is ``mean(width in the front band) - mean(width in the mirrored back
    band)``, in the same units as ``widths`` (pixels).  Positive means the
    front of the polyline is the thicker end, i.e. head-first.

    Why an *inner* band and not the outer tip (measured, 2026-09-04): the outer
    15% of a shark mask is the rostrum at one end and the CAUDAL FIN at the
    other, and a spread caudal fin is wider than a pointed snout, so the outer
    comparison votes tail-first whenever the fin is fanned toward the camera --
    on the 150 demo SAM masks it picks the wrong end on 17.  The 15-40% band
    compares branchial trunk against caudal peduncle instead: the widest part
    of the body against the narrowest, which is the largest and most reliable
    width contrast a shark silhouette offers.

    WHICH PROFILE: that measurement was made on the snout-tip -> caudal-TIP
    ``thinning`` axis, and the band is only meaningful on a profile spanning
    the whole animal, because it is the MIRRORED window (0.60-0.85) that has to
    land on the peduncle.  On the snout -> PEDUNCLE ``dt_path`` axis the
    mirrored window is mid-body and the contrast collapses -- measured
    2026-09-05 on the shipped fix2 records, the rule then votes head-first on
    142 of 150 demo frames but only 27 of 65 clip-2 ones, a minority.  No band
    rescues it: over 44 (lo, hi) pairs on a 0.05 grid the best clip-2 score is
    32 of 65 at (0.15, 0.25).  Callers holding both profiles must therefore
    score this on ``widths_thinning`` (150/150 and 61/65 on the same records);
    ``extract_mask_centerline.sequence_orientation`` does.
    """
    ...

class ValidRegion:
    """The part of an image plane that carries real pixels.

    A raw video frame is valid everywhere, and the frame border is both the
    edge of the canvas and the edge of the observation -- which is why every
    border test below is written against the first/last ``edge_margin_px`` rows
    and columns.  An UNDISTORTED frame is not: with
    ``cv2.getOptimalNewCameraMatrix(..., alpha=1.0)`` the whole source image is
    mapped into the same ``W x H`` canvas, and a barrel lens leaves black
    wedges in the middle of each canvas edge where no source pixel lands.  The
    curved boundary between wedge and image IS the real frame border there; the
    straight canvas edge, where it runs through a wedge, is nothing at all.

    So the tests take an optional valid region -- ``True`` where a source pixel
    landed -- and measure against ITS boundary.  ``valid`` is a mask of the
    same shape as the frames, usually
    ``camera.undistort_image(np.ones((H, W), np.uint8), INTER_NEAREST) > 0``.

    The two derived quantities are cached, because the region is a property of
    the camera and not of the frame: the border band (per margin width) and the
    distance-to-boundary field.  ``None`` in place of a ``ValidRegion``
    everywhere below means "the whole rectangle is valid", and takes the
    original code path bit for bit.
    """

    def __init__(self, valid: np.ndarray, source: str='') -> None:
        ...

    @classmethod
    def coerce(cls, obj) -> Optional['ValidRegion']:
        """Accept a :class:`ValidRegion`, a bare boolean array, or ``None``."""
        ...

    @property
    def shape(self) -> Tuple[int, int]:
        ...

    @property
    def frac(self) -> float:
        """Fraction of the canvas that carries a source pixel."""
        ...

    def border_band(self, edge_margin_px: int=2) -> np.ndarray:
        """Pixels within ``edge_margin_px`` of the region's boundary.

        The INVALID set dilated by ``k``, with the array border counted as
        invalid, so a region that reaches the canvas edge is bordered there
        too.  The invalid pixels themselves are therefore IN the band -- it is
        not restricted to valid ones; a caller that wants only the valid rim
        intersects with ``self.valid``.  For a fully valid region it is exactly
        the first/last ``k`` rows and columns -- the band
        :func:`mask_edge_contact_px` uses without a region.
        """
        ...

    def distance_field(self) -> np.ndarray:
        """Distance in px from each pixel to the nearest INVALID pixel, minus 1.

        The offset makes it agree with the convention of
        :func:`polyline_edge_distance_px`, where a point sitting on the first
        valid row scores 0.0 rather than 1.0.  For a fully valid rectangle the
        field is exactly ``min(x, y, W - 1 - x, H - 1 - y)``.
        """
        ...

    def distance_at(self, points_xy: Sequence[Sequence[float]]) -> np.ndarray:
        """The distance field sampled at (float) image points, nearest pixel.

        Points outside the canvas score their distance to the canvas edge as a
        negative number, so a polyline end that has left the frame entirely is
        still ordered correctly against one that is merely close to it.
        """
        ...

    def __repr__(self) -> str:
        ...

def _border_band(shape: Tuple[int, int], edge_margin_px: int, valid_region: Optional['ValidRegion']=None) -> np.ndarray:
    """The border band a contact test counts against, region-aware."""
    ...

def mask_edge_contact_px(mask_bool: np.ndarray, edge_margin_px: int=2, valid_region: Optional['ValidRegion']=None) -> int:
    """Number of mask pixels lying within ``edge_margin_px`` of any image border.

    The border band is the union of the first/last ``edge_margin_px`` rows and
    columns, so a pixel in a corner is counted once.  ``edge_margin_px`` is
    clamped to at least 1 (a margin of 0 would make the test vacuous).

    With a ``valid_region`` (an undistorted frame; see :class:`ValidRegion`) the
    band follows that region's boundary instead of the canvas rectangle.
    """
    ...

def mask_edge_contact_span_px(mask_bool: np.ndarray, polyline: Sequence[Sequence[float]], edge_margin_px: int=2, valid_region: Optional['ValidRegion']=None) -> int:
    """Border-contact pixels that lie WITHIN the body axis's own arc.

    :func:`mask_edge_contact_px` counts every silhouette pixel on the frame
    border, whatever part of the animal it belongs to.  The ``dt_path`` axis
    spans snout tip -> peduncle, so a border clipping the CAUDAL FIN removes
    outline the axis never travels and cannot corrupt the bend series measured
    on it.  This count keeps only the contact between the two end planes: the
    plane through the snout endpoint normal to the leading chord, and the one
    through the tail endpoint normal to the trailing chord (each chord the last
    10% of the polyline, so a single noisy end segment cannot tilt it).

    Measured on the demo clip: the global count rejects 47 of 150 records at
    ``edge_contact_frac_max`` 0.10 whose trunk is whole and whose nearer axis
    END sits a median 320 px inside a 93 px margin.  (On the pre-head-anchor
    axis of 2026-09-04 the same test rejected 48, and in all 48 the contact sat
    at the caudal end -- a median 0.16 trunk-lengths behind the axis's tail
    endpoint against 1.11 ahead of its snout.  That decomposition was not
    re-run after the head anchor moved.)
    """
    ...

def mask_perimeter_px(mask_bool: np.ndarray) -> int:
    """Number of boundary pixels of a mask (pixels with a 4-neighbour outside).

    Used as the denominator of ``edge_contact_frac``: what fraction of the
    silhouette's outline is the image border rather than the animal.
    """
    ...

def fin_contamination(widths: Sequence[float], span: Optional[Tuple[float, float]]=None, ratio: float=0.5, grid: int=128) -> float:
    """Fraction of a centreline's samples that lie in fin-thin pixels.

    ``widths`` is the polyline's own width profile (2x the distance transform,
    as every extractor here returns).  The reference is the MEDIAN width over
    the trunk -- ``span`` of the arc, tail clamped to 1, or the whole polyline
    when ``span`` is None -- and a sample counts as contaminated when its width
    is below ``ratio`` of it.  Widths are twice the distance transform, so the
    factor of 2 cancels and this is exactly the contract's "DT below 0.5x the
    trunk's median DT".

    A body axis that stays on the medial core scores near 0 over the trunk and
    only picks up samples at the rostrum and (if the polyline runs that far) in
    the caudal fin; an axis that climbs into the dorsal fin or up a caudal lobe
    scores high, because a fin blade's inscribed circle is small.  Medians over
    the 150 demo masks: 0.273 for the slice-mean axis (0.164 over its trunk
    span alone), 0.203 for the distance-transform path (0.156 over its trunk);
    on clip 2, 0.266 (0.133) against 0.195 (0.148).  The two axes' trunks are
    different pieces of the animal -- see the module docstring -- so it is the
    whole-line numbers that compare like with like.  The path's own figure is
    not zero and cannot be: its last stretch is the peduncle, which is thin.

    The width profile is first resampled to ``grid`` evenly spaced samples, so
    the number is a fraction of ARC LENGTH rather than of the polyline's own
    ``n_points`` (at the usual 32 samples one sample would already be 3%).
    """
    ...

def polyline_arc_px(points_xy: Sequence[Sequence[float]]) -> np.ndarray:
    """Cumulative arc length (px) of an ordered polyline, ``[0, ..., L]``."""
    ...

def trunk_cut_ratio(polyline_full: Sequence[Sequence[float]], cut_index: int) -> float:
    """Where the peduncle cut fell, as a fraction of the UNCUT path's arc.

    ``arc(snout -> cut) / arc(snout -> far end)``, both measured on the same
    curve so the path's own length noise divides out.  This is the number that
    has to be stable for the 13 station fractions to mean the same anatomy from
    frame to frame: the stations are fractions of the snout -> cut arc, so a cut
    that moves slides every station along the body.  Fixing it to the sequence
    median is what :func:`extract_trunk_axis`'s ``cut_ratio`` does.

    Why the whole path is the ruler (measured 2026-09-04, 150 demo + 65 clip-2
    dense paths).  Four rulers were compared by how tightly they hold the cut
    and by how much of the station-bend series moves with it:

    ==========================  ===========  ===========  ==============
    ruler for the cut           cut cv demo  cv clip 2    r(|d bend|,|d cut|)
    ==========================  ===========  ===========  ==============
    free (per-frame minimum)          6.96%        7.45%   +0.412 / +0.556
    snout -> girth peak              15.0%       118.4%   (unusable)
    width level at the cut           14.7%         8.87%   +0.865 / +0.469
    snout -> far end (this one)       0.14%        0.16%   -0.001 / +0.128
    ==========================  ===========  ===========  ==============

    The girth peak -- the widest inscribed circle, nominally the pectoral
    girdle -- is a broad, multi-modal maximum of the width profile; on clip 2 it
    lands on the path's FIRST sample on one frame (zero arc, no ruler at all)
    and its ratio scatters by 118%.  A fixed
    width LEVEL is scale-free but reads the noisiest part of the profile.  The
    far end (the caudal fin tip, the one landmark past the peduncle that every
    frame has) leaves 0.15% of jitter.

    PROVENANCE of that table: it is the ruler comparison of 2026-09-04, made on
    the axis as it stood then -- anchored on the detector's snout keypoint,
    before the head end was moved onto the silhouette.  All four cv figures are
    over the OK records (150 demo / 65 clip 2).  The ranking is a property of
    the rulers, not of the anchor, and was not re-run; the two rows that this
    module still measures every run did move when the head anchor did, and on
    the CURRENT records (see the module docstring) they read:

    * free cut, cv 6.82% (demo, 150 ok records; 6.78% over the 128 the quality
      gate keeps) and 6.77% (clip 2, 65 ok; 7.71% over its 30 kept), with
      r(|d bend|, |d cut|) +0.016 / +0.463 over the ok records;
    * fixed at the sequence median, cv 0.166% / 0.155% (ok) and 0.165% /
      0.162% (kept), with r(|d bend|, |d cut|) -0.002 / -0.131.
    """
    ...

def cut_index_for_ratio(polyline_full: Sequence[Sequence[float]], ratio: float) -> int:
    """Inverse of :func:`trunk_cut_ratio`: the path index at a given cut ratio.

    Clamped to the path, so a ratio at or past 1 cuts at the far end.
    """
    ...

def peduncle_cut_index(widths: Sequence[float], hint_frac: Optional[float]=None, back_frac: float=0.45, tip_frac: float=0.05, window_frac: float=0.05, waist_ratio: float=0.5) -> Tuple[int, str]:
    """Where to cut a snout -> caudal-tip axis so that it ends at the peduncle.

    Returns ``(index, source)``.  ``source`` is ``"peduncle"`` when a width
    minimum chose the cut, ``"caudal_notch_hint"`` when a detector hint chose
    between several, and ``"far_end"`` when there was no minimum to cut at (the
    line is returned whole).

    The rule is a LOCAL minimum, not the smallest width in a window, because
    the width profile along a path that runs out to the caudal fin tip is not
    monotonic: it falls to the peduncle, rises again inside the fin's fan, then
    collapses at the fin tip.  Measured on demo frame 80 (widths in px along a
    256-sample path): ... 120, 100, 86 at arc 0.69, 98, 206, 276 at arc 0.75,
    then down to 60.  A plain minimum over the back of the line lands at the
    fin tip whenever the fin is edge-on (demo frame 0: arc 0.95 instead of
    0.80), so the tip is excluded AND the candidate must be a minimum of its
    own +-``window_frac`` neighbourhood.

    When several minima qualify (a pelvic or anal notch can make one), the
    ``hint_frac`` -- the arc position of the v5 ``caudal_notch`` keypoint
    projected onto the line -- picks the nearest.  That is how the hint is
    used, rather than as the path's target: measured over 15 demo frames the
    notch keypoint sits 28 px (median) OFF the axis and 0.059 of arc BEHIND the
    width minimum, because it is a dorsal notch on the fin's leading edge, so
    steering the path to it would end the line off the midline and inside the
    fin.

    When there is NO local minimum -- a caudal fin seen edge-on, or a forked
    fin whose lobes are thin blades, where the width simply keeps falling past
    the peduncle -- the fallback is the first place in the window where the
    width drops below ``waist_ratio`` of the trunk's median (the trunk being
    everything ahead of the window).  Without that fallback the line runs all
    the way out along a lobe: on a synthetic body with a forked heterocercal
    tail it ends 85 px off the true midline instead of 2 px.
    """
    ...

def polyline_edge_distance_px(polyline: Sequence[Sequence[float]], height: int, width: int, valid_region: Optional['ValidRegion']=None) -> float:
    """Distance from the NEARER END of a polyline to the image border, in px.

    ``NaN`` when there is no polyline to measure (an old record, or a failed
    extraction), which is how :func:`centerline_quality` makes the test fail
    open rather than reject records that cannot answer it.

    With a ``valid_region`` the border is that region's boundary -- for an
    undistorted frame, the curved image of the source frame's edge -- rather
    than the canvas rectangle, which there runs through pixels no camera ray
    ever reached.
    """
    ...

def edge_tests(mask: np.ndarray, polyline: Optional[Sequence[Sequence[float]]]=None, edge_margin_px: int=2, thresholds: Optional[Dict[str, float]]=None, valid_region: Optional['ValidRegion']=None) -> Dict[str, object]:
    """The two frame-border tests, on one mask and one polyline.

    Both are the truncation evidence :func:`centerline_quality` gates on, and
    they are their own function because they need nothing but the mask and the
    axis -- no sequence statistics, no bend series -- so the extractor can ask
    them in its FIRST pass, before the sequence-median peduncle cut exists (see
    ``scripts/extract_mask_centerline.py``'s ``measure_cut_ratio``: pooling the
    cut over frames the gate then throws away lets a truncated silhouette set
    the cut for the frames that are kept).

    ``polyline`` may be ``None`` (or shorter than two points), which is how a
    failed extraction is passed in: ``edge_near_end_px`` is then NaN, the
    contact count falls back to the whole silhouette, and ``edge_ok`` reports
    only what could be measured.

    ``valid_region`` (:class:`ValidRegion`) is what makes these tests mean the
    same thing on an UNDISTORTED frame: ``alpha = 1`` undistortion leaves black
    wedges along the canvas edges, so the straight rectangle is no longer the
    boundary of the observation and both tests must be taken against the valid
    region's own curved boundary.  ``None`` keeps the rectangle.

    Returns ``touches_edge``, ``edge_contact_px``, ``edge_contact_frac``,
    ``edge_contact_px_span``, ``edge_contact_frac_span``, ``edge_near_end_px``,
    ``edge_end_margin_px``, ``perimeter_px`` and ``edge_ok``.
    """
    ...

def centerline_quality(mask: np.ndarray, record: Dict[str, object], seq_median_length: float, seq_bend_median: float, seq_bend_mad: float, edge_margin_px: int=2, thresholds: Optional[Dict[str, float]]=None, valid_region: Optional['ValidRegion']=None) -> Dict[str, object]:
    """Decide whether one centreline record is trustworthy enough to fit against.

    The bend-series term in the fitter believes every record it is handed, so a
    truncated silhouette -- the animal swimming out of frame -- shows up as a
    large spurious "bend" and drags the pose with it (measured on clip 2: over
    the stretch where the mask is a part-animal the mask bend reaches 57 deg and
    the fitted joint velocity rises 0.101 -> 0.125).  Three cheap, independent
    symptoms catch that case:

    ``edge_near_end_px``
        distance from the NEARER END of the record's polyline to the image
        border.  Truncation cuts the animal off at an end, so the axis runs
        into the border; grazing leaves the ends deep inside the frame.  This
        is the primary test.  Its threshold is ``edge_end_margin_frac`` of the
        image DIAGONAL (3% = 93 px on the 2704x1520 demo).
    ``edge_contact_frac`` / ``edge_contact_frac_span``
        mask pixels within ``edge_margin_px`` of a border, over the
        silhouette's perimeter (:func:`mask_perimeter_px`).  Truncation
        replaces a long stretch of outline with frame border; grazing touches
        it over a few pixels.  ``touches_edge`` (any contact at all) is still
        reported, but it no longer rejects on its own -- on the demo clip it is
        True on 95 of 150 whole-animal frames.  What GATES is the ``_span``
        number: contact restricted to the stretch of silhouette the axis
        actually spans (:func:`mask_edge_contact_span_px`), because a border
        clipping the caudal fin removes outline a snout -> peduncle axis never
        travels.  The unscoped fraction is still reported.
    ``fin_contamination``
        fraction of the polyline's samples thinner than half the trunk's median
        width (:func:`fin_contamination`).  Reported, not gated: it separates
        body axes from fin-climbing ones, which is a property of the extractor
        rather than of the frame.
    ``length_rel``
        the record's ``length_px`` over the sequence median.  A shark's arc
        length is nearly constant under swimming, so a short line means a
        partial mask and a long one a mask that grew a boat/diver/second animal.
    ``abs_bend_z``
        robust z-score of ``|stations_bend_deg|`` against the sequence median
        and MAD (MAD scaled by 1.4826, the normal-consistency factor).  Signed,
        and only the POSITIVE tail is a fault: an unusually straight frame is
        fine, an unusually bent one is usually a broken outline.

    Parameters
    ----------
    mask : (H, W) array
        The silhouette the record was extracted from (bool or 0/255).
    record : dict
        One extractor record; ``length_px``, ``stations_bend_deg`` and ``ok``
        are read if present (an older record without ``stations_bend_deg`` is
        accepted and simply scores 0 deg of bend).
    seq_median_length, seq_bend_median, seq_bend_mad : float
        Sequence statistics over the ``ok`` records: median ``length_px``,
        and the median / MAD of ``|stations_bend_deg|``.
    edge_margin_px : int
        Border band width for the edge test.
    thresholds : dict, optional
        Overrides for ``length_rel_min`` / ``length_rel_max`` /
        ``abs_bend_z_max`` / ``edge_end_margin_frac`` /
        ``edge_contact_frac_max`` (defaults :data:`QUALITY_GATE_DEFAULTS`).
        ``edge_end_margin_px`` may be given instead of the frac, in pixels.
    valid_region : ValidRegion, optional
        Where the frame carries real pixels, for masks that have been
        undistorted (see :class:`ValidRegion` and :func:`edge_tests`).  ``None``
        means the whole rectangle, i.e. a raw video frame.

    Returns
    -------
    dict with the border block of :func:`edge_tests` -- ``touches_edge``
    (bool), ``edge_contact_px`` (int),
    ``edge_contact_frac``, ``edge_contact_px_span``, ``edge_contact_frac_span``
    (the gated one), ``edge_near_end_px``, ``edge_end_margin_px``,
    ``perimeter_px``, ``fin_contamination``, ``area_frac`` (mask area / image
    area), ``length_rel``, ``abs_bend_z`` and ``ok_quality``.

    Notes
    -----
    A degenerate sequence (MAD 0, or a median length of 0) cannot support the
    statistic that rests on it: ``abs_bend_z`` is then reported as 0.0 and
    ``length_rel`` as NaN, so the bend test never fires and the length test
    always does -- the gate fails closed on length, open on bend.  The same
    goes for a record with no polyline: ``edge_near_end_px`` is NaN and that
    test does not fire.
    """
    ...

def _empty_result(n_points: int) -> Dict[str, object]:
    ...

def _slice_means(coords_xy: np.ndarray, weights: np.ndarray, axis: np.ndarray, origin: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean perpendicular offset per slice along ``axis`` -> centreline points."""
    ...

def extract_centerline(mask_bool: np.ndarray, n_points: int=32, snout_hint_xy: Optional[Sequence[float]]=None, n_bins: int=48, dt_power: float=0.0, smooth_frac: float=0.12, work_max_dim: int=512, min_pixels: int=50) -> Dict[str, object]:
    """Extract an ordered snout -> tail centreline from a silhouette mask.

    Parameters
    ----------
    mask_bool : (H, W) array
        Boolean (or 0/255) silhouette.
    n_points : int
        Number of arc-length resampled polyline samples.
    snout_hint_xy : (2,) or None
        Approximate snout position in mask pixels.  The polyline end nearest
        this point becomes the first sample.  If None, the end with the thicker
        *inner* band is taken as the snout -- see :func:`orientation_score`,
        which compares arc 15-40% from each end (branchial trunk vs caudal
        peduncle) rather than the outer tips (rostrum vs caudal fin).  A
        sequence should not rely on this per-frame fallback alone; decide the
        orientation once for the whole clip (``scripts/extract_mask_centerline.py``
        chains the frames and votes).
    n_bins : int
        Slices along the body axis.  More bins track curvature better (bend
        under-read 1.0 -> 0.9 deg at a true 15 deg going 32 -> 48) but are
        noisier; 48 was chosen on the demo masks.
    dt_power : float
        Weight each pixel by ``distance_transform ** dt_power`` when estimating
        the axis and the slice means.  0 (default) = plain area mean, which is
        what the stored reference series used; >0 suppresses fin-blade bias.
    smooth_frac : float
        Savitzky-Golay window as a fraction of the polyline length.
    work_max_dim : int
        The centreline is estimated on a copy of the mask downscaled so its
        longest side is at most this (a 2704x1520 mask drops from ~0.9 M to
        ~33 k pixels, ~25x faster) and mapped back to input-mask pixels.
        Widths always come from the full-resolution distance transform.
    min_pixels : int
        Masks smaller than this return ``ok = False``.

    Returns
    -------
    dict with ``polyline`` (n_points, 2) float pixels ordered snout -> tail,
    ``widths`` (n_points,) = 2x the distance transform along the midline,
    ``length_px``, ``bend_deg``, ``tail_defl_pct``, ``orient_source``
    ("hint" or "width"), ``orient_score`` (:func:`orientation_score` of the
    returned widths -- negative means the hint overruled the width evidence)
    and ``ok``.
    """
    ...

def _body_and_dt(mask_bool: np.ndarray, min_pixels: int):
    """Filled largest component + its full-resolution distance transform."""
    ...

def _downscale(body: np.ndarray, downsample: int):
    """Largest component of the mask on a 1/``downsample`` grid + the scales."""
    ...

def _grid_graphs(mask: np.ndarray, cost: np.ndarray):
    """Sparse 8-connected graphs over the mask pixels: cost-weighted and metric.

    Edge weight is the mean of the two endpoints' costs times the step length
    (1 or sqrt 2), i.e. a trapezoidal line integral of the cost field.  Two
    graphs are built from the same neighbour arrays: the cost one for the path
    itself, the metric one (weight = step) to find the far end of the body.

    ``scipy.sparse.csgraph.dijkstra`` is used rather than a Python heap:
    measured on demo frame 80 at downsample 4 (56 547 nodes) the sparse build
    plus C Dijkstra is 11.8 ms against 252 ms for an equivalent ``heapq``
    implementation (at downsample 2, 226 455 nodes: 54.7 ms against 1096 ms).
    Whole-frame cost is dominated by the full-resolution distance transform,
    not by the graph: 198 ms per 2704x1520 demo mask for the path against
    184 ms for the slice mean.
    """
    ...

def _snap_to_mask(point_xy, nearest_idx: np.ndarray, shape) -> Tuple[int, int]:
    """Nearest in-mask pixel (row, col) to a point, in the small grid's frame."""
    ...

def _to_small(point_xy, scale_x: float, scale_y: float) -> np.ndarray:
    ...

def _to_full(points_xy: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    ...

def _forward_direction(points_full: np.ndarray, head_frac: float) -> Optional[np.ndarray]:
    """Unit vector pointing FORWARD (tail -> snout) along a path's head end.

    ``points_full`` is the path in full-resolution image pixels, ordered from
    the anchor towards the tail; the direction is the total-least-squares
    chord of its leading ``head_frac`` of samples, signed to point away from
    the tail.  ``None`` when the path is too short to have a direction.
    """
    ...

def _forward_anchor_node(coords_full: np.ndarray, d_metric: np.ndarray, reach: np.ndarray, origin_full: np.ndarray, forward: np.ndarray, tail_metric: float, search_frac: float) -> Optional[int]:
    """Graph node reaching furthest forward of ``origin_full`` along ``forward``.

    Candidates are restricted to nodes within ``search_frac`` of the geodesic
    distance to the tail, so the search cannot jump to a caudal lobe that a
    bent animal has swung past its own snout.  ``None`` when nothing lies
    forward of the origin (the anchor is already the tip).
    """
    ...

def _silhouette_tip(body: np.ndarray, start_xy, forward: np.ndarray, radius: float) -> Optional[np.ndarray]:
    """Full-resolution body pixel furthest along ``forward`` near ``start_xy``.

    The graph the path is found on is downsampled, so its forward-most node is
    within about one coarse cell of the silhouette's actual tip; this recovers
    the remaining sub-cell distance at full resolution.  The window is a local
    one (``radius`` pixels) precisely so that it cannot wander onto another
    part of the animal.
    """
    ...

def _forward_reach_px(body: np.ndarray, poly: np.ndarray, head_frac: float, radius_frac: float=0.25) -> float:
    """How far the silhouette reaches FORWARD of ``poly[0]``, in pixels.

    The residual of the head anchor: the maximum projection, onto the axis's
    own forward chord, of the body pixels within ``radius_frac`` of the axis
    arc from its snout endpoint (a local window, so a bent animal's tail
    cannot answer for its head).  0 when the endpoint is the forward-most
    pixel there.
    """
    ...

def _axis_extras(poly: np.ndarray, widths: np.ndarray, axis: str, tail_source: str, span: Optional[Tuple[float, float]], snout_source: str='none') -> Dict[str, object]:
    """The contract's provenance block for one extracted axis."""
    ...

def extract_trunk_axis(mask_bool: np.ndarray, snout_hint_xy: Optional[Sequence[float]]=None, tail_hint_xy: Optional[Sequence[float]]=None, n_points: int=32, downsample: int=4, power: float=2.0, dense_points: int=256, smooth_frac: float=0.12, min_pixels: int=50, back_frac: float=0.45, tip_frac: float=0.05, eps: float=0.001, span: Tuple[float, float]=RIG_ARC_SPAN_PEDUNCLE, cut_ratio: Optional[float]=None, extend_head: bool=True, head_frac: float=0.15, head_search_frac: float=0.35) -> Dict[str, object]:
    """Snout -> peduncle body axis as a minimum-cost path through the DT core.

    The silhouette's medial axis is not the body axis: a shark's dorsal fin and
    caudal lobes are part of the same connected region, and every estimator
    that treats all mask pixels alike (a thinned skeleton, or the principal-axis
    slice mean of :func:`extract_centerline`) is pulled into them.  Here the
    filled body is turned into a cost field ``1 / (DT + eps) ** power`` and the
    axis is the cheapest path across it, so crossing a fin blade -- where the
    distance transform is small -- costs orders of magnitude more than staying
    on the thick core.  Steps:

    1. largest connected component, holes filled, full-resolution distance
       transform (it supplies the widths, as everywhere else in this module);
    2. the mask is downsampled by ``downsample`` (nodes fall as the square, and
       the path is smoothed and resampled afterwards anyway) and an 8-connected
       grid graph is built on it with the cost weights above;
    3. ANCHOR = ``snout_hint_xy`` (the v5 ``snout_tip`` keypoint) snapped to the
       nearest mask pixel; without a hint, the oriented first point of
       :func:`extract_centerline`, i.e. the existing inner-band width rule.
       The anchor only chooses WHICH END is the head: it is a detector
       keypoint (or a width rule), lands short of the rostrum by a varying
       amount, and the 13 station fractions are fractions of the snout -> cut
       arc, so an anchor that wanders slides all 13 exactly as a wandering cut
       does.  ``extend_head`` (default) therefore walks the axis FORWARD off
       it onto the silhouette itself: the head chord of the anchored path
       (its leading ``head_frac`` of samples) gives a forward direction, the
       body node reaching furthest along it -- among nodes within
       ``head_search_frac`` (0.35) of the geodesic distance to the tail, so a
       bent animal's caudal lobe cannot win -- is taken as the head, the
       min-cost
       path back to the anchor (already in the Dijkstra tree) is prepended,
       and the last sub-cell of the coarse grid is recovered by
       :func:`_silhouette_tip` at full resolution.  ``endpoints["snout"]`` is
       then a point of the OUTLINE and ``endpoints["snout_source"]`` reads
       ``"hint_extended"`` / ``"width_extended"``;
    4. TAIL = the far end of the body, the pixel with the largest geodesic
       distance from the anchor (a metric-weighted Dijkstra over the same
       graph), which on a shark is the caudal fin tip;
    5. the min-cost path snout -> far end is smoothed and resampled, then CUT
       at the caudal peduncle by :func:`peduncle_cut_index`, with
       ``tail_hint_xy`` (the v5 ``caudal_notch`` keypoint) choosing between
       candidate width minima when it is given.  The returned polyline
       therefore ends at the peduncle whatever guided it, and
       ``endpoints["tail_source"]`` names the cue that MOVED the cut (a hint
       that merely agrees with the width rule is reported as ``"peduncle"``;
       ``endpoints["tail_hint_available"]`` says whether there was one).

       ``cut_ratio`` overrides that per-frame choice: the cut is then placed at
       that fraction of the uncut path's arc (:func:`cut_index_for_ratio`) and
       ``endpoints["cut_rule"]`` reads ``"fixed_ratio"``.  This is how the
       extractor holds the cut still across a sequence -- the 13 station
       fractions are fractions of the snout -> cut arc, so a cut that wanders
       slides all 13 along the body.  The per-frame reading is always reported
       as ``cut_ratio_free``, and the applied one as ``cut_ratio``.

    Returns the same dict as :func:`extract_centerline` plus ``axis``
    (``"dt_path"``), ``endpoints`` (``snout`` / ``snout_source`` / ``tail`` /
    ``tail_source`` / ``tail_hint_available`` / ``cut_rule``),
    ``fin_contamination``,
    ``peduncle_frac`` (where the cut fell on the uncut path, as an index
    fraction), ``cut_ratio`` / ``cut_ratio_free`` (the same position as an ARC
    fraction, applied and per-frame), ``head_extent_px`` (:func:`_forward_reach_px`,
    how far the silhouette still reaches forward of the returned snout
    endpoint -- the head anchor's residual, the counterpart of
    ``cut_ratio``'s scatter at the other end), ``polyline_full`` (the uncut
    path, snout -> far end) and ``widths_full`` (its width profile).

    The polyline spans snout tip -> peduncle, NOT snout tip -> caudal fin tip,
    so the arc span that maps rig stations onto it is
    :data:`RIG_ARC_SPAN_PEDUNCLE`.  ``span`` here is only the reference window
    for ``fin_contamination``.
    """
    ...

def extract_opening_axis(mask_bool: np.ndarray, snout_hint_xy: Optional[Sequence[float]]=None, tail_hint_xy: Optional[Sequence[float]]=None, n_points: int=32, open_frac: float=0.4, n_bins: int=48, smooth_frac: float=0.12, min_pixels: int=50, back_frac: float=0.45, tip_frac: float=0.05, span: Tuple[float, float]=RIG_ARC_SPAN_PEDUNCLE) -> Dict[str, object]:
    """Cross-check axis: erase the fins by opening, then run the slice mean.

    The fins are thin blades and the trunk is not, so a morphological opening
    with a disc of ``open_frac`` x the trunk's median HALF-width (taken from
    the distance transform along a first-pass centreline) removes them.  The
    slice mean of :func:`extract_centerline` then has no fin area left to be
    pulled by.  Widths and the peduncle cut still come from the ORIGINAL
    silhouette, so the reported widths are the animal's, not the eroded body's.

    An opening cannot preserve a taper: it eats the rostrum and the peduncle
    too, so this axis is shorter than the animal at both ends and its arc
    parameterisation is not comparable with the rig span.  It exists to check
    that the distance-transform path is not inventing its geometry.
    """
    ...

def extract_axis(mask_bool: np.ndarray, axis: str='dt_path', snout_hint_xy: Optional[Sequence[float]]=None, tail_hint_xy: Optional[Sequence[float]]=None, n_points: int=32, **kwargs) -> Dict[str, object]:
    """Dispatch to one of :data:`AXIS_CHOICES` and stamp the provenance block.

    ``"thinning"`` runs :func:`extract_centerline` unchanged and adds the
    ``axis`` / ``endpoints`` / ``fin_contamination`` fields, with
    ``tail_source`` ``"far_end"``: that axis runs out to the caudal fin tip and
    is never cut.
    """
    ...
