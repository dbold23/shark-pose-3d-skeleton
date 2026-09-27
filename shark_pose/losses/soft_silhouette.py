"""Differentiable silhouette rasterisation in pure PyTorch (no PyTorch3D).

The pipeline's DEFAULT camera is a fixed pinhole in the 256x256 letterbox proxy
space (``f = 1000``, ``c = 128``; see ``shark_pose/sim2real/proxy_geometry.py``).
This module rasterises a posed mesh through that camera at an arbitrary square
resolution ``res`` by scaling both focal length and principal point, so a
silhouette rendered at ``res`` is the 256-proxy silhouette resized by
``res / 256`` -- no second camera convention to keep in sync.

The pinhole is not hard-wired: ``base_focal`` / ``base_principal`` (both
expressed at :data:`BASE_RESOLUTION`) override it, which is how a calibrated
real camera -- ``sim2real.camera_calib.CameraIntrinsics.proxy_focal()`` and
``proxy_principal_point()`` -- reaches the rasteriser.  The caller is
responsible for handing the SAME camera to the fitter's reprojection term; the
whole point is that there is only ever one.

Soft rasterisation follows SoftRas: for every triangle ``f`` and every pixel
centre ``p`` we compute the *signed* 2D distance ``d_f(p)`` from ``p`` to the
projected triangle (negative inside), turn it into a per-face coverage

    coverage_f(p) = sigmoid(-d_f(p) / sigma)

and combine faces as a probabilistic union

    coverage(p) = 1 - prod_f (1 - coverage_f(p)).

The product is evaluated in log space, which is both stable and cheap because
``log(1 - sigmoid(-d/sigma)) = logsigmoid(d/sigma)`` exactly -- no clamping,
no ``log(0)``.  The union (rather than a z-buffer) is the right operator for a
silhouette: occlusion is irrelevant when only the 2D footprint matters, and it
keeps a gradient on every face, including back-facing ones that define the
outline on a thin body like a shark's caudal fin.

One property of the union worth knowing: two triangles sharing an edge each
contribute exactly 0.5 on that edge, so a *single-layer* surface reads 0.75
along its own interior edges.  On a closed mesh this is invisible, because the
far surface covers the near surface's edges (``tests/test_soft_silhouette.py``
pins both behaviours down).

Cost
----
Evaluated densely this is ``res^2`` pixels x ``F`` faces -- 128^2 x 5628 = 92M
face-pixel pairs for the shark mesh, measured at 3.0 s per forward+backward on
MPS, which is far too slow for a 150-step fit.  But a projected face of this
mesh spans ~2 px on average and ~24 px at worst, so all but a vanishing tail of
those pairs are a pixel and a triangle nowhere near each other.  Each face is
therefore rasterised only inside its own axis-aligned pixel window, grown by
``margin_sigmas * sigma`` so the soft tail is not truncated where it still
matters, and the per-face log-transmittances are ``scatter_add``-ed into the
image.  Faces are bucketed into a few window sizes so that the small majority
does not pay for the large minority.  Measured: ~25x fewer face-pixel pairs and
~0.1 s per forward+backward for the same mesh.

Setting ``window=res`` disables the optimisation and evaluates the exact dense
product -- that is the reference the windowed path is tested against.  Very
large face chunks are wrapped in ``torch.utils.checkpoint`` so the backward
graph of a dense evaluation stays bounded.

Public API
----------
``soft_silhouette``      differentiable coverage map, (B, res, res) in [0, 1]
``soft_iou``             differentiable IoU of a coverage map against a target
``soft_scanline_width``  differentiable CHORD width along a line, no raster
``hard_silhouette``      non-differentiable ``cv2.fillPoly`` mask at native
                         image resolution, for honest IoU reporting
``mask_iou``             plain binary IoU of two masks

The scanline sampler
--------------------
:func:`soft_scanline_width` answers a different question from the map: how long
is the silhouette's chord along ONE line?  It evaluates exactly the same
per-face coverage as above -- the same signed distance, the same sigma, the same
probabilistic union -- but only at points spaced ``step`` apart along the line,
and integrates that coverage.  There is no image, so the cost is the faces that
lie near the line rather than the faces times the frame.  That is the operator
the girth score measures the model's rendered silhouette with
(``mask_centerline.chord_width``), made differentiable, so a width objective and
a width score can finally be the same function.
"""
from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
__all__ = ['BASE_FOCAL', 'BASE_RESOLUTION', 'DEFAULT_SIGMA_PROXY', 'SCANLINE_DILATE_SIGMAS', 'SCANLINE_MARGIN_SIGMAS', 'SCANLINE_SIGMA_PX', 'SCANLINE_STEP_PX', 'hard_silhouette', 'mask_iou', 'pinhole_at', 'pinhole_for_resolution', 'project_to_pixels', 'sigma_at', 'soft_iou', 'soft_scanline_coverage', 'soft_scanline_width', 'soft_silhouette']
_EPS = 1e-08
_Z_MIN = 0.0001
_UV_LIMIT = 10000.0

def pinhole_at(res: int, base_focal: Optional[float]=None, base_principal: Optional[Sequence[float]]=None) -> tuple[float, float, float]:
    """Return ``(focal, cx, cy)`` of a pinhole at resolution ``res``.

    ``base_focal`` and ``base_principal`` are the camera expressed at
    :data:`BASE_RESOLUTION` (the 256 px proxy letterbox), and both are scaled
    by ``res / 256`` so a silhouette rendered at ``res`` is the 256-proxy
    silhouette resized -- ``u_res = u_256 * res / 256``.

    The defaults are the pipeline's fixed pinhole (:data:`BASE_FOCAL`, centred).
    A REAL camera goes through here too: ``camera_calib.CameraIntrinsics``
    supplies ``proxy_focal()`` and ``proxy_principal_point()``, and the
    principal point is then NOT at the image centre, which is exactly why it is
    a separate argument rather than ``res / 2``.
    """
    ...

def sigma_at(res: int, sigma_proxy: float=DEFAULT_SIGMA_PROXY) -> float:
    """Rasteriser ``sigma`` at resolution ``res`` for a softness in PROXY px.

    ``sigma`` is a length in the pixels of the raster the coverage map is
    written into, so a constant value means something different at every
    resolution: at ``res = 64`` one raster pixel is four proxy pixels, and a
    1 px sigma is a 4 proxy-px blur that inflates the coverage far past the
    outline of the mesh.

    Measured on the rest template placed at the on-screen size the demo clip's
    SAM masks have (13-14% of the letterbox; see
    ``outputs/demo_2026-09-03/work/girth/sigma_area.json`` and
    ``tests/test_soft_silhouette.py::test_sigma_at_area_matches_hard_silhouette``),
    the soft coverage area as a multiple of the hard silhouette's is

        res         64     96    128
        sigma = 1   2.12   1.68  1.47      <- a fixed raster sigma
        this fn     0.99   1.04  1.06      <- 0.75 proxy px

    -- at ``res = 64``, the resolution the fitter is cheap enough to run at,
    a fixed ``sigma = 1`` asks the mesh to cover twice its own area, and the
    only way a mesh covers more area is by getting bigger or closer, so the
    term pushes the animal exactly the wrong way.  Expressing the softness in
    PROXY pixels and scaling it to the raster,

        sigma_raster = sigma_proxy * res / BASE_RESOLUTION,

    makes the blur a constant fraction of the body, so the area ratio stops
    drifting with resolution.  It does not become exactly 1, and should not:
    on a CLOSED mesh the outline edge is shared by two faces, each of which
    contributes 0.5 there, so the union puts the half-coverage contour about
    0.8 sigma OUTSIDE the true outline.  That offset is a fixed fraction of the
    body at every resolution, which is the whole point.  On a body filling a
    quarter of the letterbox the same sigma measures 0.97 / 1.01 / 1.02.
    :data:`DEFAULT_SIGMA_PROXY` is 0.75.

    Args:
        res: raster resolution the coverage map is written at.
        sigma_proxy: softness in the 256 px proxy letterbox's pixels.

    Returns:
        the ``sigma`` to hand :func:`soft_silhouette` at ``res``.
    """
    ...

def pinhole_for_resolution(res: int) -> tuple[float, float]:
    """Return ``(focal, principal)`` of the DEFAULT pipeline pinhole at ``res``.

    Kept for the callers (and tests) that only ever mean the fixed pinhole; it
    is :func:`pinhole_at` with both offsets equal.
    """
    ...

def project_to_pixels(points_cam: Tensor, res: int, base_focal: Optional[float]=None, base_principal: Optional[Sequence[float]]=None) -> Tensor:
    """Project camera-frame points to pixel coordinates at resolution ``res``.

    Args:
        points_cam: (..., 3) points in the camera frame (+z away from camera).
        res: square target resolution.
        base_focal: focal length at :data:`BASE_RESOLUTION`; ``None`` = the
            pipeline default :data:`BASE_FOCAL`.
        base_principal: ``(cx, cy)`` at :data:`BASE_RESOLUTION`; ``None`` =
            the image centre.

    Returns:
        (..., 2) pixel coordinates ``(u, v)``.  Points at or behind the camera
        are clamped to a small positive depth and the result to a finite range;
        callers are expected to mask such faces out.
    """
    ...

def _signed_distance(tri: Tensor, px: Tensor, py: Tensor) -> Tensor:
    """Signed distance from pixel centres to projected triangles.

    Args:
        tri: (B, M, 3, 2) triangle vertices in pixels.
        px, py: (B, M, P) pixel-centre coordinates, one set per face.

    Returns:
        (B, M, P) distance to the triangle boundary, negative inside.
    """
    ...

def _window_log_free(tri: Tensor, valid: Tensor, base: Tensor, res: int, sigma: float, window: int) -> Tensor:
    """Scatter one group of faces' ``log(1 - coverage)`` into the image.

    Args:
        tri: (B, M, 3, 2) projected triangles.
        valid: (B, M) float mask; invalid faces contribute exactly zero.
        base: (B, M, 2) integer top-left pixel index of each face's window.
        res: image resolution.
        sigma: softness in pixels.
        window: window edge length in pixels.

    Returns:
        (B, res * res) log-transmittance from these faces.
    """
    ...

def soft_silhouette(verts_cam: Tensor, faces: Tensor, res: int=128, sigma: float=1.0, chunk: int=512, face_stride: int=1, window: Optional[int]=None, margin_sigmas: float=8.0, use_checkpoint: Optional[bool]=None, base_focal: Optional[float]=None, base_principal: Optional[Sequence[float]]=None) -> Tensor:
    """Differentiable silhouette coverage of a posed mesh.

    Args:
        verts_cam: (B, V, 3) vertices in the camera frame (translation applied).
        faces: (F, 3) integer vertex indices.
        res: square output resolution.
        sigma: softness in RASTER pixels -- the pixels of ``res``, not of the
            proxy -- so do not hard-code one: pass
            ``sigma_at(res, DEFAULT_SIGMA_PROXY)``, which fixes the blur as a
            length on the IMAGE and therefore keeps the coverage map the same
            size as the mesh at every resolution (a fixed 1 raster px is a
            4 proxy-px blur at ``res = 64`` and inflates the coverage to 2.1x
            the hard silhouette's area; the default lands at 0.99-1.06x --
            ``girth/sigma_area.json``).  The older advice here ("~1 px is a
            good compromise at res = 128", sharper being "less well
            conditioned") is superseded, and its conditioning worry does not
            survive measurement: at the demo's on-screen size the gradient norm
            of ``1 - soft_iou`` w.r.t. the vertices is 0.0863 at sigma 1 and
            0.1534 at the default at ``res = 128``, i.e. the sharper raster
            gives a 1.8x STRONGER gradient (6.0x at res 64, 3.0x at 96 --
            ``girth/round1/sigma_gradient.json``).
        chunk: memory budget, expressed as a number of faces at full-image
            width.  A chunk holds ``chunk * res^2`` face-pixel pairs however
            the faces are windowed, so this bounds peak memory regardless of
            the window size chosen.
        face_stride: keep every ``face_stride``-th face (1 = all).  Decimating
            a closed mesh shrinks its silhouette, so use this for speed only.
        window: force a per-face window edge length instead of choosing one per
            face.  ``res`` gives the exact dense product; ``None`` (default)
            picks the smallest window from an internal ladder that contains
            each face plus its ``margin_sigmas * sigma`` tail.
        margin_sigmas: how far past a face the soft tail is kept, in units of
            ``sigma``.  At the default 8 a truncated face contributes at most
            ``3.4e-4`` coverage, so the windowing is invisible next to the
            rasteriser's own softness.
        use_checkpoint: recompute face chunks in backward instead of storing
            them.  ``None`` enables it only for chunks large enough to matter.
        base_focal, base_principal: the camera to rasterise through, expressed
            at :data:`BASE_RESOLUTION` (see :func:`pinhole_at`).  ``None``/
            ``None`` is the fixed pipeline pinhole; a calibrated camera passes
            ``CameraIntrinsics.proxy_focal()`` and ``proxy_principal_point()``
            so that this term and the landmark term share ONE camera.

    Returns:
        (B, res, res) coverage in [0, 1], indexed ``[batch, v, u]``.
    """
    ...

def soft_iou(coverage: Tensor, target: Tensor, eps: float=1e-06) -> Tensor:
    """Differentiable IoU between a coverage map and a target occupancy map.

    Uses the probabilistic form ``sum(a*b) / sum(a + b - a*b)``, which reduces
    to the ordinary IoU when both maps are binary.

    Args:
        coverage: (B, H, W) values in [0, 1].
        target: (B, H, W) or (H, W) values in [0, 1].

    Returns:
        (B,) IoU per batch element.
    """
    ...

def _scanline_run_mask(coverage: Tensor, centre: int, dilate_samples: int) -> Tensor:
    """(B, K, P) 0/1 mask of the ONE covered run nearest the centre.

    ``mask_centerline.chord_width`` does not return the total covered length
    along its ray: it returns the length of ONE contiguous run of silhouette --
    the run containing the station point when there is one, else the run whose
    nearest sample is closest to it.  This is that rule, both halves of it,
    applied to the soft profile at its half level and then dilated by
    ``dilate_samples`` so the soft tail -- which is where the gradient lives --
    stays inside the run.  A disjoint second run stays outside ONLY when its
    gap exceeds the dilation (about 2 proxy px at the shipped settings, see
    :data:`SCANLINE_DILATE_SIGMAS`); a closer one is merged, which is the
    measured source of the worst per-frame scanline/chord deviations.

    Before 2026-09-06 only the first half was implemented: with the centre
    sample uncovered the WHOLE line was kept, which sums every disjoint run the
    line crosses -- 2.0x the chord on two cylinders with the point in the gap
    between them (``test_scanline_run_selection_takes_the_nearest_run_from_a_gap``
    is that case, and pins WHICH run is taken by giving the two different
    radii).  On the real fits the difference is not observable and the change
    is a correctness one: at all eight girth3 rows' optima the centre sample is
    covered in 4160 of 4160 station-frames, and over a whole fit's trajectory
    in 19300 of 19300, so both rules return the same widths
    (``outputs/demo_2026-09-03/work/girth3/fix1/rule_delta.json``,
    ``branch_probe.json``).  The stations that DO project off the model's
    silhouette there are ones the camera has cropped, where the soft line still
    crosses one run and it is the raster chord, not this, that reads something
    else (``fix1/operator_frames.json``).

    Ties are broken the way ``chord_width``'s ``argmin`` breaks them: the
    earlier run along the line wins.  When no sample is covered at all there is
    no run to pick and the whole line is kept, which reads a near-zero width
    because no sample is at half coverage either.

    Selecting the run is a support decision, exactly like the rasteriser's
    window placement, so it carries no gradient: the caller multiplies the
    (differentiable) coverage by this (detached) mask.
    """
    ...

def soft_scanline_coverage(verts_px: Tensor, faces: Tensor, points_px: Tensor, normals_px: Tensor, half_len: Tensor | float, sigma: float=SCANLINE_SIGMA_PX, step: float=SCANLINE_STEP_PX, face_valid: Optional[Tensor]=None, face_stride: int=1, margin_sigmas: float=SCANLINE_MARGIN_SIGMAS, sample_window: Optional[int]=None) -> tuple[Tensor, Tensor]:
    """Soft silhouette coverage sampled along one line per station.

    Args:
        verts_px: (B, V, 2) vertices ALREADY projected to pixels.  Any pixel
            space will do -- the fitter's proxy, or native frame pixels -- as
            long as ``points_px``, ``sigma``, ``step`` and ``half_len`` are
            lengths in the same one.  Projection is the caller's so that this
            and the fit's reprojection term cannot drift apart.
        faces: (F, 3) vertex indices.
        points_px: (B, K, 2) the point each line passes through.
        normals_px: (B, K, 2) the direction each line runs along; normalised
            here, so it need not be a unit vector.
        half_len: (B, K), (K,) or scalar: how far the line reaches on either
            side of its point.  Detached -- it decides which faces and which
            samples take part, and a support decision carries no gradient.
        sigma: softness in the pixels of ``verts_px``; see
            :data:`SCANLINE_SIGMA_PX`, which is NOT the rasteriser's sigma.
        step: sample spacing, same pixels.
        face_valid: (B, F) mask; a face that is False contributes exactly zero
            (this is where "behind the camera" belongs).
        face_stride: keep every n-th face.  Decimating a closed mesh shrinks
            its silhouette; for speed only.
        margin_sigmas: how far past a face its tail is kept
            (:data:`SCANLINE_MARGIN_SIGMAS`).
        sample_window: force a per-face window of this many samples instead of
            choosing one per face from an internal ladder.  The number of
            samples on the grid gives the exact dense evaluation, which is what
            the windowed path is tested against.

    Returns:
        ``(coverage (B, K, P), t (P,))`` -- the probabilistic-union coverage at
        each sample and the sample's signed offset from the line's point.  The
        grid is symmetric and odd-length, so ``t[P // 2] == 0``; it is sized by
        the LARGEST ``half_len``, and samples past a station's own ``half_len``
        read zero because no face of theirs was gated in.

    The cost is the faces near a line, not the faces times an image.  A face
    takes part when its extent perpendicular to the line straddles the line
    (within the margin) and its extent along the line reaches the segment; it
    is then evaluated only inside its own window of samples, the smallest on
    the ladder that holds its along-line extent plus both tails.
    """
    ...

def soft_scanline_width(verts_px: Tensor, faces: Tensor, points_px: Tensor, normals_px: Tensor, half_len: Tensor | float, sigma: float=SCANLINE_SIGMA_PX, step: float=SCANLINE_STEP_PX, face_valid: Optional[Tensor]=None, face_stride: int=1, margin_sigmas: float=SCANLINE_MARGIN_SIGMAS, dilate_sigmas: float=SCANLINE_DILATE_SIGMAS, sample_window: Optional[int]=None, run_select: bool=True) -> Tensor:
    """(B, K) differentiable chord width of the mesh's silhouette per line.

    The integral of :func:`soft_scanline_coverage` along the line: for a convex
    cross-section the coverage is 1 inside the silhouette and 0 outside, so
    ``sum(coverage) * step`` IS the chord length -- the same number
    ``mask_centerline.chord_width`` marches out on a rendered mask, and the
    same number the girth score divides by.  With ``run_select`` (the default)
    the integral is restricted to the contiguous run containing the line's
    point, which is that function's rule too (:func:`_scanline_run_mask`).

    MEASURED against ``chord_width`` on the model's OWN hard render, at the
    fitted optimum of the shipped from-scratch fits of both clips (girth2 row
    i, 32 frames each; the numbers are medians of model-scanline / model-chord
    per station, ``outputs/demo_2026-09-03/work/girth3/operator_accept.json``):
    every trunk station 3-9 agrees to within 3% on both clips at the default
    sigma.  A rest-pose synthetic render is NOT evidence for this: the previous
    round's band operator agreed to 3% there and to 9-16% here.

    That 3% is a MEDIAN OVER FRAMES per station, AT ROW I, and neither
    qualification is decorative.  Re-measured on all ten of girth round 3's
    fitted optima (20 frames x 13 stations each,
    ``.../girth3/review2/operator_optima.json``) the worst trunk station median
    runs 1.6-2.6% on the demo and 1.5-3.3% on clip 2, so a live-girth optimum
    can sit outside the 3%.  The per-frame tail is much fatter still -- the
    worst single trunk station-frame is 1.03-1.28x on the demo and 1.22-1.66x
    on clip 2 once stations the camera has cropped and chords cut by the frame
    edge are set aside, and up to 5.6x with them in (26-34 of 260 station-frames
    on clip 2, none on the demo), because there the raster operator has no body
    under the point to march over at all (the per-frame breakdown for the eight
    rows that existed when it was written is
    ``.../girth3/fix1/operator_frames.json``).  What
    the rest of the tail is made of, measured (girth3/review3/worst_case.py):
    the run mask's dilation -- 1.0 proxy px each side at the shipped sigma and
    step -- merges a second hard run whose gap to the chosen one is under about
    2 proxy px, so the scanline integrates both where the chord marches one.
    Frames with such a neighbour are few (clip2_ii 5 of 126 clean trunk
    station-frames, clip2_i 8 of 127) but carry the worst deviations (61% /
    22% max); without them the two agree within 4.2% / 3.2% per frame.

    What the two still cannot agree about is a station where the silhouette's
    run is genuinely ambiguous -- the peduncle and the caudal (stations 10-12),
    where the chord's run selection and this one part company on whether the
    fin's root is body -- which is why the width term compares trunk stations.
    """
    ...

def hard_silhouette(verts_cam: np.ndarray, faces: np.ndarray, width: int, height: int, res: int=BASE_RESOLUTION, base_focal: Optional[float]=None, base_principal: Optional[Sequence[float]]=None) -> np.ndarray:
    """Binary silhouette at native image resolution via ``cv2.fillPoly``.

    Non-differentiable; this is the honest yardstick the soft rasteriser is
    scored against, at the resolution the SAM masks actually live at.  The
    projection goes through the same pinhole and then the inverse letterbox, so
    the mask is in original-frame pixels.

    One ``fillPoly`` call per triangle is deliberate: a single batched call
    fills the whole list with the even-odd rule, so on a closed mesh the front
    and back faces cancel and the interior comes out hollow.

    Args:
        verts_cam: (V, 3) camera-frame vertices.
        faces: (F, 3) vertex indices.
        width, height: native frame size.
        res: proxy resolution the pinhole is expressed at (default 256).
        base_focal, base_principal: the camera, expressed at
            :data:`BASE_RESOLUTION` (see :func:`pinhole_at`).  With a calibrated
            camera the mask this returns lives in the UNDISTORTED frame, so
            score it against an undistorted mask.

    Returns:
        (height, width) uint8 mask, 0 or 255.
    """
    ...

def mask_iou(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Binary IoU of two masks; ``None`` when both are empty."""
    ...
