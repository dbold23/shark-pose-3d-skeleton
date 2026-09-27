"""Tests for the pure-torch soft silhouette rasteriser.

The reference the windowed fast path is checked against is the exact dense
product (``window=res``), and the reference *that* is checked against is
analytic geometry: a square of known size at known depth projects to a square
of computable pixel area.
"""
import numpy as np
import pytest
import torch
from shark_pose.losses.soft_silhouette import BASE_FOCAL, BASE_RESOLUTION, DEFAULT_SIGMA_PROXY, SCANLINE_SIGMA_PX, hard_silhouette, sigma_at, mask_iou, pinhole_for_resolution, project_to_pixels, soft_iou, soft_scanline_coverage, soft_scanline_width, soft_silhouette
HALF = 1.0
DEPTH = 10.0

@pytest.fixture
def square():
    ...

def expected_side_px(res):
    ...

def test_pinhole_matches_pipeline_at_base_resolution():
    ...

def test_projection_scales_linearly_with_resolution():
    ...

def test_centre_of_frame_projects_to_principal_point():
    ...

def test_shape_and_range(square):
    ...

def test_interior_saturates_and_far_corner_is_empty(square):
    ...

def test_shared_edge_of_a_single_layer_dips_to_three_quarters(square):
    """Two triangles meeting on an edge each give 0.5 there: 1 - 0.5^2 = 0.75.

    A documented property of the probabilistic union, not a bug.  It is
    invisible on a closed mesh, where the far surface covers the near
    surface's interior edges.
    """
    ...

def test_a_closed_mesh_has_no_interior_holes():
    """Front and back layers of a closed box fill each other's shared edges."""
    ...

def test_area_matches_analytic_projection(square):
    ...

def test_translation_moves_the_silhouette(square):
    ...

def test_faces_behind_the_camera_are_dropped(square):
    ...

def test_sigma_controls_edge_softness(square):
    ...

def test_windowed_matches_dense_for_flat_geometry(square):
    ...

def test_windowed_matches_dense_for_a_many_face_mesh():
    ...

def test_chunk_size_does_not_change_the_result(square):
    ...

def test_no_double_counting_when_faces_need_different_windows():
    """A tiny face and a huge face land in different window buckets."""
    ...

def test_batch_elements_are_independent(square):
    ...

def test_gradients_flow_to_vertices(square):
    ...

def test_gradient_pushes_the_silhouette_toward_the_target(square):
    """One Adam step on soft IoU should move a shifted square back."""
    ...

def test_checkpointing_does_not_change_gradients(square):
    ...

def test_soft_iou_of_identical_binary_maps_is_one():
    ...

def test_soft_iou_of_disjoint_maps_is_zero():
    ...

def test_soft_iou_matches_hand_computed_overlap():
    ...

def test_soft_iou_broadcasts_an_unbatched_target(square):
    ...

def test_hard_silhouette_area_matches_analytic_square(square):
    ...

def test_hard_silhouette_is_filled_not_hollow(square):
    """Even-odd filling of a closed mesh would hollow the interior out."""
    ...

def test_hard_and_soft_silhouettes_agree(square):
    ...

def test_mask_iou_edge_cases():
    ...

@pytest.mark.parametrize('kwargs', [{'res': 0}, {'sigma': 0.0}, {'chunk': 0}, {'face_stride': 0}, {'window': 999}])
def test_invalid_arguments_raise(square, kwargs):
    ...

def test_bad_tensor_shapes_raise(square):
    ...

def test_face_stride_decimates(square):
    ...

def test_sigma_at_scales_the_softness_with_the_raster():
    """One length, expressed once at 256, meaning the same thing everywhere."""
    ...

@pytest.fixture
def small_box():
    """A CLOSED box, small in frame -- an area test needs both properties.

    Closed, because the union reads 0.75 along a single-layer surface's own
    interior edges (the far surface covers them on a closed mesh, which is what
    a real body is).  Small, because the inflation a blur adds is proportional
    to the PERIMETER: on a body filling the frame it is a rounding error at any
    resolution, and the drift this test is about would not show.
    """
    ...

def test_sigma_at_area_matches_hard_silhouette_across_resolutions(small_box):
    """The point of the proxy-px convention: the soft AREA stops drifting.

    A fixed raster ``sigma`` blurs the outline by a constant number of RASTER
    pixels, so the inflation it adds is a constant fraction of the PERIMETER
    and therefore shrinks as the resolution grows -- the coverage map is a
    different-sized target at every res, and the term that chases it moves the
    animal.  ``sigma_at`` holds the blur at a fixed fraction of the body.
    """
    ...
SCAN_RES = 512

def cylinder(radius=0.5, length=6.0, depth=10.0, n_theta=48, n_seg=12):
    """A CLOSED capped cylinder lying along x, centred on the optical axis.

    Closed because a silhouette chord through an open surface is not a chord
    through a body: the rim of a real mesh is shared by a front face and a back
    face, and that shared rim is exactly what makes the soft union's outward
    bias -- the thing the scanline's sigma is chosen against -- appear at all.
    """
    ...

def scan_and_chord(verts_cam, faces, res=SCAN_RES):
    """``(scanline width, hard chord)`` at the cylinder's centre, same pixels.

    ``hard_silhouette`` at ``width = height = res`` is the identity letterbox,
    so its mask lives in exactly the pixel space ``project_to_pixels`` writes
    into and the two widths are directly comparable.
    """
    ...

def test_scanline_width_matches_the_hard_chord_on_a_cylinder():
    """THE acceptance, in miniature: soft scanline == hard chord, one body.

    The real acceptance is the same comparison on the fitted optimum of two
    real clips (``outputs/demo_2026-09-03/work/girth3/operator_accept.json``,
    every trunk station within 2%); this pins the same property on geometry
    whose answer is known, so a regression shows up here first.
    """
    ...

def test_scanline_width_scales_with_the_cross_section():
    """A body 1.2x thicker reads 1.2x wider -- the whole point of the term."""
    ...

def test_scanline_sigma_is_set_by_the_mesh_not_by_the_rasteriser():
    """Why the scanline's sigma is not the rasteriser's, on known geometry.

    Two biases fight, and both scale with sigma measured against the mesh's own
    triangles.  OUTWARD: every silhouette edge is shared by a front face and a
    back face, so the union reads 0.75 there and its half-coverage contour sits
    outside the outline -- and where the mesh is fine, many more faces than two
    put their tails on the same rim.  INWARD: the union also dips to 0.9375
    wherever a sample lands on an interior edge shared by four faces, and a fine
    mesh has such an edge every couple of pixels.  An AREA integral pays neither
    much; a scanline crosses the rim twice and every interior edge once, so it
    pays both.

    Measured here, soft scanline / hard chord on the same cylinder:

        mesh            sigma 0.15    sigma 0.75 (the rasteriser's)
        1152 faces        0.992          0.996
        12800 faces       0.977          1.035

    -- the coarse mesh cannot tell the two sigmas apart and the fine one is 6%
    apart on them.  The shark's template is the fine case (0.9 px triangles),
    which is why the shipped default is
    :data:`SCANLINE_SIGMA_PX` and the real acceptance is measured on the shark
    at its fitted optimum, not on a synthetic cylinder.
    """
    ...

def test_scanline_windowed_matches_the_dense_evaluation():
    """The per-face sample windows are an optimisation, not a change of answer."""
    ...

def test_scanline_run_selection_takes_only_the_run_at_the_point():
    """A second, disjoint silhouette on the same line is not part of the chord.

    ``mask_centerline.chord_width`` returns the contiguous run containing the
    point, not the total covered length; a fin sheet the line clips further out
    is not the body's width there and must not be added to it.
    """
    ...

def test_scanline_run_selection_takes_the_nearest_run_from_a_gap():
    """The point in the GAP: ``chord_width`` takes the nearest run, so does this.

    The case the run-selection test above does NOT cover, and the one real fits
    hit -- a station whose projected point has fallen off the body between two
    pieces of silhouette.  Keeping the whole line there (what this did before
    2026-09-06) sums both runs, which is neither operator's answer.

    Two cylinders of different radius with a 90 px gap: from a point on the
    wide side of the gap both operators read the wide run, from the thin side
    both read the thin one, and the two answers differ by 2x -- so this pins
    WHICH run is chosen, not merely that one is.
    """
    ...

def test_scanline_gradient_points_at_the_target():
    """A gradient step on the mesh moves the measured width toward the target."""
    ...

def test_scanline_rejects_bad_inputs():
    ...
