# shark-pose-3d

3D pose and body shape of free-swimming white sharks from monocular underwater video.

> **This is a skeleton.** It is the real module layout, class and function signatures and
> docstrings of a private research codebase, with every function body replaced by `...`
> and all data, weights, configuration and credentials left out. It shows how the system
> is built; it does not run. The full code is private while the work is prepared for
> publication. Copyright Daniel Sambold, all rights reserved.

![Per-individual fit pipeline](docs/pipeline.svg)

White sharks cannot be weighed, so body condition has to come from video. A 16-keypoint
pose detector and SAM2 silhouettes drive a fit of a rigged parametric shark model
(SharkSMPL: 16-joint skeleton, PCA shape space, linear blend skinning) over a window of
frames. The fit is then read as a girth field, station chords and length proportions,
with a support contract that refuses a measurement the footage cannot back.

## Where it stands

- Run over the archive: 1,011 single-window records on 250 individuals.
- Quotable per animal today: length proportions and girth-profile shape.
- Not yet quotable: absolute girth. On a known-truth control the fitter reads 7 to 12 %
  wide, and no second view or configuration removed it. Certifying it needs an external
  width measurement (drone stations or a physical phantom).

## Layout

| Package | What it holds |
|---|---|
| `model_3d/` | SharkSMPL: skeleton, shape and pose spaces, skinning, glTF I/O |
| `encoder/` | SPIN-style regressor, SMPLify fitting, joint per-individual fit |
| `losses/` | Keypoint, soft-silhouette, pose/shape priors, temporal and adversarial terms |
| `sim2real/`, `synthetic/` | Domain-randomised synthetic renders and proxy inputs |
| `temporal/` | Sequence smoothing across frames |
| `morphometrics/` | Girth field, deprojection, station chords, aggregation with refusal |
| `corpus/` | Which windows of the archive are trustworthy enough to fit |
| `integration/` | Bridges to the annotation platform and segmentation |
| `tests/` | The test suite's structure and test names |

```
shark_pose/
    active_learning/
    core/
    corpus/
    data/
    encoder/
    integration/
    losses/
    model_3d/
    morphometrics/
    refinement/
    sim2real/
    synthetic/
        blender/
    temporal/
tests/
```
