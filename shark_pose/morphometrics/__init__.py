"""3D morphometrics from v2 fit records.

Read-only: this package loads a record written by ``scripts/fit_video.py
--model v2``, re-runs ``SharkSMPL`` forward on the stored parameters, and
measures the posed mesh.  It changes no default, edits no asset and never
re-fits.

    from shark_pose.morphometrics import MorphAssets, Record, extract_clip, aggregate_clip

    assets = MorphAssets.load("data/model_assets_v2")
    record = Record.load("outputs/.../demo_none.json")
    result = aggregate_clip(extract_clip(record, assets))
"""
from .aggregate import ClipResult, IndividualResult, aggregate_clip, aggregate_individual, apply_support_contract, boxcar_smooth, bootstrap_median_ci, compose_band, gate_frames, measurement_support, median_mad, proxy_joints_to_native, resolve_girth_observation, resolve_individual_scale, rows_to_csv, rows_to_markdown, scale_propagate, support_state, weighted_median
from .extract import ClipExtraction, MorphAssets, Record, assign_to_polyline, girth_band_corners, divergence_volume, extract_clip, extract_frame, mesh_watertightness, ramanujan_perimeter, section_profile, station_population, template_reference
from .schema import CI_KINDS, CSV_COLUMNS, EXPECTED_RANGES, GIRTH_FIELD_QUANTITIES, PER_STATION_NEVER_OBSERVED, TRUNK_ONLY_UNMEASURED, resolve_observed_or_prior, RANGE_ORIENTATION_EXCEPTIONS, range_verdict_applies, QUANTITIES, THREE_D_ONLY, TWOD_COLUMNS, MorphometricRow, QuantitySpec
__all__ = ['CI_KINDS', 'CSV_COLUMNS', 'ClipExtraction', 'ClipResult', 'EXPECTED_RANGES', 'GIRTH_FIELD_QUANTITIES', 'PER_STATION_NEVER_OBSERVED', 'RANGE_ORIENTATION_EXCEPTIONS', 'TRUNK_ONLY_UNMEASURED', 'range_verdict_applies', 'IndividualResult', 'MorphAssets', 'MorphometricRow', 'QUANTITIES', 'QuantitySpec', 'Record', 'THREE_D_ONLY', 'TWOD_COLUMNS', 'aggregate_clip', 'aggregate_individual', 'apply_support_contract', 'assign_to_polyline', 'bootstrap_median_ci', 'boxcar_smooth', 'compose_band', 'divergence_volume', 'extract_clip', 'extract_frame', 'gate_frames', 'proxy_joints_to_native', 'girth_band_corners', 'measurement_support', 'median_mad', 'mesh_watertightness', 'ramanujan_perimeter', 'resolve_girth_observation', 'resolve_individual_scale', 'resolve_observed_or_prior', 'rows_to_csv', 'rows_to_markdown', 'scale_propagate', 'section_profile', 'station_population', 'support_state', 'template_reference', 'weighted_median']
