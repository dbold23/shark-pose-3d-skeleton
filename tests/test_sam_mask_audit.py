"""S4/masks: the fitter must refuse a sam_masks bundle whose PNGs are missing.

The defect: 108 of 463 corpus window bundles carry a GPU-written
``sam_masks/index.json`` listing 150 masks and zero mask PNGs; ``load_block_masks``
silently returns nothing and the fit loses its silhouette term without a word.
"""
import json
import sys
from pathlib import Path
import pytest
from fit_video import audit_sam_masks, check_sam_masks

def _bundle(tmp_path: Path, claimed: int, written: int) -> Path:
    ...

def test_complete_bundle_is_ok_and_passes(tmp_path):
    ...

def test_missing_pngs_raise_and_name_the_bundle(tmp_path):
    ...

def test_partial_bundle_also_raises(tmp_path):
    ...

def test_allow_missing_masks_downgrades_to_a_warning(tmp_path, caplog):
    ...

def test_no_index_and_empty_index_are_not_refused(tmp_path):
    ...
