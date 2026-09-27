"""Tests for S8 S1 body coordinates, on real stage 7 records.

Known (u, theta) patches are placed on a record's own fitted tube, projected to native pixels as
annotation polygons, and mapped back by scar_map.py. This checks the geometry, the letterbox, the
record lookup and the CSV contract; it does not check the tube against the true mesh (that needs the
SAM masks on the owner's Mac: s8/s1_scar_body/src/tube_iou.py).
"""
from __future__ import annotations
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from shark_pose.archive.body_coords import Camera, build_tube, cast, sections_from_template_reference, surface_point, visible_cells

def _record():
    ...

def test_joints_project_to_the_records_own_2d():
    ...

def test_letterbox_matches_proxy_geometry():
    ...

def test_round_trip_on_a_real_frame():
    ...

def test_only_one_flank_is_visible_broadside():
    ...

def test_scar_map_recovers_placed_patches(tmp_path):
    ...

def test_tube_silhouette_contains_the_projected_spine():
    ...
