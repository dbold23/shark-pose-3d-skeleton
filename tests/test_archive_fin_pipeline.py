"""End-to-end test of the S8 R3 chain (fin_crops.process_video -> fin_match) on synthetic videos.

The model steps are replaced by exact stand-ins: the 'detector' returns the rendered keypoints and the
'segmenter' returns the rendered mask, so this checks decode, frame choice, cropping, file layout,
profile extraction, encounter grouping and scoring, not the models.
"""
from __future__ import annotations
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from fin_crops import process_video
from shark_pose.archive.synth_fins import random_identity, render

def _write_video(path, sightings, fps=10):
    ...

def test_chain_on_synthetic_videos(tmp_path):
    ...

def test_still_images_with_item_map(tmp_path):
    """The all_fin_IDs/ route: stills named by encounter code, mapped to encounters by an item map."""
    ...
