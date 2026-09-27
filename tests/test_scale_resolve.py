"""D1/D2 -- the label ladder (`scale_resolve`) and gate clauses 6 and 7.

What these tests exist for, in one sentence each:

* ``_fuse_scale`` used to take ``bounded[0]`` -- the first clip **in directory
  order** that carried a length field -- so an individual whose own sources
  disagree published one of them, silently, and which one depended on the shell
  glob.  ``APT22062207`` carries 14.0 ft and 9.5 ft: a **3.2x** volume error
  decided by a sort.
* the band around that label was the uncited ``[0.10, 0.20]``, which the
  propagator reads through ``max()`` as a flat +-20 %, i.e. **x3.375** on
  volume.  No derivation for it exists in either repo.  May et al.'s observer
  error is a length in CENTIMETRES, so the same sigma is a 20.5 % band on a
  2.44 m animal and 8.6 % on a 5.79 m one, and a flat fraction is wrong at both
  ends of the corpus.
* the girth gate's only independence rule was within-video window overlap, so
  two clips 7 s apart in one continuous view counted as an independent pair.

The disagreeing-individual fixture is ``tests/data/scale_label_disagreements.csv``,
built from the corpus by
``outputs/demo_2026-09-03/work/individuals/implement/scale/src/build_label_fixture.py``
(``outputs/`` is not tracked, so the rows are committed rather than read).
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import pytest
from shark_pose.morphometrics.aggregate import CLIP_PASS_GAP_S, GIRTH_STATION_IN_FRAME_MIN_FRAC, _fuse_scale, fuse_girth
from shark_pose.morphometrics.extract import Record
from shark_pose.sim2real import world_frame as wf

def _fixture_rows() -> dict:
    ...

def test_the_individuals_whose_own_labels_disagree():
    """The 19 + APT22062207, scored by the spec's own 5 % rule.

    **The plan row predicts all 19 are ``contested``; the data says 16.**  Three
    of them (`ANO24100304` 3.6 %, `PR12121305` 3.3 %, `PR15110602` 4.7 %)
    disagree by less than the spec's 5 % bar, so they resolve -- with the
    disagreement published in ``spread_frac`` and half of it added to the band,
    because a 3-5 % length disagreement is still 10-15 % of volume.  The rule is
    the spec's (4.6 clause 4); the prediction was the plan's.
    """
    ...

def test_apt22062207_refuses_its_3_2x_volume_disagreement():
    """14.0 ft against 9.5 ft: the worst label disagreement in the corpus."""
    ...

def test_catalog_size_ft_is_quarantined_always():
    """Mean 8.02 % and max 120 % off the filename token over 868 individuals,
    and 433 of its 2522 sub-4 ft values are CLIP INDICES."""
    ...

def test_five_sources_are_one_observation():
    """P7: independence is COUNTED, not summed -- and nothing is averaged.

    Pairwise median |rel| is 0.0 % on every human-source pair and the deploy PDF
    agrees with the ft label to 0.17 %, so the five in-corpus sources are one
    field estimate copied five times.  285 of 296 fieldsheet/filename pairs are
    bit-identical and must vote once.
    """
    ...

def test_the_median_is_over_distinct_values_not_over_rows():
    """Many bit-identical copies must not outvote the one different reading."""
    ...

def test_sigma_is_a_length_in_centimetres_not_a_fraction():
    """The corpus spans 2.438-5.791 m, so one sigma is two different bands."""
    ...

def test_the_band_is_one_named_multiple_and_both_ends_are_it():
    """5.4: ``aggregate`` takes ``max(uncertainty_frac)``, so writing
    ``[1 sigma, 2 sigma]`` there would make every record silently quote 2 sigma
    -- which is exactly how the uncited ``[0.10, 0.20]`` became a flat 20 %."""
    ...

def test_crew_sigma_is_a_range_and_the_half_normal_fit_is_recorded():
    """The MAE and SD inversions disagree by 12.1 % on the crew class; they are
    one assumption applied twice, not two observations."""
    ...

def test_observer_class_is_the_data_entry_user_and_says_so():
    """2.5: the `observer` column is the record's ``user`` field -- 508 of 514
    field-sheet rows read ``DM`` -- not a field observer.  ``crew`` is assigned
    by ABSENCE of provenance for 81 of the 111 fit-capable individuals."""
    ...

def test_a_label_outside_the_calibration_range_is_flagged_by_SIDE():
    """All 31 of the 111 that fall outside sit ABOVE 4.70 m, where the source's
    own trend has observers under-estimating: a symmetric band is
    anti-conservative there, so the record must say which side."""
    ...

def test_no_usable_label_is_absent_with_a_reason_never_zero():
    ...

def test_scale_block_reproduces_the_shipped_scale_with_a_new_band():
    """AN15092101's fitted record, re-banded: the SAME metres_per_model_unit to
    the last bit, a band that is no longer the uncited [0.10, 0.20]."""
    ...

def test_scale_block_accepts_a_label_that_lives_on_no_file_name():
    """R4(ii): ``parse_length_label`` reads the video FILE NAME and nothing
    else, so a FinID-only or fieldsheet-only individual had no route to a scale
    at all and the whole named-observer rung was unreachable in code."""
    ...

def test_scale_block_refuses_a_contested_individual():
    ...

def test_an_unlabelled_clip_is_still_unmet_with_its_old_reason():
    ...

def _clip_result(mpu, span, label_m, source, name='v.mp4'):
    ...

def test_fuse_scale_no_longer_takes_the_first_clip_in_directory_order():
    ...

def _girth_record(clip, video, t0_s=0.0, n_frames=300, fps=60.0, creation_time=None, pass_id=None, stations_observed=None, clips=('a', 'b'), n_clips=2):
    ...

def test_clause_6_refuses_two_windows_of_one_video():
    """Two clips 7 s apart in one pass are not two views of the animal."""
    ...

def test_clause_6_separates_two_videos_of_one_pass_from_two_passes():
    ...

def test_clause_6_records_the_fallback_when_no_creation_time_exists():
    """Every record fitted before the metadata sweep: the clause falls back to
    'different video' and SAYS so rather than passing off a pass test."""
    ...

def test_clause_6_needs_one_independent_pair_not_all_of_them():
    """At N=3 the pair that identifies the lateral axis is what the clause has
    to hold on; a third, correlated window adds information to the shared field
    without invalidating it.  This is AN15092101's N=3 record: two of its three
    windows come from ``12ft_03.mp4``."""
    ...

def test_clause_7_refuses_the_off_canvas_station_and_only_that_one():
    ...

def test_clause_7_is_not_evaluated_when_no_record_carries_the_field():
    """No record on disk carries ``stations_observed`` -- the per-station
    in-frame flags are one of the five TRUNK_ONLY fitting edits deferred to
    phase 6 -- so the clause must report that, never a pass."""
    ...

def _fixed_clip_result(mpu, span, label_m, source, name='v.mp4', ci_sigma_multiple=None):
    """A clip result whose label resolution carries a chosen coverage."""
    ...

def test_the_individual_band_is_quoted_at_the_coverage_its_clips_declared():
    """``ci_sigma_multiple`` was dropped at fusion: the clip table and the
    individual table of ONE output directory quoted two different coverages,
    and the individual asserted the one it had not used."""
    ...

def test_clips_that_declare_two_coverages_are_refused_not_averaged():
    ...

def test_the_observer_class_basis_says_when_it_was_assigned_by_absence():
    """``entry_user`` was written unconditionally, beside a reason sentence
    saying the opposite -- on all 111 fit-capable individuals."""
    ...

def test_clause_6_refuses_a_pair_whose_video_it_cannot_identify():
    """FAIL CLOSED.  ``_clip_video`` falls back to the clip id, which always
    differs, so two windows of ONE video with no ``scale.source_name`` scored
    ``different_video_only`` and PASSED -- the configuration of every
    ratio-lane record, and exactly what clause 6 exists to refuse."""
    ...

def test_clause_2_is_scored_on_the_pair_clause_6_admits():
    """The two clauses were scored on DIFFERENT pairs: the separation max ran
    over every (i, j) while clause 6 passed on any ONE independent pair, so
    ``g_x`` could be observed on the geometry of a pair the record itself
    marks ``independent: false``."""
    ...

def test_a_record_with_no_matrix_says_the_gated_pair_is_unknown():
    """Back-compat: every record fitted before task C2 publishes one scalar and
    no pairing, and at N >= 3 the pair it belongs to cannot be recovered."""
    ...

def test_clause_7_reads_the_key_its_producer_will_write():
    """One key, two payloads, at two levels: the FIT record's per-station
    FRACTIONS are ``stations_in_frame_frac`` and the FUSED block's
    ``stations_observed`` is this clause's own verdict.  The collision between
    them forced merge repair cdf2f64; the producer (F1) writes the new name."""
    ...

def test_a_scale_the_ladder_cannot_see_is_refused_not_published():
    """``_scale_block`` published ``state: "bounded"`` with the record's own
    number and the uncited [0.10, 0.20] band beside a ``label_resolution`` that
    says ``unmet`` -- one block, two halves, disagreeing.  Unreachable while
    every label is a filename token; reachable the moment D1(ii)'s field-sheet
    or FinID rung scales a record whose file name carries nothing."""
    ...

def test_the_coverage_flag_reaches_the_band_it_names():
    """A coverage the caller asks for must survive to the published band.

    ``--ci-sigma-multiple`` would otherwise be overridden by whatever coverage
    the RECORDS were written at, because ``_fuse_scale`` reads the multiple off
    the clips: the flag has to re-resolve the clip's own rows, not be dropped
    the moment a record carries a resolution of its own.
    """
    ...
