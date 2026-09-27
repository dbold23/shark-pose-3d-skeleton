"""THE clip_meta.json CONTRACT -- the label and the pass (boxfix, 2026-09-14).

``clip_meta.json`` is the sidecar the window prep writes and ``--clip-meta``
hands to the fit.  It carried ``source_video``, ``start_frame`` and ``fps``, and
two whole record blocks were silently degraded by what it did NOT carry.

* **THE LABEL.**  ``parse_length_label`` reads a length token off the file NAME.
  About a third of this corpus's names carry a unit-less number
  (``AN18112502_16_F_05.mp4``, ``AN1911070604_F_13_camera.mp4``, ``'145'`` =
  14.5 ft) that it cannot read, so ``scale.state`` came out ``unmet`` and TL,
  volume and mass were ABSENT -- on individuals whose label the ledger resolves
  without ambiguity.  97 box-fitted records are in that state.  Plan row D1(ii).
* **THE PASS.**  ``aggregate._clip_independence`` (gate clause 6) has read
  ``summary.time.pass_id`` and ``summary.time.creation_time`` since it was
  written and **nothing ever wrote them**, so the clause fell back to "different
  video" on every record in the corpus and AN14131001, AN15100103 and
  AN22121502 -- whose videos the ledger puts in ONE pass -- published ``g_x`` as
  observed.

Both are properties of the SIDECAR, not of the fit, which is why both are
repairable at aggregation with no refit (``repair_summary_from_clip_meta``).

The invariant that makes the change safe to ship over 100+ existing records:
**a sidecar that carries neither group changes nothing.**  ``test_bit_identity_*``
below are that invariant, asserted on the JSON bytes.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from shark_pose.morphometrics.aggregate import CLAUSE6_BASIS_LADDER, CLIP_PASS_GAP_S, _fuse_scale, clip_label_resolution, fuse_girth, label_ladder_knobs
from shark_pose.morphometrics.extract import Record
from shark_pose.sim2real import world_frame as wf
SPAN = 4.545244216918945

def _ledger(value_m, state='bounded', source='ledger_resolved', **extra):
    """A clip_meta sidecar carrying the ledger's resolved label."""
    ...

def test_a_sidecar_without_label_fields_says_nothing():
    ...

def test_a_length_with_no_named_source_is_refused_loudly():
    """A scale with no provenance is what the label ladder exists to refuse, so
    it raises rather than quietly becoming an anonymous row."""
    ...

def test_an_unknown_label_state_is_refused():
    ...

def test_a_label_state_with_no_value_is_refused():
    ...

def test_a_non_positive_sigma_is_refused():
    ...

def test_unmet_is_carried_as_a_stated_absence_not_a_missing_field():
    ...

def test_an_implausible_sidecar_label_is_not_used_as_a_scale():
    ...

def test_the_ledger_label_shares_the_field_estimate_group():
    """It is a RECONCILIATION of the four field sources, never a new
    observation, so it must not be able to raise n_independent_groups."""
    ...

def test_bit_identity_when_the_filename_token_parses():
    """THE SHIP CONDITION. A window whose name carries a readable token gets the
    block it has always got -- same keys, same bytes -- whether or not a
    clip_meta is passed, as long as that sidecar carries no label."""
    ...

def test_bit_identity_for_an_unlabelled_clip_with_no_sidecar_label():
    ...

def test_a_unitless_filename_is_rescued_by_the_sidecar_label():
    """The defect, fixed: 97 box records read `unmet` on a name like this."""
    ...

def test_an_agreeing_pair_votes_once_and_the_token_wins():
    ...

def test_under_the_bar_the_ledger_wins_and_both_are_recorded():
    """3.35 ft/4 % apart: under the 5 % contested bar, so an absolute is still
    published -- at the LEDGER's value, not at the mean of the two, which is a
    number neither source states."""
    ...

def test_over_the_bar_the_pair_is_contested_and_publishes_no_absolute():
    ...

def test_a_ledger_contested_individual_is_contested_whatever_the_filename_says():
    """The ledger refused this individual's label before the fit ever ran; a
    parseable file name does not get to overrule it."""
    ...

def test_a_ledger_unmet_sidecar_falls_back_to_the_filename_token():
    ...

def test_neither_route_has_a_label_and_the_block_names_both():
    ...

def test_the_sidecar_sigma_replaces_the_may_class_band():
    ...

def test_scale_resolve_is_untouched_with_both_knobs_at_their_defaults():
    ...

def test_a_preference_that_names_no_row_is_recorded_as_not_applied():
    ...

def _clip_result(scale):
    ...

def test_fuse_scale_keeps_the_clips_preferred_source():
    """Without this the individual re-resolves the pooled rows with no
    preference and publishes the MEDIAN of the two -- a third number, which
    neither the clip table nor the ledger states."""
    ...

def test_fuse_scale_refuses_two_different_preferences():
    ...

def test_every_rebuild_path_carries_the_knobs_the_clip_declared():
    """A block resolved WITH a preference must be re-resolved WITH it. Rebuilt
    without, the same rows return the group median -- on two disagreeing rows a
    number neither source said -- at the May class band."""
    ...

def test_the_knobs_refuse_two_animals_worth_of_answers():
    ...

def test_a_sidecar_without_pass_fields_stamps_nothing():
    ...

def test_an_unparseable_creation_time_is_refused_at_load_not_at_read():
    """``aggregate._creation_time_s`` returns None on anything it cannot parse,
    and a silent None there is a clause-6 fallback nobody ordered."""
    ...

def test_a_creation_time_with_no_convention_is_refused():
    ...

def test_time_block_stamps_the_pass_when_the_sidecar_names_one():
    ...

def _sep_matrix(clips):
    """Every pair 46.529 deg apart -- the pairwise block a record fitted after
    task C2 carries.  WITHOUT it ``_independent_separation`` never restricts,
    the gated amplification is the published one whatever clause 6 said, and a
    clause-6 refusal cannot be told from a rank-1 record: the assertion below
    that a refusal cites CLAUSE 6 is only real because this block is here."""
    ...

def _girth_record(clip, video, t0_s=0.0, n_frames=300, fps=60.0, creation_time=None, pass_id=None, convention=None, clips=('a', 'b'), n_clips=2):
    ...

def test_two_videos_of_one_pass_are_refused_under_and_and_admitted_under_or():
    """AN14131001's configuration, and the defect this contract closes: two
    DIFFERENT videos whose ledger pass_id is the same one."""
    ...

def test_two_passes_are_independent_under_both_readings():
    ...

def test_a_creation_time_pass_is_used_when_no_pass_id_is_recorded():
    ...

def test_no_pass_information_publishes_different_video_only_and_claims_nothing():
    """Every record fitted before this contract. The clause falls back to the
    video test, SAYS so, and its pass is explicitly not a pass claim."""
    ...

def test_two_windows_of_one_video_are_refused_under_both_readings():
    ...

def test_the_basis_ladder_reports_the_weakest_test_any_pair_was_scored_on():
    """N = 3, AN15092101's shape: two windows of one video plus a third video
    that names no pass. The record must not advertise the strong basis one of
    its pairs happens to have."""
    ...

def test_the_two_readings_are_published_on_every_n_ge_2_record():
    ...

def _box_summary():
    """A box-fitted record's two blocks, exactly as the 97 on disk carry them:
    an unmet scale on an unreadable name, and a time block with no pass."""
    ...

def test_repair_gives_an_already_fitted_record_its_label_and_its_pass():
    ...

def test_repair_rebuilds_the_depth_cross_check_when_given_its_inputs():
    ...

def test_repair_with_an_empty_sidecar_changes_nothing():
    ...

def test_repair_of_a_record_whose_token_parses_leaves_the_scale_alone():
    ...

def test_two_clips_of_one_individual_may_not_carry_two_labels():
    ...

def test_a_malformed_sidecar_stops_the_fit_at_load(tmp_path):
    ...

def _ledger_header(name: str) -> list:
    ...

def test_the_contract_field_names_are_the_ledgers_own():
    """Checked against the LEDGER's own headers, not against a second copy of
    the same literals (review, 2026-09-14): the point of the spelling is that
    the prep can copy a column across without a rename table, and asserting the
    constants equal themselves stays green through the rename that breaks it.

    `videos.csv` spells the timestamp `creation_time_utc` and the pass
    `pass_id`.  The LABEL group is built from `individuals_tiers.csv`'s
    `scale_state` and `scale_distinct_values` -- NOT `label_union`, which is a
    third column, distinct from both that and `length_label_m`.
    """
    ...

def test_resource_stamp_names_the_cuda_keys_too():
    """gpu_box patch: on a CUDA build every torch.mps.* call raises and is
    swallowed, so a box-fitted record carried no accelerator memory at all."""
    ...

def test_a_clause_6_refusal_names_clause_6_and_not_rank():
    """`_g_x_stage` read the CLAUSE-6-GATED amplification to decide whether the
    record is rank 1, and the gate returns None whenever it admits no pair -- so
    a clause-6 refusal on a TWO-VIEW record published itself as "one view is
    rank 1 in (g_x, g_z)".  The verdict was right and the explanation named a
    clause that had not decided, on 9 of the 29 N >= 2 aggregations in morph_v2.
    """
    ...

def test_the_ledgers_contest_survives_a_sibling_window_that_parses():
    """FAR22102801: the ledger refuses the label (4.572 | 5.1816, 12.5 % apart)
    and one of its windows is named `..._17ft_...`.  The contested block carried
    NO label_resolution, so every rebuild path re-synthesised the filename token
    and republished the refused 5.1816 m as a bounded absolute."""
    ...

def test_the_contest_holds_even_when_the_two_rows_in_hand_agree():
    """The ledger reconciles sources this library never sees, so "the rows in
    hand agree" is no answer to "the ledger's own sources do not"."""
    ...

def test_a_contested_row_is_the_only_row_that_carries_a_refusal():
    """`label_row` gains a key only when there is a refusal to carry, so every
    row built without one is byte-identical to what it was before."""
    ...

def test_two_clocks_on_unlike_conventions_are_not_differenced():
    """A GoPro stamps the END of recording; the ledger records which convention
    each video follows.  `clip_meta_pass` refuses a timestamp with no
    convention -- and differencing two KNOWN but unlike conventions is that same
    error with the refusal skipped: one pass's start-stamped and end-stamped
    videos differ by the first video's own duration."""
    ...

def test_one_convention_on_both_clips_still_runs_the_pass_test():
    ...

def test_the_weaker_clause_6_reading_can_be_asked_for_not_only_read_about():
    """`rule` had one caller, which passed the default, and no test: mutating
    the adoption line to always take the weaker reading left the suite green
    but for one case.  It is now reachable from `fuse_girth`."""
    ...

def test_the_or_definition_names_its_own_exception():
    """A pair whose video cannot be identified is refused under BOTH readings,
    so a flat "OR" would describe something clause6_or does not compute."""
    ...

def test_a_clip_declaring_no_sigma_does_not_agree_with_one_that_does():
    """`label_ladder_knobs` compared only the DECLARING blocks, so a mixed pair
    was a singleton set: no raise, and the declaring clip's band was applied to
    the whole individual -- against its own 'one animal, one band'."""
    ...

def test_two_clips_may_not_carry_one_length_at_two_sigmas():
    ...

def test_the_ledger_route_can_cite_the_row_it_came_from():
    """The ledger route's provenance terminated at the FIELD NAME
    `clip_meta.length_label_m`, where the filename route names the video: a
    record could not say which sidecar, ledger build or row produced the length
    that drives TL, volume and mass."""
    ...

def test_the_pass_can_cite_the_ledger_row_too():
    ...

def test_a_repaired_scale_block_says_it_was_repaired():
    """`time.pass_source` has always testified to its own repair; the scale
    block was replaced with no marker, and the only record of it was the REPORT
    -- which only a cooperating caller keeps."""
    ...
