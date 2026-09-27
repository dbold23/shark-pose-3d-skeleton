"""Re-identification benchmark from the ledger's known cross-encounter links.

The ledger links 49 of 1 739 non-quarantined animals across more than one
encounter (tag id, code pointer, shared name, or one key reused across
encounters). Those links are the only identity truth the archive holds, so they
are the benchmark: for every probe encounter of a linked animal, rank all OTHER
encounters by descriptor similarity and record where the first encounter of the
same animal lands.

Rules the scorer enforces, because each one was a leak somewhere else:

* The probe's own encounter is never in its gallery. Windows of one video (and
  videos of one encounter) share day, light, range and camera; scoring them as
  matches measures the day, not the shark (S8 R1: rank 1 rises from 52/555 to
  290/923 when same-video windows are allowed).
* The gallery holds every embedded encounter, linked or not. An unlinked
  encounter ranked above the true one may itself be the same shark, so ranks are
  PESSIMISTIC; the top unlinked hits are written out as a review queue instead of
  being called errors.
* Descriptors are compared per encounter (the unit the ledger links), built as the
  re-normalised mean of the encounter's L2-normalised rows.

Input embedding table: CSV whose first column is ``video_name`` and whose
remaining numeric columns are the descriptor, one or more rows per video (for
example one per frame or per window), or an ``.npz`` with ``names`` (video names)
and ``X`` (rows x dims).
"""
from __future__ import annotations
import csv
import collections
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class Truth:
    """video -> encounter -> animal, with the covariates the strata need."""
    video_encounter: dict[str, str]
    encounter_animal: dict[str, str]

def build_truth(ledger_dir: str | Path, extra_links: list[tuple[str, str]] | None=None) -> Truth:
    """Read ``animals.csv`` and ``videos.csv`` from a ledger directory.

    ``extra_links``: (encounter, encounter) pairs a person confirmed as one shark (``reid_system.confirmed_links``);
    their animals are merged. None (the default) leaves the ledger's truth exactly as it is.
    """
    ...

def _merge_links(enc_animal, meta, linked, kind, pairs):
    ...

def load_embeddings(path: str | Path) -> tuple[list[str], np.ndarray]:
    ...

def _unit(X: np.ndarray) -> np.ndarray:
    ...

def encounter_descriptors(truth: Truth, names: list[str], X: np.ndarray):
    """Mean of L2-normalised rows per encounter, re-normalised. Unknown videos are counted, not used."""
    ...

def evaluate(truth: Truth, names: list[str], X: np.ndarray, *, sex_filter: bool=False, review_k: int=5) -> dict:
    """Rank every other encounter for each probe encounter of a linked animal.

    ``sex_filter``: drop gallery encounters whose known sex conflicts with the probe's
    known sex (unknown sex never conflicts).
    """
    ...

def evaluate_similarity(truth: Truth, encs: list[str], S: np.ndarray, *, sex_filter: bool=False, review_k: int=5, coverage: dict | None=None) -> dict:
    """Same scoring from a precomputed encounter x encounter similarity (higher = more alike).

    For matchers that are not a fixed-length vector (e.g. shift-tolerant fin-edge alignment).
    Entries may be -inf where two encounters share no comparable view; they rank last.
    """
    ...
