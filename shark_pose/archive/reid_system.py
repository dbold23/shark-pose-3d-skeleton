"""Re-identification system (S8 RID): one ranked list of candidate animals for an encounter, fused from cues.

Cues, each an encounter x encounter similarity with NaN where two encounters cannot be compared:

  fin      dorsal-fin trailing-edge profiles (``fin_edge``), symmetrised median of best matches. Side-free:
           the silhouette of the fin is the same from either side.
  pigment  flank crop embeddings, compared ONLY within the same flank (left with left, right with right): the
           two flanks carry different pigment, so a left-right comparison is not evidence. Per encounter and
           side, the re-normalised mean of the L2-normalised crop rows; similarity = the best cosine over the
           sides both encounters show.
  scar     scars placed on body coordinates (``body_coords``: u along the spine arc, theta round the body),
           compared only inside the region BOTH encounters saw. Kernel-weighted matches over
           sqrt(n_a * n_b) of the scars inside that region, so an unmatched scar in a region the other
           encounter saw counts against. NaN when the shared region holds no scar in either.
  shape    fitted 3D shape descriptors (S8 R1: length ratios + g_z). -Mahalanobis^2 under the WITHIN-ENCOUNTER
           covariance: windows of one encounter are one shark by construction, so the scale of shape noise
           is learnt without using a single cross-encounter identity label. R1 closed shape as a tie-breaker,
           hence its small weight.

Fusion (pre-registered in s8/reid/prereg.md; the constants below are those numbers): for each probe row and
cue, a robust z-score (median, 1.4826 MAD) over the gallery entries where that cue exists, provided at least
``MIN_GALLERY_FOR_Z`` do, clipped to +-``Z_CLIP``; fused score = sum over available cues of weight x z. A missing cue adds 0 (neither
for nor against). An entry sharing no cue with the probe scores -inf and ranks last. Candidate animals are
scored by their best encounter.

v2 (s8/reid/prereg_v2.md): per-part cues (``profile_cue`` with sides, ``notch_cue``, ``vector_cue``,
``embedding_cue``) grouped by body part (``CUE_PART``) and fused by ``fuse_parts``: a part's score is the mean z
of its cues, weighted by ``PART_WEIGHTS``. ``admit`` decides from a cue's own ranks on known links whether it
enters; ``crossfit_folds`` splits the linked animals so that choice is scored on animals it did not see.
``fuse`` (v1) is unchanged.

Nothing here changes an existing output. ``reid_bench.build_truth`` gained ``extra_links`` (default None,
bit-identical when None) so that links a person confirms from the review queue grow the benchmark.
"""
from __future__ import annotations
import collections
import csv
from dataclasses import dataclass, field
import numpy as np
from shark_pose.archive.body_coords import N_THETA, N_U
from shark_pose.archive.fin_edge import encounter_similarity_matrix
MIN_GALLERY_FOR_Z = 10
Z_CLIP = 5.0
SCAR_DU, SCAR_DTHETA = (0.05, 20.0)
SCAR_OTHER_CLASS = 0.5

@dataclass
class Cue:
    name: str
    encs: list[str]
    S: np.ndarray

def _nan_diag(S):
    ...

def _unit(X):
    ...

def profile_cue(name, details, encounters, sides=None) -> Cue:
    """Edge / line detail profiles (``fin_edge`` matcher). ``sides``: None for silhouettes (side-free), else a
    side per profile and only same-side profiles are compared (best over the sides both encounters show)."""
    ...

def fin_cue(details, encounters) -> Cue:
    ...

def embedding_cue(name, X, encounters, sides) -> Cue:
    """Appearance embeddings, same side only (a side 'any' is compared with 'any')."""
    ...

def pigment_cue(X, encounters, sides) -> Cue:
    ...

def notch_cue(name, notch_sets: dict) -> Cue:
    """``notch_sets[enc]``: pooled notch positions (``part_features.pool_notches``). Keys (enc, side) instead
    compare only the same side (a paired fin: the left and right pectoral are different fins), best over the
    sides both encounters show."""
    ...

@dataclass
class Scar:
    u: float
    theta: float

def scar_cell(u, theta):
    ...

def scar_pair_score(A: list[Scar], covA, B: list[Scar], covB) -> float:
    ...

def scar_cue(scars: dict[str, list[Scar]], coverage: dict[str, np.ndarray]) -> Cue:
    """``coverage[enc]``: (N_U, N_THETA) cells seen; ``scars[enc]``: its scars (may be empty = seen clean)."""
    ...

def shape_cue(F, encounters, name: str='shape') -> Cue:
    ...

@dataclass
class Fused:
    encs: list[str]
    S: np.ndarray
    z: dict[str, np.ndarray]
    weights: dict[str, float]

def robust_z(v):
    ...

def _cue_z(c: Cue, pos: dict, n: int, min_gallery: int) -> np.ndarray:
    """Per-probe-row robust z of one cue on the union encounter list (NaN where it does not contribute)."""
    ...

def fuse(cues: list[Cue], weights: dict | None=None, min_gallery: int=MIN_GALLERY_FOR_Z) -> Fused:
    ...

def single(cue: Cue, min_gallery: int=MIN_GALLERY_FOR_Z) -> Fused:
    """One cue through the same z-scoring, for comparing each cue with the fusion on equal terms."""
    ...
ADMIT_MIN_PROBES = 5
ADMIT_CHANCE_X = 5.0
ADMIT_FLOOR = 0.1
CROSSFIT_SEED = 20260924

def vector_cue(name, F, encounters) -> Cue:
    """Fixed-length descriptors (e.g. fin shape): -Mahalanobis^2 under the within-encounter covariance."""
    ...

def fuse_parts(cues: list[Cue], part_weights: dict | None=None, cue_part: dict | None=None, min_gallery: int=MIN_GALLERY_FOR_Z) -> Fused:
    """Score = sum over parts of weight x (mean z over that part's cues present for the pair). A cue whose part
    has no weight, or that is not in ``cue_part``, is ignored; its z is still reported."""
    ...

def chance_rank10(probe: dict) -> float:
    """Chance of a random ranking putting a mate in the top 10 (encounter level): 1 - C(g-m,10)/C(g,10)."""
    ...

def admit(probes: list[dict]) -> dict:
    """Pre-registered admission of one cue from its OWN single-cue ranks on known links."""
    ...

def crossfit_folds(animals, seed: int=CROSSFIT_SEED) -> dict[str, int]:
    """Linked animal -> fold 0/1, balanced, seeded: cue admission and best-single choice are made on one fold
    and scored on the other, so the fused estimate is not flattered by choosing on the same probes."""
    ...

def rank_animals(fused: Fused, probe: str, encounter_animal: dict[str, str], *, k: int=10, sex: dict[str, str] | None=None, exclude_same_animal: bool=True) -> list[dict]:
    """Top-k candidate animals for one probe encounter, each with its best encounter and the per-cue z there.

    Encounters absent from ``encounter_animal`` (new, unassigned) are candidates under their own name.
    ``sex``: encounter -> 'M'/'F'/''; a known conflicting sex removes the candidate.
    """
    ...

def confirmed_links(decisions_csv: str) -> list[tuple[str, str]]:
    """Rows of a filled review sheet with decision 'same' -> (probe encounter, candidate's encounter) pairs."""
    ...
