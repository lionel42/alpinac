"""Candidate fragment formulas and their sub-formula relations (step 2).

The legacy code organised the candidates in a networkx directed graph
(Hasse diagram of the "is a sub-formula of" order). Everything the
identification needs from that graph only depends on the order itself:
ancestors, descendants, maximal candidates and singletons. Here the order is
stored as a boolean dominance matrix ``below[i, j] = (formula j < formula i)``.
"""

from __future__ import annotations

import logging

import numba
import numpy as np

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def _dominance(formulas):
    """``out[i, j]`` is True if formula j is a strict sub-formula of formula i."""
    n, n_el = formulas.shape
    out = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            strict = False
            ok = True
            for e in range(n_el):
                a = formulas[i, e]
                b = formulas[j, e]
                if b > a:
                    ok = False
                    break
                if b < a:
                    strict = True
            out[i, j] = ok and strict
    return out


class CandidateSet:
    """Candidate formulas of a compound.

    :attr formulas: (n, N_ELEMENTS) element counts, abundant isotopes.
    :attr peaks: for each candidate, the indices of the measured peaks for
        which the knapsack proposed it (empty for constructed molecular ions).
    :attr below: dominance matrix, ``below[i, j]``: j is a strict sub-formula of i.
    """

    def __init__(self, formulas: np.ndarray, peaks: list[list[int]]):
        self.formulas = np.asarray(formulas, dtype=np.int64).reshape(len(peaks), -1)
        self.peaks = [list(p) for p in peaks]
        used = np.flatnonzero(self.formulas.any(axis=0))
        self.below = _dominance(np.ascontiguousarray(self.formulas[:, used]))

    def __len__(self) -> int:
        return len(self.peaks)

    @classmethod
    def from_knapsack(cls, solutions: list[np.ndarray]) -> tuple[CandidateSet, np.ndarray]:
        """Build the candidates from the knapsack solutions of each peak.

        Peaks are processed from the heaviest to the lightest (as the legacy
        graph construction). A formula proposed for several peaks ("twin")
        is a single candidate with several peaks.

        :return: the candidate set, and for each peak the number of formulas
            proposed by the knapsack.
        """
        index: dict[bytes, int] = {}
        formulas: list[np.ndarray] = []
        peaks: list[list[int]] = []
        n_per_peak = np.zeros(len(solutions), dtype=np.int64)
        for i_peak in range(len(solutions) - 1, -1, -1):
            sols = solutions[i_peak]
            n_per_peak[i_peak] = len(sols)
            for f in sols:
                key = f.tobytes()
                if key in index:
                    peaks[index[key]].append(i_peak)
                else:
                    index[key] = len(formulas)
                    formulas.append(f)
                    peaks.append([i_peak])
        if formulas:
            arr = np.array(formulas)
        else:
            arr = np.zeros((0, len(solutions[0]) if solutions else 0), dtype=np.int64)
        return cls(arr, peaks), n_per_peak

    # --- order relations -------------------------------------------------
    def n_descendants(self) -> np.ndarray:
        return self.below.sum(axis=1)

    def descendants(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.below[i])

    def ancestors(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.below[:, i])

    def maximal(self, among: np.ndarray | None = None) -> np.ndarray:
        """Indices of candidates without ancestor (within the subset ``among``)."""
        if among is None:
            return np.flatnonzero(~self.below.any(axis=0))
        among = np.asarray(among, dtype=np.int64)
        sub = self.below[np.ix_(among, among)]
        return among[~sub.any(axis=0)]

    def singletons(self) -> np.ndarray:
        """Candidates comparable to no other candidate."""
        return np.flatnonzero(~self.below.any(axis=0) & ~self.below.any(axis=1))

    def find(self, formula: np.ndarray) -> int | None:
        hits = np.flatnonzero((self.formulas == formula).all(axis=1))
        return int(hits[0]) if len(hits) else None

    # --- modifications ---------------------------------------------------
    def subset(self, keep: np.ndarray) -> CandidateSet:
        """A new set with only the candidates ``keep`` (indices, order kept)."""
        keep = np.asarray(keep, dtype=np.int64)
        new = CandidateSet.__new__(CandidateSet)
        new.formulas = self.formulas[keep]
        new.peaks = [list(self.peaks[i]) for i in keep]
        new.below = self.below[np.ix_(keep, keep)]
        return new

    def add(self, formula: np.ndarray) -> int:
        """Add a candidate without measured peak (constructed molecular ion)."""
        formula = np.asarray(formula, dtype=np.int64)
        f = np.vstack([self.formulas, formula[None, :]])
        used = np.flatnonzero(f.any(axis=0))
        row = _dominance(np.ascontiguousarray(f[:, used]))
        self.formulas = f
        self.peaks.append([])
        self.below = row
        return len(self.peaks) - 1

    def hasse_edges(self) -> list[tuple[int, int]]:
        """Edges of the fragmentation graph: (parent, child) with no candidate in between."""
        below = self.below
        edges = []
        for i in range(len(self)):
            children = np.flatnonzero(below[i])
            for j in children:
                # j is a direct child if no other child k of i has j below it
                if not below[children, j].any():
                    edges.append((i, int(j)))
        return edges


def remove_singletons(
    cands: CandidateSet, n_per_peak: np.ndarray | None = None
) -> tuple[CandidateSet, np.ndarray]:
    """Remove the singletons, except the last candidates of a peak.

    A candidate that is a sub-formula or super-formula of no other candidate
    is very unlikely, all fragments of a molecule being related. But if all
    candidates of a peak are singletons, they are kept (otherwise the peak
    would have no candidate left). A singleton proposed for several peaks
    loses the peaks for which other candidates exist.

    :param n_per_peak: number of candidates of each peak. If None it is
        computed from the current candidates.
    :return: the new set, and the indices (in the old set) of the removed candidates.
    """
    singles = cands.singletons()
    if len(singles) == 0:
        return cands, np.zeros(0, dtype=np.int64)
    n_peaks = 1 + max((max(p) for p in cands.peaks if p), default=-1)
    if n_per_peak is None:
        n_per_peak = np.zeros(n_peaks, dtype=np.int64)
        for p in cands.peaks:
            for i_peak in p:
                n_per_peak[i_peak] += 1
    # singletons of each peak
    singles_of_peak: dict[int, list[int]] = {}
    remove = set()
    for s in singles:
        if not cands.peaks[s]:
            remove.add(int(s))
        for i_peak in cands.peaks[s]:
            singles_of_peak.setdefault(i_peak, []).append(int(s))
    peaks = [list(p) for p in cands.peaks]
    for i_peak, nodes in singles_of_peak.items():
        if len(nodes) >= n_per_peak[i_peak]:
            continue  # only singletons for this peak: keep them
        for s in nodes:
            if len(peaks[s]) == 1:
                remove.add(s)
            else:
                peaks[s].remove(i_peak)
    cands.peaks = peaks
    removed = np.array(sorted(remove), dtype=np.int64)
    keep = np.setdiff1d(np.arange(len(cands)), removed)
    return cands.subset(keep), removed
