"""Isotopologues of the candidates and their match to measured peaks (steps 3-4).

For a candidate made of abundant isotopes only, all isotopologues are
enumerated with their probability relative to the all-abundant one,
``p = prod_e multinomial(n_e; k_e0, k_e1, ...) * prod_j (a_ej / a_e0) ** k_ej``.

Only isotopologues with ``p >= LOD / I`` are kept, where ``I`` is the summed
area of the peaks the candidate was proposed for (the others would be below
the detection limit). As in the legacy code, if some ``p > 1`` (e.g. Br[81Br]
vs Br2), the intensities are normalised to the maximum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import comb

import numpy as np

from alpinac.elements import ELECTRON_MASS, ELEMENTS, MASS
from alpinac.formula import format_isotopologue
from alpinac.peaks import Compound


@lru_cache(maxsize=4096)
def _element_patterns(i_el: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All isotope patterns of ``n`` atoms of element ``i_el``.

    :return: counts (n_patterns, n_isotopes), relative probabilities and
        masses, sorted by decreasing probability (all-abundant first).
    """
    isotopes = ELEMENTS[i_el].isotopes
    k = len(isotopes)
    patterns: list[tuple[int, ...]] = []

    def rec(prefix: tuple[int, ...], left: int):
        if len(prefix) == k - 1:
            patterns.append((left,) + prefix)  # abundant isotope first
            return
        for c in range(left + 1):
            rec(prefix + (c,), left - c)

    if k == 1:
        patterns = [(n,)]
    else:
        rec((), n)
    counts = np.array(patterns, dtype=np.int64)
    a0 = isotopes[0].abundance
    ratio = np.array([iso.abundance / a0 for iso in isotopes])
    masses = np.array([iso.mass for iso in isotopes])
    probs = np.empty(len(counts))
    for i, c in enumerate(counts):
        p, left = 1.0, n
        for j in range(1, k):
            if c[j]:
                p *= comb(left, int(c[j])) * ratio[j] ** int(c[j])
                left -= int(c[j])
        probs[i] = p
    order = np.lexsort((np.arange(len(probs)), -probs))
    return counts[order], probs[order], counts[order] @ masses


def enumerate_isotopologues(
    formula: np.ndarray, min_rel: float
) -> tuple[list[dict[int, np.ndarray]], np.ndarray, np.ndarray]:
    """Isotopologues of ``formula`` with a relative probability >= ``min_rel``.

    :return: the isotope patterns (element index -> isotope counts, only for
        elements with several isotopes), the relative probabilities and the
        neutral masses, sorted by decreasing probability.
    """
    elements = [int(i) for i in np.flatnonzero(formula)]
    multi = [i for i in elements if len(ELEMENTS[i].isotopes) > 1]
    base_mass = float(sum(formula[i] * MASS[i] for i in elements if i not in multi))
    tables = [_element_patterns(i, int(formula[i])) for i in multi]
    # Maximal probability that the remaining elements can contribute
    max_rest = np.ones(len(multi) + 1)
    for j in range(len(multi) - 1, -1, -1):
        max_rest[j] = max_rest[j + 1] * tables[j][1].max()

    patterns: list[dict[int, np.ndarray]] = []
    probs: list[float] = []
    masses: list[float] = []
    chosen: list[int] = [0] * len(multi)

    def rec(level: int, p: float, m: float):
        if level == len(multi):
            patterns.append(
                {multi[j]: tables[j][0][chosen[j]] for j in range(len(multi))}
            )
            probs.append(p)
            masses.append(m)
            return
        counts, ps, ms = tables[level]
        for idx in range(len(ps)):
            q = p * ps[idx]
            if q * max_rest[level + 1] < min_rel:
                break  # patterns are sorted by decreasing probability
            chosen[level] = idx
            rec(level + 1, q, m + ms[idx])

    rec(0, 1.0, base_mass)
    probs_arr = np.array(probs)
    masses_arr = np.array(masses)
    order = np.argsort(-probs_arr, kind="stable")
    return [patterns[i] for i in order], probs_arr[order], masses_arr[order]


@dataclass
class Isotopologues:
    """The isotopologues of one candidate and their match to the measured peaks.

    Arrays with a spectrum axis have the shape (n_spectra, n_isotopologues).
    """

    formula: np.ndarray
    patterns: list[dict[int, np.ndarray]]
    rel: np.ndarray
    """Relative intensity of each isotopologue (max 1)."""
    mass: np.ndarray
    """m/z of each isotopologue (positive charge :attr:`charge`)."""
    is_abundant: np.ndarray
    """True for the isotopologue made of abundant isotopes only."""
    peak: np.ndarray
    """Index of the matched measured peak, -1 if none."""
    meas_max: np.ndarray
    """Measured area of the matched peak (upper bound of the assigned intensity), 0 if none."""
    belongs: np.ndarray
    """(n_spectra,) True if the candidate explains a peak of that spectrum."""
    charge: int = 1
    k_init: np.ndarray = field(default=None)  # type: ignore[assignment]
    k: np.ndarray = field(default=None)  # type: ignore[assignment]
    """(n_spectra,) scaling factor: fitted intensity = k * rel."""
    opt: np.ndarray = field(default=None)  # type: ignore[assignment]
    """Fitted intensity of each isotopologue: ``min(k * rel, meas_max)``."""

    def __post_init__(self):
        if self.k is None:
            self.init_k()

    @property
    def n(self) -> int:
        return len(self.rel)

    @property
    def mass_min(self) -> float:
        return float(self.mass.min())

    @property
    def mass_max(self) -> float:
        return float(self.mass.max())

    @property
    def total(self) -> float:
        """Total fitted intensity (all spectra)."""
        return float(self.opt.sum())

    def init_k(self):
        """Initial guess of k (alpinac step 4): ``sum(meas_max) / sum(rel)`` per spectrum.

        This replaces equation 4 of Guillevic et al., as in the legacy code.
        """
        self.k_init = self.meas_max.sum(axis=1) / self.rel.sum()
        self.set_k(self.k_init.copy())

    def set_k(self, k: np.ndarray):
        self.k = np.asarray(k, dtype=float)
        self.opt = np.minimum(self.rel[None, :] * self.k[:, None], self.meas_max)

    def labels(self) -> list[str]:
        return [format_isotopologue(self.formula, p) for p in self.patterns]

    @classmethod
    def build(
        cls,
        formula: np.ndarray,
        peaks: list[int],
        compound: Compound,
        pruning_factor: float = 1.1,
        charge: int = 1,
    ) -> Isotopologues:
        """Enumerate and match the isotopologues of a candidate.

        :param peaks: the measured peaks the candidate was proposed for
            (empty for constructed molecular ions).
        :param charge: charge of the ion.
        """
        formula = np.asarray(formula, dtype=np.int64)
        if peaks:
            lod_ratio = compound.lod / float(compound.area[peaks].sum())
        else:
            lod_ratio = 1.0
        patterns, rel, neutral = enumerate_isotopologues(
            formula, lod_ratio / pruning_factor
        )
        keep = rel >= lod_ratio
        patterns = [p for p, k in zip(patterns, keep) if k]
        rel, neutral = rel[keep], neutral[keep]
        if rel.max() > 1.0:
            rel = rel / rel.max()
        is_abundant = np.array(
            [all(c[1:].sum() == 0 for c in p.values()) for p in patterns]
        )
        mass = (neutral - charge * ELECTRON_MASS) / charge

        n_spec = compound.n_spectra
        peak = np.full((n_spec, len(rel)), -1, dtype=np.int64)
        meas_max = np.zeros((n_spec, len(rel)))
        belongs = np.zeros(n_spec, dtype=bool)
        for i_peak in peaks:
            belongs[compound.spectrum_id[i_peak]] = True

        for i_iso in range(len(rel)):
            if is_abundant[i_iso]:
                # matched to the peaks the knapsack proposed it for
                for i_peak in peaks:
                    s = compound.spectrum_id[i_peak]
                    peak[s, i_iso] = i_peak
                    meas_max[s, i_iso] = compound.area[i_peak]
                continue
            for s in np.flatnonzero(belongs):
                in_spec = np.flatnonzero(compound.spectrum_id == s)
                err = np.abs(compound.mass[in_spec] - mass[i_iso])
                best = int(np.argmin(err))
                if err[best] <= compound.mass_tol_isotope[in_spec[best]]:
                    peak[s, i_iso] = in_spec[best]
                    meas_max[s, i_iso] = compound.area[in_spec[best]]
        return cls(
            formula=formula,
            patterns=patterns,
            rel=rel,
            mass=mass,
            is_abundant=is_abundant,
            peak=peak,
            meas_max=meas_max,
            belongs=belongs,
            charge=charge,
        )
