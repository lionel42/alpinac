"""Step 1: enumerate the chemical formulas matching a measured mass (knapsack).

All formulas made of the allowed elements (most abundant isotopes only) are
generated, such that

- the neutral mass is within ``[mass_min, mass_max]``,
- the double bond equivalent is >= 0, i.e. ``2 + sum(n_e * (v_e - 2)) >= 0``,
- each count is at most its maximum (from the target formula or from the mass).

Elements with a valence <= 0 (noble gases, metals in the table) never form
bonds and are not enumerated, as in the legacy code.

Adduct pseudo-elements (valence 2) may be part of the allowed elements for CI
spectra; :func:`resolve_adducts` converts them back to real elements.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from alpinac.config import IdentificationConfig
from alpinac.elements import IS_ADDUCT, MASS, N_ELEMENTS, VALENCE
from alpinac.formula import parse_elements, parse_formula, remove_adduct_elements


@numba.njit(cache=True)
def _enumerate(masses, dbe_coef, n_max, lo, hi):
    """Enumerate count vectors with ``lo <= sum(n*m) <= hi`` and ``2 + sum(n*c) >= 0``.

    Elements must be sorted by decreasing mass. Returns an (n, E) int array.
    """
    n_el = masses.shape[0]
    out = np.zeros((64, n_el), dtype=np.int64)
    n_out = 0
    if n_el == 0:
        return out[:0]

    # Suffix bounds for pruning: the maximal mass, and the maximal DBE
    # increase, that the elements i..end can still add.
    suffix_mass = np.zeros(n_el + 1)
    suffix_dbe = np.zeros(n_el + 1, dtype=np.int64)
    for i in range(n_el - 1, -1, -1):
        suffix_mass[i] = suffix_mass[i + 1] + n_max[i] * masses[i]
        suffix_dbe[i] = suffix_dbe[i + 1] + max(dbe_coef[i], 0) * n_max[i]

    counts = np.zeros(n_el, dtype=np.int64)
    # Explicit DFS: at each level, the count of the element is decreasing
    # from its maximal possible value down to 0.
    cur_mass = np.zeros(n_el + 1)
    cur_dbe = np.zeros(n_el + 1, dtype=np.int64)
    cur_dbe[0] = 2
    level = 0
    counts[0] = min(n_max[0], int(math.floor((hi + 1e-12) / masses[0]))) + 1
    while level >= 0:
        counts[level] -= 1
        if counts[level] < 0:
            level -= 1
            continue
        m = cur_mass[level] + counts[level] * masses[level]
        d = cur_dbe[level] + counts[level] * dbe_coef[level]
        if level == n_el - 1:
            if m >= lo and m <= hi and d >= 0:
                total = 0
                for j in range(n_el):
                    total += counts[j]
                if total > 0:
                    if n_out == out.shape[0]:
                        bigger = np.zeros((2 * n_out, n_el), dtype=np.int64)
                        bigger[:n_out] = out
                        out = bigger
                    out[n_out] = counts
                    n_out += 1
            continue
        # Prune: the remaining elements cannot reach the lower bound,
        # or cannot bring the DBE back to >= 0.
        if m + suffix_mass[level + 1] < lo:
            # Counts are decreasing: lower counts will not reach lo either.
            counts[level] = 0
            continue
        if d + suffix_dbe[level + 1] < 0:
            continue
        cur_mass[level + 1] = m
        cur_dbe[level + 1] = d
        level += 1
        remaining = hi - m
        counts[level] = min(n_max[level], int(math.floor((remaining + 1e-12) / masses[level]))) + 1
    return out[:n_out]


class FormulaSpace:
    """The allowed elements and their maximal counts for the enumeration."""

    def __init__(self, n_max: np.ndarray):
        n_max = np.asarray(n_max, dtype=np.int64)
        usable = (n_max > 0) & (VALENCE >= 1)
        idx = np.flatnonzero(usable)
        # Heaviest elements first (better pruning)
        idx = idx[np.argsort(-MASS[idx], kind="stable")]
        self.elements = idx
        self.n_max_all = n_max
        self._masses = MASS[idx].copy()
        self._dbe_coef = (VALENCE[idx] - 2).astype(np.int64)
        self._n_max = n_max[idx].copy()

    def enumerate(self, mass_min: float, mass_max: float) -> np.ndarray:
        """All formulas of the space with a neutral mass in ``[mass_min, mass_max]``.

        :return: (n, N_ELEMENTS) element counts.
        """
        sub = _enumerate(self._masses, self._dbe_coef, self._n_max, float(mass_min), float(mass_max))
        out = np.zeros((sub.shape[0], N_ELEMENTS), dtype=np.int64)
        out[:, self.elements] = sub
        return out

    @classmethod
    def from_target(
        cls,
        mass_max: float,
        elements: str | None,
        target_formula: str | None,
        adduct: np.ndarray | None = None,
    ) -> FormulaSpace:
        """Maximal counts as in the legacy ``define_knapsack_target``.

        With a target formula, its counts are the maxima. Otherwise each
        element of ``elements`` may appear as often as the mass allows.
        Adduct pseudo-element counts (CI) are added to the maxima.
        """
        if target_formula is not None:
            n_max = parse_formula(target_formula)
        else:
            mask = parse_elements(elements)
            n_max = np.where(mask, np.floor(mass_max / MASS), 0).astype(np.int64)
        n_max = n_max.copy()
        if adduct is not None:
            n_max += np.where(IS_ADDUCT, adduct, 0)
        return cls(n_max)


def resolve_adducts(formulas: np.ndarray) -> np.ndarray:
    """Convert adduct pseudo-elements back to real elements and remove duplicates.

    The order of first appearance is kept.
    """
    if formulas.shape[0] == 0 or not formulas[:, IS_ADDUCT].any():
        return formulas
    real = remove_adduct_elements(formulas)
    _, first = np.unique(real, axis=0, return_index=True)
    return real[np.sort(first)]


def knapsack(
    mass_min: np.ndarray,
    mass_max: np.ndarray,
    group: np.ndarray,
    group_adduct: list[np.ndarray],
    config: IdentificationConfig,
) -> list[np.ndarray]:
    """Step 1 for all measured masses of a compound.

    :param mass_min, mass_max: neutral mass windows of the measured peaks.
    :param group: index of the adduct group of each peak (peaks of spectra
        with the same maximal adduct share the same element bounds).
    :param group_adduct: adduct pseudo-element counts of each group.
    :return: for each peak, the (n, N_ELEMENTS) candidate formulas
        (adduct pseudo-elements already resolved).
    """
    solutions: list[np.ndarray] = [None] * len(mass_min)  # type: ignore[list-item]
    for i_group, adduct in enumerate(group_adduct):
        idx = np.flatnonzero(group == i_group)
        if len(idx) == 0:
            continue
        # The bounds come from the heaviest peak (peaks are sorted by mass)
        heaviest = idx[-1]
        space = FormulaSpace.from_target(
            mass_max[heaviest], config.elements, config.target_formula, adduct
        )
        for i in idx:
            solutions[i] = resolve_adducts(space.enumerate(mass_min[i], mass_max[i]))
    return solutions
