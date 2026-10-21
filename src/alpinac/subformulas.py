"""Number of valid sub-formulas of a formula (denominator of the likelihood).

A sub-formula ``s`` of ``f`` is valid if ``0 < s <= f`` element-wise, its
DBE is >= 0 and its neutral mass is >= ``min_mass`` (the smallest detectable
mass). Elements with a valence <= 0 are ignored, as in the knapsack.
A single atom has exactly one sub-formula (itself).
"""

from __future__ import annotations

import numba
import numpy as np

from alpinac.elements import MASS, VALENCE


@numba.njit(cache=True)
def _count(n_max, masses, dbe_coef, min_mass):
    n_el = n_max.shape[0]
    if n_el == 0:
        return 0
    # Maximal DBE increase and maximal mass that elements i..end can add
    suffix_dbe = np.zeros(n_el + 1, dtype=np.int64)
    suffix_mass = np.zeros(n_el + 1)
    for i in range(n_el - 1, -1, -1):
        suffix_dbe[i] = suffix_dbe[i + 1] + max(dbe_coef[i], 0) * n_max[i]
        suffix_mass[i] = suffix_mass[i + 1] + n_max[i] * masses[i]
    counts = np.zeros(n_el, dtype=np.int64)
    cur_mass = np.zeros(n_el + 1)
    cur_dbe = np.zeros(n_el + 1, dtype=np.int64)
    cur_dbe[0] = 2
    total_atoms = np.zeros(n_el + 1, dtype=np.int64)
    level = 0
    counts[0] = n_max[0] + 1
    n = 0
    while level >= 0:
        counts[level] -= 1
        if counts[level] < 0:
            level -= 1
            continue
        m = cur_mass[level] + counts[level] * masses[level]
        d = cur_dbe[level] + counts[level] * dbe_coef[level]
        a = total_atoms[level] + counts[level]
        if level == n_el - 1:
            if a > 0 and d >= 0 and m >= min_mass:
                n += 1
            continue
        if m + suffix_mass[level + 1] < min_mass:
            counts[level] = 0
            continue
        if d + suffix_dbe[level + 1] < 0:
            continue
        cur_mass[level + 1] = m
        cur_dbe[level + 1] = d
        total_atoms[level + 1] = a
        level += 1
        counts[level] = n_max[level] + 1
    return n


def count_valid_subformulas(formula: np.ndarray, min_mass: float) -> int:
    """Number of valid sub-formulas of ``formula`` (see the module docstring)."""
    formula = np.asarray(formula, dtype=np.int64)
    if formula.sum() == 1:
        return 1
    idx = np.flatnonzero((formula > 0) & (VALENCE >= 1))
    # Heaviest first for the mass pruning
    idx = idx[np.argsort(-MASS[idx], kind="stable")]
    return int(
        _count(
            formula[idx].copy(),
            MASS[idx].copy(),
            (VALENCE[idx] - 2).astype(np.int64),
            float(min_mass),
        )
    )
