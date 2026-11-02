from itertools import product

import numpy as np
import pytest
from conftest import legacy_oracle

from alpinac.elements import VALENCE
from alpinac.formula import dbe2, mass, parse_formula
from alpinac.subformulas import count_valid_subformulas


def brute_force(f, min_mass):
    idx = [i for i in np.flatnonzero(f) if VALENCE[i] >= 1]
    n = 0
    for combo in product(*[range(f[i] + 1) for i in idx]):
        s = np.zeros_like(f)
        s[idx] = combo
        if s.sum() and dbe2(s) >= 0 and mass(s) >= min_mass:
            n += 1
    return n


@pytest.mark.parametrize(
    "formula", ["CCl4", "C2H3ClF2", "C4N", "C6F14", "SF6", "C7H8", "HCl"]
)
@pytest.mark.parametrize("min_mass", [0.0, 23.0, 50.0])
def test_count_matches_brute_force(formula, min_mass):
    f = parse_formula(formula)
    assert count_valid_subformulas(f, min_mass) == brute_force(f, min_mass)


def test_single_atom():
    assert count_valid_subformulas(parse_formula("F"), 23.0) == 1


def test_count_matches_legacy(reference_file):
    """Same as legacy, except formulas without mono-valent atoms (PLAN.md, change 13)."""
    for ref in legacy_oracle(reference_file):
        for formula, node in ref["nodes"].items():
            f = parse_formula(formula)
            expected = node["no_all_subfragments"]
            got = count_valid_subformulas(f, ref["min_measured_mass"])
            if got != expected:
                # legacy sometimes ignored the minimal mass for these
                assert not np.any((f > 0) & (VALENCE == 1))
                assert count_valid_subformulas(f, 0.0) == expected
