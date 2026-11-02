import numpy as np
import pytest
from conftest import legacy_oracle

from alpinac.config import IdentificationConfig
from alpinac.elements import INDEX, N_ELEMENTS
from alpinac.formula import dbe2, format_formula, mass, parse_formula
from alpinac.io import read_peaks
from alpinac.knapsack import FormulaSpace, knapsack, resolve_adducts
from alpinac.peaks import Compound, split_compounds


def brute_force(n_max, lo, hi):
    """All formulas by brute force over small bounds."""
    from itertools import product

    from alpinac.elements import VALENCE

    idx = [i for i in np.flatnonzero(n_max) if VALENCE[i] >= 1]
    out = set()
    for combo in product(*[range(n_max[i] + 1) for i in idx]):
        f = np.zeros(N_ELEMENTS, dtype=int)
        f[idx] = combo
        if f.sum() and lo <= mass(f) <= hi and dbe2(f) >= 0:
            out.add(format_formula(f))
    return out


@pytest.mark.parametrize("lo, hi", [(0.5, 60.0), (40.0, 41.0), (100.0, 160.0)])
def test_enumeration_matches_brute_force(lo, hi):
    n_max = parse_formula("C4H6O2N2Cl3F2")
    space = FormulaSpace(n_max)
    got = {format_formula(f) for f in space.enumerate(lo, hi)}
    assert got == brute_force(n_max, lo, hi)


def test_noble_gases_are_not_enumerated():
    space = FormulaSpace.from_target(140.0, "CHXeAr", None)
    for f in space.enumerate(0.5, 140.0):
        assert f[INDEX["Xe"]] == 0 and f[INDEX["Ar"]] == 0


def test_target_formula_bounds():
    space = FormulaSpace.from_target(200.0, None, "CCl4")
    got = {format_formula(f) for f in space.enumerate(0.5, 200.0)}
    assert got == {"C", "CCl", "CCl2", "CCl3", "CCl4", "Cl", "Cl2"}


def test_adduct():
    # CH5 (CI with H adduct) has a negative DBE, it is only allowed with the adduct
    adduct = np.zeros(N_ELEMENTS, dtype=int)
    adduct[INDEX["ad.H"]] = 1
    m = mass(parse_formula("CH5"))
    without = FormulaSpace.from_target(m + 0.01, "CH", None)
    assert "CH5" not in {
        format_formula(f) for f in without.enumerate(m - 1e-4, m + 1e-4)
    }
    with_adduct = FormulaSpace.from_target(m + 0.01, "CH", None, adduct)
    found = resolve_adducts(with_adduct.enumerate(m - 1e-4, m + 1e-4))
    assert [format_formula(f) for f in found] == ["CH5"]


def test_knapsack_contains_legacy_solutions(reference_file):
    """All legacy solutions are found; the extra ones are valid (see PLAN.md, change 10)."""
    config = IdentificationConfig()
    compounds = split_compounds(read_peaks(reference_file))
    for ref in legacy_oracle(reference_file):
        c = Compound.from_peaks(compounds[ref["compound"]], config)
        sols = knapsack(
            c.mass_min,
            c.mass_max,
            np.zeros(c.n_peaks, int),
            [np.zeros(N_ELEMENTS, int)],
            config,
        )
        for i, (new, old) in enumerate(zip(sols, ref["knapsack"])):
            new = {format_formula(f): f for f in new}
            old = {format_formula(parse_formula(f)) for f in old}
            assert old <= set(new)
            for name in set(new) - old:
                f = new[name]
                assert c.mass_min[i] <= mass(f) <= c.mass_max[i]
                assert dbe2(f) >= 0
                assert f[INDEX["H"]] == 1
