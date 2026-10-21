import numpy as np
import pytest

from conftest import legacy_oracle
from alpinac.formula import format_formula, parse_formula, parse_tokens
from alpinac.io import read_peaks
from alpinac.isotopes import enumerate_isotopologues, Isotopologues
from alpinac.peaks import Compound, split_compounds


def test_cl2_pattern():
    patterns, rel, mass = enumerate_isotopologues(parse_formula("Cl2"), 0.0)
    a = 0.242353 / 0.757647
    np.testing.assert_allclose(rel, [1.0, 2 * a, a**2])
    assert mass[1] - mass[0] == pytest.approx(36.9659026 - 34.96885271)


def test_threshold_prunes():
    _, rel, _ = enumerate_isotopologues(parse_formula("C6Cl6"), 0.01)
    assert (rel >= 0.01).all()
    _, rel_all, _ = enumerate_isotopologues(parse_formula("C6Cl6"), 0.0)
    assert len(rel) == (rel_all >= 0.01).sum()


def test_br2_normalised():
    peaks = read_peaks(next(iter(sorted(__import__("conftest").FRAGMENTS.glob("*CH2Br2.txt")))))
    c = Compound.from_peaks(peaks)
    iso = Isotopologues.build(parse_formula("Br2"), [], c)
    # Br[81Br] is more likely than Br2: the maximum is normalised to 1
    assert iso.rel.max() == pytest.approx(1.0)


def test_matches_legacy(reference_file):
    compounds = split_compounds(read_peaks(reference_file))
    for ref in legacy_oracle(reference_file):
        c = Compound.from_peaks(compounds[ref["compound"]])
        peaks_of = {format_formula(parse_formula(f)): p for f, p in ref["nodes_step2"]}
        for formula, node in ref["nodes"].items():
            f = parse_formula(formula)
            iso = Isotopologues.build(f, peaks_of[format_formula(f)], c)
            new = {frozenset(parse_tokens(l).items()): i for i, l in enumerate(iso.labels())}
            old = {frozenset(parse_tokens(l).items()): i for i, l in enumerate(node["iso_list"])}
            assert new.keys() == old.keys()
            for key, i in new.items():
                j = old[key]
                assert iso.rel[i] == pytest.approx(node["iso_list_I_rel"][j], rel=1e-9)
                legacy_peak = node["meas_mass_idx"][0][j]
                assert iso.peak[0, i] == (-1 if legacy_peak is None else legacy_peak)
                assert iso.meas_max[0, i] == pytest.approx(node["iso_list_I_meas_max"][0][j])
            assert iso.k_init[0] == pytest.approx(node["k_guess_init"][0], rel=1e-9)
            assert iso.total == pytest.approx(node["iso_list_I_sum"], rel=1e-9)
