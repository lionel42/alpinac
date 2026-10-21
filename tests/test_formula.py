import numpy as np
import pytest

from alpinac.elements import ELEMENTS, INDEX, ISOTOPE_INDEX, SYMBOLS
from alpinac.formula import (
    charge,
    dbe,
    format_formula,
    format_isotopologue,
    formula_mass,
    ion_mass,
    parse_formula,
    parse_tokens,
    remove_adduct_elements,
    same_formula,
    senior_rule_ii,
)


def test_elements_sorted_by_mass():
    masses = [e.mass for e in ELEMENTS]
    assert masses == sorted(masses)


def test_isotopes_most_abundant_first():
    for e in ELEMENTS:
        ab = [i.abundance for i in e.isotopes]
        assert ab == sorted(ab, reverse=True)
    assert [i.label for i in ELEMENTS[INDEX["Cl"]].isotopes] == ["Cl", "[37Cl]"]
    assert ISOTOPE_INDEX["[13C]"] == (INDEX["C"], 1)


def test_valences():
    assert ELEMENTS[INDEX["C"]].valence == 4
    assert ELEMENTS[INDEX["Cl"]].valence == 1
    assert ELEMENTS[INDEX["S"]].valence == 6
    assert ELEMENTS[INDEX["ad.H"]].valence == 2


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("CFCl3", {"C": 1, "F": 1, "Cl": 3}),
        ("H3C2F2Cl", {"C": 2, "H": 3, "F": 2, "Cl": 1}),
        ("(CF3)2CO", {"C": 3, "F": 6, "O": 1}),
        ("C((CH3)2)2", {"C": 5, "H": 12}),
        ("[13C]H4", {"C": 1, "H": 4}),
        ("CF3+", {"C": 1, "F": 3}),
    ],
)
def test_parse_formula(formula, expected):
    counts = parse_formula(formula)
    assert {SYMBOLS[i]: int(counts[i]) for i in np.flatnonzero(counts)} == expected


@pytest.mark.parametrize("bad", ["Xx2", "C2(", "C)2", "2C", "c"])
def test_parse_formula_errors(bad):
    with pytest.raises(ValueError):
        parse_formula(bad)


def test_parse_tokens_keeps_isotopes():
    assert parse_tokens("C[37Cl]Cl2") == {"C": 1, "Cl": 2, "[37Cl]": 1}


def test_charge():
    assert charge("CF3") == 0
    assert charge("CF3+") == 1
    assert charge("C2+2") == 2


def test_masses():
    # CF3+ = 12 + 3 * 18.99840316 - electron
    assert formula_mass("CF3+") == pytest.approx(68.9946609, abs=1e-7)
    assert formula_mass("CF3", ionised=True) == pytest.approx(68.9946609, abs=1e-7)
    assert formula_mass("CF3") == pytest.approx(68.9952095, abs=1e-7)
    assert ion_mass(parse_formula("CF3")) == pytest.approx(68.9946609, abs=1e-7)
    # isotopes are taken into account
    assert formula_mass("[13C]H4") - formula_mass("CH4") == pytest.approx(1.00335484)


def test_dbe():
    assert dbe(parse_formula("C6H6")) == 4
    assert dbe(parse_formula("CCl4")) == 0
    assert dbe(parse_formula("CF3")) == 0.5
    assert dbe(parse_formula("COS")) == 4


def test_senior_rule_ii():
    assert senior_rule_ii(parse_formula("CCl4"))
    assert not senior_rule_ii(parse_formula("CFCl"))
    assert not senior_rule_ii(parse_formula("SF"))
    assert senior_rule_ii(parse_formula("SF6"))


def test_format_formula_hill():
    assert format_formula(parse_formula("H3C2F2Cl")) == "C2H3ClF2"
    assert format_formula(parse_formula("NF3")) == "F3N"
    assert format_formula(parse_formula("Cl2")) == "Cl2"
    assert same_formula("H3CF", "CH3F")


def test_format_isotopologue():
    counts = parse_formula("CCl3F")
    rare = {INDEX["Cl"]: np.array([2, 1])}
    assert format_isotopologue(counts, rare) == "CCl2[37Cl]F"
    assert format_isotopologue(counts, None) == "CCl3F"


def test_remove_adduct_elements():
    counts = parse_formula("CH4")
    counts[INDEX["ad.H"]] = 1
    assert format_formula(remove_adduct_elements(counts)) == "CH5"
