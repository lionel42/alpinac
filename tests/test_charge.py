"""Doubly charged fragments (``IdentificationConfig.max_charge``)."""

import numpy as np
import pytest

from conftest import DATA
from alpinac import identify
from alpinac.candidates import CandidateSet
from alpinac.config import IdentificationConfig
from alpinac.elements import ELECTRON_MASS
from alpinac.formula import format_formula, formula_mass, parse_formula
from alpinac.io import read_peaks
from alpinac.isotopes import Isotopologues
from alpinac.knapsack import knapsack_charged
from alpinac.peaks import Compound

BENZENE = DATA / "benzene_ei.txt"


def benzene_peaks():
    peaks = read_peaks(BENZENE)
    return peaks[peaks["compound"] != -1]


def test_formula_mass_of_doubly_charged():
    assert formula_mass("C6H6++") == pytest.approx(
        (formula_mass("C6H6") - 2 * ELECTRON_MASS) / 2
    )
    # (2 f)++ and f+ have the same m/z
    assert formula_mass("C6H6++") == pytest.approx(formula_mass("C3H3+"))


def test_isotopologues_of_doubly_charged():
    c = Compound.from_peaks(benzene_peaks())
    p = int(np.argmin(np.abs(c.mass - formula_mass("C6H6++"))))
    iso = Isotopologues.build(parse_formula("C6H6"), [p], c, charge=2)
    assert iso.charge == 2
    assert iso.mass[0] == pytest.approx(formula_mass("C6H6++"))
    # the 13C isotopologue is half a mass unit higher
    assert iso.mass[1] - iso.mass[0] == pytest.approx(1.00335 / 2, abs=1e-4)


def test_knapsack_charged():
    c = Compound.from_peaks(benzene_peaks())
    p = int(np.argmin(np.abs(c.mass - formula_mass("C6H3++"))))
    eligible = np.zeros(c.n_peaks, dtype=bool)
    eligible[p] = True
    config = IdentificationConfig(target_formula="C6H6")
    sols = knapsack_charged(c.mass_min, c.mass_max, eligible, 2, config)
    assert [format_formula(f) for f in sols[p]] == ["C6H3"]
    assert all(len(s) == 0 for i, s in enumerate(sols) if i != p)


def test_charge_in_order_and_twins():
    f = np.array([parse_formula("C6H6"), parse_formula("C6H6"), parse_formula("C6H5")])
    c = CandidateSet(f, [[0], [1], [2]], np.array([1, 2, 1]))
    # same formula: the doubly charged ion is below the singly charged one
    assert set(c.descendants(0)) == {1, 2}
    assert set(c.descendants(1)) == {2}
    assert c.find(parse_formula("C6H6"), charge=2) == 1
    # same formula proposed with two charges: two candidates
    sols = [np.array([parse_formula("C3H3")])]
    charged = {2: [np.array([parse_formula("C6H6")])]}
    c, n_per_peak = CandidateSet.from_knapsack(sols, charged)
    assert list(c.charge) == [1, 2] and list(n_per_peak) == [2]


def test_benzene_doubly_charged():
    peaks = benzene_peaks()
    r = identify(peaks, target_formula="C6H6")
    assert r.best.formula == "C6H6"
    df = r.peaks_df()
    doubly = df[df["charge"] == 2]
    assert set(doubly["formula"]) >= {"C6H3++", "C5[13C]H6++"}
    # half-integer peaks are explained by doubly charged ions only
    half = (df["mass"] % 1 > 0.25) & (df["mass"] % 1 < 0.75)
    assert (df.loc[half & (df["formula"] != ""), "charge"] == 2).all()
    frag = r.fragments_df().set_index("formula")
    assert frag.loc["C6H6++", "charge"] == 2
    assert frag.loc["C6H6++", "mass"] == pytest.approx(formula_mass("C6H6++"))
    assert all(m.formula.count("+") == 0 for m in r.molecular_ions)

    r1 = identify(peaks, target_formula="C6H6", max_charge=1)
    assert (r1.peaks_df()["charge"] <= 1).all()
    assert r.signal_percent > r1.signal_percent


def test_no_doubly_charged_without_half_integer_peak():
    peaks = benzene_peaks()
    frac = peaks["mass"] % 1
    peaks = peaks[(frac < 0.25) | (frac > 0.75)]
    r = identify(peaks, target_formula="C6H6")
    assert (r.peaks_df()["charge"] <= 1).all()
