"""End-to-end identification on the reference set (regression test).

The expected ranks were produced by ``benchmarks/run_reference.py`` and
validated against the legacy code (see PLAN.md, step 14).
"""

import math

import pandas as pd
import pytest

from conftest import DATA, FRAGMENTS
from alpinac import IdentificationConfig, identify, identify_file
from alpinac.formula import same_formula
from alpinac.io import read_peaks
from alpinac.peaks import split_compounds

EXPECTED = pd.read_csv(DATA / "expected_reference_ranks.csv")


@pytest.mark.parametrize(
    "file", sorted(EXPECTED["file"].unique()), ids=lambda f: f.split(".frag.")[-1]
)
def test_reference_ranks(file):
    results = identify_file(FRAGMENTS / file)
    for row in EXPECTED[EXPECTED["file"] == file].itertuples():
        r = results[row.compound]
        rank = next(
            (m.rank for m in r.molecular_ions if same_formula(m.formula, row.true)),
            None,
        )
        expected = (
            None
            if (isinstance(row.rank_true, float) and math.isnan(row.rank_true))
            else int(row.rank_true)
        )
        assert rank == expected
        if isinstance(row.best, str) and row.best:
            assert r.best.formula == row.best


def test_result_content():
    peaks = split_compounds(
        read_peaks(FRAGMENTS / "200602.1550.std.7.frag.train_CFC11.txt")
    )[0]
    r = identify(peaks)
    assert r.ok and r.best.formula == "CCl3F"
    assert r.signal_percent > 90
    # every peak appears in the peak table
    assert set(r.peaks_df()["peak"]) == set(range(len(peaks)))
    frags = r.fragments_df()
    assert list(frags["rank"]) == list(range(1, len(frags) + 1))
    assert frags["likelihood"].is_monotonic_decreasing
    # the assigned area never exceeds the measured one
    df = r.peaks_df()
    assert (
        df.groupby("peak")["assigned_area"].max()
        <= df.groupby("peak")["area"].first() + 1e-9
    ).all()


def test_target_elements_restricts():
    peaks = split_compounds(
        read_peaks(FRAGMENTS / "200602.1550.std.7.frag.train_CFC11.txt")
    )[0]
    r = identify(peaks, target_elements="CClF")
    for f in r.fragments:
        assert "H" not in f.formula and "O" not in f.formula


def test_target_formula():
    peaks = split_compounds(
        read_peaks(FRAGMENTS / "200602.1550.std.7.frag.train_CCl4.txt")
    )[0]
    r = identify(peaks, IdentificationConfig(target_formula="CCl4"))
    assert r.best.formula == "CCl4"
    assert all(set(f.formula) <= set("CCl1234") for f in r.fragments)


def test_no_solution():
    peaks = split_compounds(
        read_peaks(FRAGMENTS / "200602.1550.std.7.frag.train_CFC11.txt")
    )[0]
    r = identify(peaks, target_elements="Si")
    assert r.status in ("no_solution", "ok")
    r = identify(peaks.iloc[:3], target_elements="B")
    assert r.status == "no_solution"


def test_ci_and_ei_spectra():
    r = identify_file(DATA / "frags_Xe.csv", target_elements="CClHXe")[3]
    assert r.ok
    assert {m.spectrum for m in r.molecular_ions} <= {"0", "1"}
