import numpy as np
import pandas as pd
import pytest
from conftest import DATA, legacy_oracle

from alpinac.io import read_jdx, read_peaks
from alpinac.peaks import Compound, split_compounds


def test_read_frag_file():
    peaks = read_peaks(DATA / "fragments" / "200602.1550.std.7.frag.train_CFC11.txt")
    assert len(peaks) > 5
    assert set(peaks["ionisation"]) == {"EI"}
    assert (peaks["adduct"] == "").all()
    assert peaks["compound"].dtype.kind == "i"


def test_read_file_with_adduct_and_spectrum_columns():
    peaks = read_peaks(DATA / "frags_Xe.csv")
    ei = peaks["ionisation"] == "EI"
    assert (peaks.loc[ei, "adduct"] == "").all()  # "None" means no adduct
    assert set(peaks.loc[~ei, "adduct"]) <= {"H", ""}
    assert (
        peaks["spectrum"].dtype.kind in "OUT"
        or peaks["spectrum"].map(type).eq(str).all()
    )
    # compound -1 (unassigned peaks) is skipped
    assert -1 not in split_compounds(peaks)


def test_compound_matches_legacy(reference_file):
    compounds = split_compounds(read_peaks(reference_file))
    for ref in legacy_oracle(reference_file):
        c = Compound.from_peaks(compounds[ref["compound"]])
        np.testing.assert_allclose(c.neutral_mass, ref["meas_mass"], rtol=1e-12)
        np.testing.assert_allclose(c.mass_U, ref["meas_mass_U_combined"], rtol=1e-12)
        np.testing.assert_allclose(
            c.mass_tol_isotope, ref["meas_mass_u_peak_width"], rtol=1e-12
        )
        np.testing.assert_allclose(c.area, ref["meas_I"])
        assert c.sigma_slope == pytest.approx(ref["sigma_m_slope"], rel=1e-9)
        assert c.sigma_intercept == pytest.approx(ref["sigma_m_b"], rel=1e-9)
        assert c.alpha == pytest.approx(ref["alpha_pseudo_voigt"])
        assert c.lod == pytest.approx(ref["LOD"])


def test_spectra_and_adducts():
    peaks = pd.DataFrame(
        {
            "mass": [50.1, 51.2, 52.3],
            "area": [100.0, 200.0, 300.0],
            "mass_u_ppm": 5.0,
            "peak_width": 0.005,
            "peak_alpha": 0.5,
            "ionisation": ["EI", "CI", "CI"],
            "adduct": ["", "H", "H"],
            "spectrum": ["ei", "ci", "ci"],
        }
    )
    c = Compound.from_peaks(peaks)
    assert c.spectra == ["ei", "ci"]
    assert list(c.spectrum_id) == [0, 1, 1]
    assert c.spectrum_adduct_formula == ["", "H"]
    assert c.spectrum_adduct[1].sum() == 1


def test_adduct_on_ei_is_rejected():
    peaks = pd.DataFrame(
        {
            "mass": [50.1],
            "area": [1.0],
            "mass_u_ppm": 5.0,
            "peak_width": 0.005,
            "peak_alpha": 0.5,
            "adduct": ["H"],
        }
    )
    with pytest.raises(ValueError, match="EI"):
        Compound.from_peaks(peaks)


def test_read_jdx():
    pytest.importorskip("jcamp")
    peaks, meta = read_jdx(DATA / "110-82-7-Mass.jdx")
    assert meta["formula"] == "C6H12"
    assert Compound.from_peaks(peaks).unit_mass
