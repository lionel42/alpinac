import numpy as np
import pytest

from alpinac.fitting import fit_profiles, mass_axis, pseudo_voigt


def test_pseudo_voigt_area():
    x = np.linspace(99, 101, 200001)
    # Lorentzian part: area A (the tails outside +-1 m/z hold ~0.6 %)
    y = pseudo_voigt(x, 3.0, 100.0, 0.005, 1.0)[0]
    assert np.trapezoid(y, x) == pytest.approx(3.0, rel=1e-2)
    # Gaussian part: legacy convention exp(-(x-mu)^2 / (2 sigma)^2) with the
    # 1 / (sigma sqrt(2 pi)) normalisation, i.e. an area of sqrt(2) * A
    y = pseudo_voigt(x, 3.0, 100.0, 0.005, 0.0)[0]
    assert np.trapezoid(y, x) == pytest.approx(3.0 * np.sqrt(2), rel=1e-6)


def test_mass_axis_legacy_points():
    x = mass_axis(100.2, 101.7, 8.0, 0.5)
    assert x[0] == 99.5 and x[-1] == 102.5
    assert len(x) == round(8 * (102.5 - 99.5 + 1))


def make_problem(offset=0.0):
    x = np.linspace(99.5, 102.5, 2000)
    sigma, alpha = 0.006, 0.5
    iso1 = pseudo_voigt(
        x, np.array([1.0, 0.3]), np.array([100.0, 101.0]), sigma, alpha
    ).sum(0)
    iso2 = pseudo_voigt(
        x, np.array([1.0, 0.1]), np.array([100.05, 102.0]), sigma, alpha
    ).sum(0)
    true_k = np.array([50.0, 20.0])
    peaks_m = np.array([100.0, 101.0, 100.05, 102.0]) - offset
    peaks_a = np.array([50.0, 15.0, 20.0, 2.0])

    def measured(o):
        return pseudo_voigt(x, peaks_a, peaks_m + o, sigma, alpha).sum(0)

    return x, np.array([iso1, iso2]), measured, true_k


def test_fit_recovers_k():
    x, profiles, measured, true_k = make_problem()
    k, off = fit_profiles(x, profiles, measured, 1e-3, None)
    np.testing.assert_allclose(k, true_k, rtol=1e-6)


def test_fit_recovers_offset():
    x, profiles, measured, true_k = make_problem(offset=0.002)
    k, off = fit_profiles(x, profiles, measured, 1e-3, 0.004)
    assert off == pytest.approx(0.002, abs=1e-5)
    np.testing.assert_allclose(k, true_k, rtol=1e-3)


def test_k_lower_bound():
    x, profiles, measured, _ = make_problem()
    k, _ = fit_profiles(
        x, profiles * np.array([[1.0], [0.0]]) + 0.0, measured, 5.0, None
    )
    assert k.min() >= 5.0


def test_same_as_lmfit():
    lmfit = pytest.importorskip("lmfit")
    x, profiles, measured, true_k = make_problem(offset=0.001)
    p = lmfit.Parameters()
    p.add("off", value=0.0, min=-0.004, max=0.004)
    p.add("k0", value=10.0, min=1e-3)
    p.add("k1", value=10.0, min=1e-3)
    res = lmfit.minimize(
        lambda p: (
            p["k0"].value * profiles[0]
            + p["k1"].value * profiles[1]
            - measured(p["off"].value)
        ),
        p,
    )
    k, off = fit_profiles(x, profiles, measured, 1e-3, 0.004)
    np.testing.assert_allclose(
        k, [res.params["k0"].value, res.params["k1"].value], rtol=1e-3
    )
