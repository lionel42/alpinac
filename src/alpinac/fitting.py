"""Fit of the isotopologue profiles to the measured mass profile (step 7).

The measured peaks of a mass window and the isotopologues of the candidates
are both turned into profiles along a mass axis (sum of pseudo-Voigt peaks).
The measured profile is modelled as ``sum_j k_j * profile_j``, with a
possible constant mass offset of the measured peaks.

For a fixed offset the model is linear in the ``k_j``: they are solved by
bounded linear least squares (``k_j >= k_min``). The offset is optimised by a
scan followed by a bounded scalar minimisation (variable projection).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar, nnls

C_PV = np.sqrt(2 * np.log(2))


def pseudo_voigt(x: np.ndarray, area, mu, sigma, alpha) -> np.ndarray:
    """Pseudo-Voigt peak(s) on ``x``.

    ``area``, ``mu`` and ``sigma`` may be arrays (one value per peak): the
    result then has the shape (n_peaks, len(x)).

    The Gaussian part uses the legacy convention ``exp(-(x-mu)**2 / (2 sigma)**2)``
    because the peak widths in the peak files were fitted with this function.
    """
    x = np.asarray(x)[None, :]
    area = np.atleast_1d(area)[:, None]
    mu = np.atleast_1d(mu)[:, None]
    sigma = np.atleast_1d(sigma)[:, None]
    gauss = (
        area
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((x - mu) ** 2) / (2 * sigma) ** 2)
    )
    sigma_l = sigma * C_PV
    lorentz = area / np.pi * sigma_l / ((x - mu) ** 2 + sigma_l**2)
    return (1 - alpha) * gauss + alpha * lorentz


def mass_axis(
    m_start: float, m_stop: float, points_per_mass: float, offset: float
) -> np.ndarray:
    """Mass axis of a window, as in the legacy ``do_meas_mass_profile_x``."""
    lo = round(m_start) - offset
    hi = round(m_stop) + offset
    return np.linspace(lo, hi, round(points_per_mass * (hi - lo + 1)))


def _solve_k(A: np.ndarray, b: np.ndarray, k_min: float) -> tuple[np.ndarray, float]:
    """min ||A k - b|| subject to k >= k_min."""
    shift = A @ np.full(A.shape[1], k_min)
    k, rnorm = nnls(A, b - shift)
    return k + k_min, rnorm


def fit_profiles(
    x: np.ndarray,
    iso_profiles: np.ndarray,
    measured: callable,
    k_min: float,
    offset_bound: float | None,
    k_init: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Fit ``measured(offset) ~ sum_j k_j * iso_profiles[j]``.

    :param iso_profiles: (n_candidates, len(x)) profiles of the isotopologue sets.
    :param measured: function offset -> measured profile on ``x``.
    :param k_min: lower bound of the k factors.
    :param offset_bound: the offset is in ``[-offset_bound, offset_bound]``;
        None for no offset.
    :param k_init: current k values (not needed by this method, which finds
        the optimum for each offset).
    :return: the k factors and the offset.
    """
    A = iso_profiles.T
    if not offset_bound:
        k, _ = _solve_k(A, measured(0.0), k_min)
        return k, 0.0

    def cost(offset: float) -> float:
        return _solve_k(A, measured(offset), k_min)[1]

    grid = np.linspace(-offset_bound, offset_bound, 21)
    costs = np.array([cost(o) for o in grid])
    i = int(np.argmin(costs))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    best_offset, best_cost = grid[i], costs[i]
    if hi > lo:
        res = minimize_scalar(
            cost,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": offset_bound * 1e-4},
        )
        if res.fun < best_cost:
            best_offset = float(res.x)
    k, _ = _solve_k(A, measured(best_offset), k_min)
    return k, float(best_offset)
