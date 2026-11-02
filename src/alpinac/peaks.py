"""Measured peaks of one compound."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpinac.config import IdentificationConfig
from alpinac.elements import ADDUCT_INDEX, ELECTRON_MASS, N_ELEMENTS
from alpinac.formula import parse_formula
from alpinac.io import normalize_peaks


@dataclass
class Compound:
    """The measured peaks of one compound (group of co-eluting peaks).

    Peaks are sorted by increasing mass. All arrays have one value per peak.
    """

    peaks: pd.DataFrame
    """The standard peak table of this compound (sorted by mass)."""
    mass: np.ndarray
    """Measured m/z."""
    area: np.ndarray
    lod_peak: np.ndarray
    """LOD at each peak."""
    mass_U: np.ndarray
    """Expanded mass uncertainty: ``k * sqrt(u**2 + u_cal**2)`` (m/z)."""
    mass_tol_isotope: np.ndarray
    """Tolerance to match a minor isotopologue: ``k_loose * (width + sqrt(u**2 + u_cal**2))``."""
    spectrum_id: np.ndarray
    """Index of the spectrum (in :attr:`spectra`) of each peak."""
    spectra: list[str]
    """Spectrum names, in order of first appearance (by increasing mass)."""
    spectrum_ionisation: list[str]
    spectrum_adduct: list[np.ndarray]
    """Maximal adduct of each spectrum as counts of adduct pseudo-elements."""
    spectrum_adduct_formula: list[str]
    sigma_slope: float
    sigma_intercept: float
    """Peak width (sigma) in the mass domain: ``sigma(m) = slope * m + intercept``."""
    alpha: float
    """Lorentzian fraction of the pseudo-Voigt peak shape (mean over peaks)."""
    lod: float
    """Limit of detection used for this compound: min(config LOD, smallest peak)."""
    unit_mass: bool
    """True if all masses are integers (unit mass resolution data)."""
    mass_axis_offset: float

    @property
    def n_peaks(self) -> int:
        return len(self.mass)

    @property
    def n_spectra(self) -> int:
        return len(self.spectra)

    @property
    def sum_area(self) -> float:
        return float(self.area.sum())

    @property
    def neutral_mass(self) -> np.ndarray:
        """Masses with one electron added back, as used by the knapsack.

        For unit mass data the masses are used as they are.
        """
        return self.mass if self.unit_mass else self.mass + ELECTRON_MASS

    @property
    def mass_min(self) -> np.ndarray:
        return self.neutral_mass - self.mass_U

    @property
    def mass_max(self) -> np.ndarray:
        return self.neutral_mass + self.mass_U

    @property
    def half_integer(self) -> np.ndarray:
        """Peaks at a half-integer m/z (fractional part in ``(0.25, 0.75)``).

        Singly charged ions of the masses measured here have a mass defect
        well below 0.25, such peaks are the signature of doubly charged ions
        (with an odd nominal mass).
        """
        frac = self.mass % 1.0
        return (frac > 0.25) & (frac < 0.75)

    def sigma(self, mass: np.ndarray | float) -> np.ndarray | float:
        return self.sigma_slope * mass + self.sigma_intercept

    @property
    def rt(self) -> float:
        """Area-weighted retention time."""
        return float(np.sum(self.peaks["rt"].to_numpy() * self.area) / self.sum_area)

    @classmethod
    def from_peaks(
        cls, peaks: pd.DataFrame, config: IdentificationConfig | None = None
    ) -> Compound:
        """Build a compound from a peak table (all rows are used)."""
        config = config or IdentificationConfig()
        peaks = normalize_peaks(peaks)
        if len(peaks) == 0:
            raise ValueError("No peaks")
        peaks = peaks.sort_values("mass", kind="stable").reset_index(drop=True)
        mass = peaks["mass"].to_numpy()
        area = peaks["area"].to_numpy()

        unit_mass = bool(np.all(mass == np.round(mass)))
        if unit_mass:
            mass_U = np.full(len(mass), 0.49995)
            mass_tol = mass_U.copy()
            sigma_slope, sigma_intercept, alpha = 0.0, 0.0, 0.0
            mass_axis_offset = 0.0
        else:
            u = peaks["mass_u_ppm"].to_numpy() * 1e-6 * mass
            u_cal = peaks["mass_cal_u_ppm"].to_numpy() * 1e-6 * mass
            u_comb = np.sqrt(u**2 + u_cal**2)
            mass_U = config.mass_u_k * u_comb
            width = peaks["peak_width"].to_numpy()
            mass_tol = config.mass_u_k_loose * (width + u_comb)
            alpha = float(peaks["peak_alpha"].mean())
            sigma_slope, sigma_intercept = _fit_linear(mass + ELECTRON_MASS, width)
            mass_axis_offset = 0.5

        # Spectra, in order of first appearance
        spectra: list[str] = []
        spectrum_id = np.empty(len(peaks), dtype=np.int64)
        for i, name in enumerate(peaks["spectrum"]):
            if name not in spectra:
                spectra.append(name)
            spectrum_id[i] = spectra.index(name)

        ionisations, adducts, adduct_formulas = [], [], []
        for i_spec, name in enumerate(spectra):
            sel = peaks[spectrum_id == i_spec]
            ion = sel["ionisation"].unique()
            add = sel["adduct"].unique()
            if len(ion) > 1 or len(add) > 1:
                raise ValueError(
                    f"All peaks of spectrum {name!r} must have the same ionisation "
                    f"and adduct, got {list(ion)} and {list(add)}"
                )
            ion, add = str(ion[0]), str(add[0])
            if ion not in ("EI", "CI"):
                raise ValueError(f"Unknown ionisation {ion!r} for spectrum {name!r}")
            if add and ion == "EI":
                raise ValueError(f"Adducts are not allowed for EI (spectrum {name!r})")
            adduct = np.zeros(N_ELEMENTS, dtype=np.int64)
            if add:
                real = parse_formula(add)
                for i_el in np.flatnonzero(real):
                    if i_el not in ADDUCT_INDEX:
                        raise ValueError(f"Element of adduct {add!r} not supported")
                    adduct[ADDUCT_INDEX[i_el]] = real[i_el]
            ionisations.append(ion)
            adducts.append(adduct)
            adduct_formulas.append(add)

        lod = float(min(config.lod, area.min()))

        return cls(
            peaks=peaks,
            mass=mass,
            area=area,
            lod_peak=peaks["lod"].to_numpy(),
            mass_U=mass_U,
            mass_tol_isotope=mass_tol,
            spectrum_id=spectrum_id,
            spectra=spectra,
            spectrum_ionisation=ionisations,
            spectrum_adduct=adducts,
            spectrum_adduct_formula=adduct_formulas,
            sigma_slope=float(sigma_slope),
            sigma_intercept=float(sigma_intercept),
            alpha=alpha,
            lod=lod,
            unit_mass=unit_mass,
            mass_axis_offset=mass_axis_offset,
        )


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares line ``y = a x + b`` (constant if x has no spread)."""
    n = len(x)
    den = np.sum(x**2) - np.sum(x) ** 2 / n
    if den == 0:
        return 0.0, float(np.mean(y))
    a = (np.sum(x * y) - np.sum(x) * np.sum(y) / n) / den
    b = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / n / den
    return float(a), float(b)


def split_compounds(peaks: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split a peak table by compound index (compound -1, unassigned peaks, is skipped)."""
    peaks = normalize_peaks(peaks)
    return {
        int(k): g.reset_index(drop=True)
        for k, g in peaks.groupby("compound", sort=False)
        if int(k) != -1
    }
