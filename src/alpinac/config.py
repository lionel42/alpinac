"""Parameters of the identification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

DEFAULT_ELEMENTS = "HCNOSFClBrI"


@dataclass(frozen=True)
class IdentificationConfig:
    """Parameters of the identification (Guillevic et al. 2021).

    The defaults are the ones of the legacy ``const_identification`` module.
    """

    target_elements: str | None = None
    """Elements possibly present, e.g. ``"CHClF"``. Defaults to
    :data:`DEFAULT_ELEMENTS` when neither this nor ``target_formula`` is given.
    The identification is faster and more reliable with fewer elements, but
    an element that is present and not listed makes the result wrong."""

    target_formula: str | None = None
    """If given, only sub-formulas of this formula are enumerated."""

    mass_u_k: float = 3.0
    """Coverage factor of the mass uncertainty (eq. 1 of Guillevic et al.)."""

    mass_u_k_loose: float = 3.0
    """Coverage factor of the tolerance used to match minor isotopologues to
    measured peaks (peak width + mass uncertainty)."""

    min_detect_mass: float = 23.0
    """Smallest detectable mass of the instrument (m/z). The smallest
    measured mass is used if it is lower. Used to count the possible
    sub-fragments of a candidate (likelihood estimator)."""

    lod: float = 80.0
    """Instrumental limit of detection (peak area). The smallest measured peak
    is used if it is lower."""

    max_opt_signal: float = 95.0
    """Percentage of the measured signal to reproduce before stopping the
    optimisation."""

    max_ratio_not_opt_masses: float = 0.1
    """Fraction of measured masses allowed to stay unexplained."""

    max_added_nodes: int = 300
    """Safety limit on the number of candidates added to the optimisation."""

    ppm_mass_res: float = 8.0
    """Sampling of the mass profiles used to fit the isotopologues.
    As in the legacy code this is the number of points per m/z unit
    (see PLAN.md, it is documented as ppm in the legacy code)."""

    min_no_masses: int = 6
    """Below this number of measured masses, each maximal fragment is
    optimised separately ("few masses" mode)."""

    isotope_pruning_factor: float = 1.1
    """Isotopologues whose expected intensity is below LOD / factor are not
    generated."""

    below_lod_factor: float = 300.0
    """A candidate is removed after a fit if its total fitted intensity times
    this factor is below the LOD (legacy rule)."""

    @property
    def elements(self) -> str | None:
        if self.target_elements is None and self.target_formula is None:
            return DEFAULT_ELEMENTS
        return self.target_elements

    def replace(self, **changes) -> IdentificationConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)
