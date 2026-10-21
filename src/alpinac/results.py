"""Results of the identification of one compound."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Fragment:
    """A fragment formula kept at the end of the identification."""

    rank: int
    """Rank by likelihood among all fragments (1 = most likely)."""
    formula: str
    mass: float
    """m/z of the ion (most abundant isotopes, single charge)."""
    dbe: float
    likelihood: float
    """Likelihood estimator of Guillevic et al. 2021 (eq. 5)."""
    signal_percent: float
    """Percentage of the total measured signal explained by this fragment and all its sub-fragments."""
    intensity: float
    """Fitted intensity of this fragment (all its isotopologues, all spectra)."""
    is_maximal: bool
    """No other fragment contains this one."""
    measured: bool
    """The knapsack proposed it for a measured peak (False for constructed molecular ions)."""


@dataclass
class MolecularIon:
    """A molecular ion candidate."""

    rank: int
    """Rank among the molecular ion candidates of the spectrum (1 = most likely)."""
    formula: str
    mass: float
    dbe: float
    likelihood: float
    signal_percent: float
    spectrum: str
    adduct: str
    """Maximal adduct of the spectrum (CI), empty for EI."""
    measured: bool
    """The molecular ion itself was measured (False if it was reconstructed)."""


@dataclass
class PeakAssignment:
    """What explains one isotopologue contribution to a measured peak.

    A measured peak can have several rows (several isotopologues of different
    fragments). Unexplained peaks have one row with an empty formula.
    """

    peak: int
    """Index of the peak (peaks sorted by mass)."""
    mass: float
    area: float
    spectrum: str
    formula: str
    """Formula of the isotopologue, e.g. ``"CCl2[37Cl]"``, empty if unexplained."""
    fragment: str
    """Formula of the fragment (abundant isotopes) this isotopologue belongs to."""
    exact_mass: float
    mass_error_ppm: float
    assigned_area: float
    """Part of the peak area explained by this isotopologue."""


@dataclass
class IdentificationResult:
    """Result of the identification of one compound."""

    compound: int | str
    status: str
    """``"ok"``, ``"no_solution"`` (no formula matches the masses),
    ``"timeout"`` or ``"error"``."""
    message: str = ""
    rt: float = float("nan")
    n_peaks: int = 0
    signal_percent: float = 0.0
    """Percentage of the measured signal explained by the fragments."""
    molecular_ions: list[MolecularIon] = field(default_factory=list)
    fragments: list[Fragment] = field(default_factory=list)
    peaks: list[PeakAssignment] = field(default_factory=list)
    runtime: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def best(self) -> MolecularIon | None:
        """The most likely molecular ion (of the first spectrum)."""
        return self.molecular_ions[0] if self.molecular_ions else None

    @property
    def molecular_ion_formulas(self) -> list[str]:
        return [m.formula for m in self.molecular_ions]

    def molecular_ions_df(self) -> pd.DataFrame:
        return _df(self.molecular_ions, MolecularIon)

    def fragments_df(self) -> pd.DataFrame:
        return _df(self.fragments, Fragment)

    def peaks_df(self) -> pd.DataFrame:
        return _df(self.peaks, PeakAssignment)

    def summary(self) -> dict:
        return {
            "compound": self.compound,
            "status": self.status,
            "message": self.message,
            "rt": self.rt,
            "n_peaks": self.n_peaks,
            "signal_percent": self.signal_percent,
            "best": self.best.formula if self.best else None,
            "molecular_ions": self.molecular_ion_formulas,
            "runtime": self.runtime,
        }

    def save(self, directory: str | Path) -> Path:
        """Write ``molecular_ions.csv``, ``fragments.csv``, ``peaks.csv`` and ``summary.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.molecular_ions_df().to_csv(directory / "molecular_ions.csv", index=False)
        self.fragments_df().to_csv(directory / "fragments.csv", index=False)
        self.peaks_df().to_csv(directory / "peaks.csv", index=False)
        (directory / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        return directory

    def __str__(self) -> str:
        head = f"Compound {self.compound} [{self.status}]"
        if not self.ok:
            return f"{head}: {self.message}"
        lines = [
            f"{head} rt={self.rt:.1f}s, {self.n_peaks} peaks, {self.signal_percent:.1f}% of signal explained"
        ]
        for m in self.molecular_ions[:5]:
            lines.append(
                f"  {m.rank}. {m.formula:<14} likelihood {m.likelihood:6.2f}  m/z {m.mass:.4f}"
            )
        return "\n".join(lines)


def _df(items: list, cls) -> pd.DataFrame:
    columns = list(cls.__dataclass_fields__)
    return pd.DataFrame([asdict(i) for i in items], columns=columns)
