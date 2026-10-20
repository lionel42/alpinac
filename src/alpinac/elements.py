"""Element and isotope tables.

The exact masses and the isotopic abundances are from IUPAC:
Meija, J. et al., Atomic weights of the elements 2013 (IUPAC technical report),
Pure and Applied Chemistry 88(3), 265-291 (2016), doi:10.1515/pac-2015-0305.

Formulas are represented as integer count vectors over the *element* axis
defined here (one entry per element, the count includes all isotopes).
Isotopologues additionally use the isotope tables of each element.

Elements are ordered by increasing mass of their most abundant isotope.

Adduct pseudo-elements (``ad.H``, ``ad.C``, ...) are used for chemical
ionisation: they have the mass of the real element but a valence of 2, so
they do not change the double bond equivalent. They only appear inside the
knapsack enumeration and are converted back to the real elements afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import pandas as pd

ELECTRON_MASS = 0.00054858


@dataclass(frozen=True)
class Isotope:
    label: str
    """Label as written in formulas, e.g. ``"[13C]"`` (``"C"`` for the most abundant one)."""
    nominal_mass: int
    mass: float
    abundance: float


@dataclass(frozen=True)
class Element:
    symbol: str
    valence: int
    isotopes: tuple[Isotope, ...]
    """Isotopes ordered by decreasing abundance (the first one is the most abundant)."""
    adduct_of: str | None = None
    """For adduct pseudo-elements, the symbol of the real element."""

    @property
    def mass(self) -> float:
        """Mass of the most abundant isotope."""
        return self.isotopes[0].mass

    @property
    def is_adduct(self) -> bool:
        return self.adduct_of is not None


def _load_elements() -> tuple[Element, ...]:
    table = pd.read_csv(files("alpinac") / "data" / "periodic_table.csv", index_col=0)
    groups: dict[str, list[tuple]] = {}
    for row in table.itertuples(index=False):
        label = str(row.element)
        if label.startswith("ad."):
            symbol = label
        else:
            # "[13C]" -> "C"
            symbol = label.strip("[]").lstrip("0123456789")
        groups.setdefault(symbol, []).append(row)

    elements = []
    for symbol, rows in groups.items():
        rows = sorted(rows, key=lambda r: -float(r.occurence))
        # The table is not consistent for some noble gas isotopes:
        # use the valence of the most abundant isotope.
        valence = int(rows[0].valence_index)
        isotopes = tuple(
            Isotope(
                label=str(r.element),
                nominal_mass=int(r.unit_mass),
                mass=float(r.mass),
                abundance=float(r.occurence),
            )
            for r in rows
        )
        # The most abundant isotope is written without brackets in the table
        assert not isotopes[0].label.startswith("["), symbol
        elements.append(
            Element(
                symbol=symbol,
                valence=valence,
                isotopes=isotopes,
                adduct_of=symbol[3:] if symbol.startswith("ad.") else None,
            )
        )
    elements.sort(key=lambda e: (e.mass, e.is_adduct))
    return tuple(elements)


ELEMENTS: tuple[Element, ...] = _load_elements()
N_ELEMENTS = len(ELEMENTS)
SYMBOLS: tuple[str, ...] = tuple(e.symbol for e in ELEMENTS)
INDEX: dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}

MASS = np.array([e.mass for e in ELEMENTS], dtype=np.float64)
"""Mass of the most abundant isotope of each element."""
VALENCE = np.array([e.valence for e in ELEMENTS], dtype=np.int64)
IS_ADDUCT = np.array([e.is_adduct for e in ELEMENTS], dtype=bool)
N_ISOTOPES = np.array([len(e.isotopes) for e in ELEMENTS], dtype=np.int64)

ADDUCT_INDEX: dict[int, int] = {
    INDEX[e.adduct_of]: i for i, e in enumerate(ELEMENTS) if e.is_adduct
}
"""Real element index -> adduct pseudo-element index."""
ADDUCT_TO_ELEMENT: dict[int, int] = {v: k for k, v in ADDUCT_INDEX.items()}
"""Adduct pseudo-element index -> real element index."""

ISOTOPE_INDEX: dict[str, tuple[int, int]] = {
    iso.label: (i, j)
    for i, e in enumerate(ELEMENTS)
    if not e.is_adduct
    for j, iso in enumerate(e.isotopes)
}
"""Isotope label (e.g. ``"[37Cl]"`` or ``"Cl"``) -> (element index, isotope index)."""

# Hill order: C, H, then alphabetical. Adduct pseudo-elements go last.
HILL_ORDER = np.array(
    sorted(
        range(N_ELEMENTS),
        key=lambda i: (
            ELEMENTS[i].is_adduct,
            {"C": 0, "H": 1}.get(SYMBOLS[i], 2),
            SYMBOLS[i],
        ),
    ),
    dtype=np.int64,
)


def element_index(symbol: str) -> int:
    try:
        return INDEX[symbol]
    except KeyError:
        raise ValueError(f"Unknown element {symbol!r}") from None
