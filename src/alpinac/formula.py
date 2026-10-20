"""Chemical formulas: parsing, formatting, masses, DBE.

A formula is an integer vector of element counts over
:data:`alpinac.elements.SYMBOLS` (see :func:`parse_formula`).
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from alpinac.elements import (
    ADDUCT_TO_ELEMENT,
    ELECTRON_MASS,
    ELEMENTS,
    HILL_ORDER,
    INDEX,
    ISOTOPE_INDEX,
    MASS,
    N_ELEMENTS,
    SYMBOLS,
    VALENCE,
)

_TOKEN = re.compile(r"\[\d+[A-Z][a-z]?\]|[A-Z][a-z]?|\(|\)|\d+")


def _split_charge(formula: str) -> tuple[str, int]:
    """Split a trailing charge: ``"CF3+"`` -> ``("CF3", 1)``, ``"C2+2"`` -> ``("C2", 2)``."""
    formula = formula.strip().replace(" ", "")
    if "+" not in formula:
        return formula, 0
    body, _, charge = formula.partition("+")
    if charge == "":
        return body, 1
    if charge.strip("+") == "":
        return body, 1 + len(charge)
    if not charge.isdigit():
        raise ValueError(f"Invalid charge in formula {formula!r}")
    return body, int(charge)


def parse_tokens(formula: str) -> Counter[str]:
    """Count the atoms of a formula, keeping isotope labels.

    Supports parentheses (``"(CF3)2CO"``) and isotope labels (``"[13C]H4"``).
    A trailing charge (``+``) is ignored.

    :return: counter label -> count, where label is an element symbol
        (all isotopes of that element at natural abundance... in practice the
        most abundant isotope) or an isotope label such as ``"[37Cl]"``.
    """
    body, _ = _split_charge(formula)
    pos = 0
    stack: list[Counter[str]] = [Counter()]
    last: Counter[str] | None = None
    for match in _TOKEN.finditer(body):
        if match.start() != pos:
            raise ValueError(f"Cannot parse formula {formula!r} at {body[pos:]!r}")
        pos = match.end()
        token = match.group()
        if token == "(":
            stack.append(Counter())
            last = None
        elif token == ")":
            if len(stack) == 1:
                raise ValueError(f"Unbalanced parenthesis in {formula!r}")
            last = stack.pop()
            stack[-1].update(last)
        elif token.isdigit():
            if last is None:
                raise ValueError(f"Misplaced count in {formula!r}")
            n = int(token)
            for label, count in last.items():
                stack[-1][label] += count * (n - 1)
            last = None
        else:
            if token not in ISOTOPE_INDEX:
                raise ValueError(f"Unknown element or isotope {token!r} in {formula!r}")
            last = Counter({token: 1})
            stack[-1].update(last)
    if pos != len(body):
        raise ValueError(f"Cannot parse formula {formula!r} at {body[pos:]!r}")
    if len(stack) != 1:
        raise ValueError(f"Unbalanced parenthesis in {formula!r}")
    return +stack[0]


def parse_formula(formula: str) -> np.ndarray:
    """Element count vector of a formula (isotopes are counted as their element)."""
    counts = np.zeros(N_ELEMENTS, dtype=np.int64)
    for label, n in parse_tokens(formula).items():
        counts[ISOTOPE_INDEX[label][0]] += n
    return counts


def parse_elements(elements: str) -> np.ndarray:
    """Boolean mask of the elements present in a string such as ``"CHClF"``."""
    return parse_formula(elements) > 0


def charge(formula: str) -> int:
    return _split_charge(formula)[1]


def formula_mass(formula: str, ionised: bool = False) -> float:
    """Exact mass of a formula string (isotope labels are taken into account).

    If the formula has a charge (``"CF3+"``), or ``ionised`` is True (single
    charge), the mass of the lost electrons is removed and the result is m/z.
    """
    z = charge(formula)
    if z == 0 and ionised:
        z = 1
    tokens = parse_tokens(formula)
    mass = 0.0
    for label, n in tokens.items():
        i_el, i_iso = ISOTOPE_INDEX[label]
        mass += n * ELEMENTS[i_el].isotopes[i_iso].mass
    return (mass - z * ELECTRON_MASS) / max(z, 1)


def mass(counts: np.ndarray) -> float:
    """Neutral mass of an element count vector (most abundant isotopes)."""
    return float(np.dot(counts, MASS))


def ion_mass(counts: np.ndarray, z: int = 1) -> float:
    """m/z of an element count vector, for a positive ion of charge ``z``."""
    return (mass(counts) - z * ELECTRON_MASS) / z


def dbe(counts: np.ndarray) -> float:
    """Double bond equivalent (rings plus double bonds).

    Gross, Mass spectrometry, 2007, eq. (6.9):
    ``DBE = 1 + sum(n_i * (v_i - 2)) / 2``.
    """
    return 1.0 + float(np.dot(counts, VALENCE - 2)) / 2.0


def dbe2(counts: np.ndarray) -> int:
    """Twice the DBE (an integer)."""
    return int(2 + np.dot(counts, VALENCE - 2))


def senior_rule_ii(counts: np.ndarray) -> bool:
    """SENIOR rule (ii) (Kind and Fiehn 2007).

    The sum of valences is at least twice the maximum valence.
    It rejects for example CFCl as a molecular ion.
    """
    present = counts > 0
    if not present.any():
        return False
    return int(np.dot(counts, VALENCE)) >= 2 * int(VALENCE[present].max())


def remove_adduct_elements(counts: np.ndarray) -> np.ndarray:
    """Move the counts of adduct pseudo-elements to their real elements."""
    counts = np.array(counts, copy=True)
    for i_ad, i_el in ADDUCT_TO_ELEMENT.items():
        counts[..., i_el] += counts[..., i_ad]
        counts[..., i_ad] = 0
    return counts


def format_formula(counts: np.ndarray) -> str:
    """Formula string in Hill order (C, H, then alphabetical)."""
    parts = []
    for i in HILL_ORDER:
        n = int(counts[i])
        if n > 0:
            parts.append(SYMBOLS[i] if n == 1 else f"{SYMBOLS[i]}{n}")
    return "".join(parts)


def format_isotopologue(counts: np.ndarray, rare: dict[int, np.ndarray] | None) -> str:
    """Formula string of an isotopologue.

    :param counts: element counts (all isotopes together)
    :param rare: element index -> counts of each isotope of that element
        (index 0 is the most abundant isotope). Elements not in the dict have
        all their atoms as the most abundant isotope.
    """
    if not rare:
        return format_formula(counts)
    parts = []
    for i in HILL_ORDER:
        n = int(counts[i])
        if n == 0:
            continue
        if i not in rare:
            parts.append(SYMBOLS[i] if n == 1 else f"{SYMBOLS[i]}{n}")
            continue
        isotopes = ELEMENTS[i].isotopes
        for j, k in enumerate(rare[i]):
            k = int(k)
            if k > 0:
                label = isotopes[j].label
                parts.append(label if k == 1 else f"{label}{k}")
    return "".join(parts)


def same_formula(a: str, b: str) -> bool:
    """True if two formula strings describe the same element counts."""
    return bool(np.array_equal(parse_formula(a), parse_formula(b)))


__all__ = [
    "INDEX",
    "charge",
    "dbe",
    "dbe2",
    "format_formula",
    "format_isotopologue",
    "formula_mass",
    "ion_mass",
    "mass",
    "parse_elements",
    "parse_formula",
    "parse_tokens",
    "remove_adduct_elements",
    "same_formula",
    "senior_rule_ii",
]
