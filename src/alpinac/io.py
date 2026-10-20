"""Read peak lists.

The standard peak table (a :class:`pandas.DataFrame`) has one row per
measured peak and the columns:

=================  ========  ==========================================================
column             required  meaning
=================  ========  ==========================================================
``mass``           yes       measured m/z (centre of the peak)
``area``           yes       peak area (signal intensity)
``mass_u_ppm``     yes       mass uncertainty due to measurement noise, ppm
``peak_width``     yes       sigma of the peak in the mass domain (m/z)
``peak_alpha``     yes       Lorentzian fraction of the pseudo-Voigt peak shape
``mass_cal_u_ppm`` no (0)    mass uncertainty due to the mass calibration, ppm
``compound``       no (0)    index of the group of co-eluting peaks (-1 = no compound)
``lod``            no (inf)  limit of detection at that mass
``rt``             no (0)    retention time, s
``ionisation``     no (EI)   ``"EI"`` or ``"CI"``
``adduct``         no        maximal adduct formula (CI only), e.g. ``"H"``
``spectrum``       no (0)    spectrum id: peaks measured in the same run with the same detector
``area_u``, ``sn``  no        not used
=================  ========  ==========================================================
"""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PEAK_COLUMNS = (
    "rt",
    "mass",
    "mass_u_ppm",
    "mass_cal_u_ppm",
    "area",
    "area_u",
    "peak_width",
    "peak_alpha",
    "compound",
    "lod",
    "sn",
    "ionisation",
    "adduct",
    "spectrum",
)
REQUIRED_COLUMNS = ("mass", "area", "mass_u_ppm", "peak_width", "peak_alpha")

# Column names used in the files of the legacy extraction
_RENAME = {
    "RT": "rt",
    "compound_bin": "compound",
    "LOD": "lod",
    "SN": "sn",
    "Ionisation": "ionisation",
    "Ionization": "ionisation",
    "ionization": "ionisation",
    "Adduct": "adduct",
    "Spectrum_id": "spectrum",
    "run_bin": "spectrum",
}

# Positional columns of the legacy files without header
_LEGACY_POSITIONS = (
    "rt",
    "mass",
    "mass_u_ppm",
    "mass_cal_u_ppm",
    "area",
    "area_u",
    "peak_width",
    "peak_alpha",
    "compound",
    "lod",
    "sn",
    "ionisation",
    "adduct",
    "spectrum",
)


def normalize_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a peak table with standard column names, types and defaults."""
    df = df.rename(columns=_RENAME).copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required peak columns: {missing}")
    defaults = {
        "rt": 0.0,
        "mass_cal_u_ppm": 0.0,
        "area_u": np.nan,
        "compound": 0,
        "lod": np.inf,
        "sn": np.nan,
        "ionisation": "EI",
        "adduct": "",
        "spectrum": "0",
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    for col in ("rt", "mass", "mass_u_ppm", "mass_cal_u_ppm", "area", "area_u",
                "peak_width", "peak_alpha", "lod", "sn"):
        df[col] = df[col].astype(float)
    df["compound"] = df["compound"].astype(int)
    df["ionisation"] = df["ionisation"].astype(str).str.upper()
    adduct = df["adduct"].fillna("").astype(str)
    df["adduct"] = adduct.where(~adduct.str.lower().isin(["none", "nan"]), "")
    df["spectrum"] = df["spectrum"].astype(str)
    return df[list(PEAK_COLUMNS)].reset_index(drop=True)


def read_peaks(path: str | PathLike) -> pd.DataFrame:
    """Read a peak list file (``.txt``/``.csv`` written by the alpinac extraction, or ``.jdx``).

    :return: the standard peak table (see the module docstring).
    """
    path = Path(path)
    if path.suffix.lower() == ".jdx":
        peaks, _ = read_jdx(path)
        return peaks
    with path.open() as f:
        first = ""
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                first = line
                break
    has_header = first.split()[0] in ("RT", "rt") if first.split() else False
    if has_header:
        df = pd.read_csv(path, sep=r"\s+", comment="#", dtype={"Spectrum_id": str})
    else:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
        df.columns = _LEGACY_POSITIONS[: len(df.columns)]
    return normalize_peaks(df)


def read_jdx(path: str | PathLike) -> tuple[pd.DataFrame, dict]:
    """Read a NIST unit-mass EI spectrum (JCAMP-DX, requires ``jcamp``).

    :return: the peak table and the metadata (``name``, ``formula``, ``cas``).
    """
    from jcamp import jcamp_read

    with open(path) as f:
        data = jcamp_read(f)
    y = np.asarray(data["y"], dtype=float)
    peaks = pd.DataFrame(
        {
            "mass": np.asarray(data["x"], dtype=float),
            "area": y,
            "area_u": 0.05 * y,
            "mass_u_ppm": 0.0,
            "peak_width": 0.0,
            "peak_alpha": 0.0,
            "lod": y.min() * 0.1,
        }
    )
    meta = {
        "name": data.get("title"),
        "formula": "".join(str(data.get("molform", "")).split()) or None,
        "cas": data.get("cas registry no"),
    }
    return normalize_peaks(peaks), meta
