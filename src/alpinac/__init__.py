"""ALPINAC: ALgorithmic Process for Identification of Non-targeted Atmospheric Compounds.

Identification of the chemical formula of a compound from its high
resolution (TOF) mass spectrum (Guillevic et al. 2021,
https://doi.org/10.1186/s13321-021-00544-w).

Quick start::

    import alpinac

    results = alpinac.identify_file("run.frag.txt", target_elements="CHClF")
    for k, r in results.items():
        print(r)              # summary with the molecular ion candidates
        r.peaks_df()          # what explains each measured peak

    # From a DataFrame of peaks (one compound)
    r = alpinac.identify(peaks_df)
"""

from alpinac.api import identify_compounds, identify_file, save_results
from alpinac.config import IdentificationConfig
from alpinac.formula import format_formula, formula_mass, parse_formula
from alpinac.identify import identify
from alpinac.io import read_peaks
from alpinac.peaks import Compound, split_compounds
from alpinac.results import Fragment, IdentificationResult, MolecularIon, PeakAssignment

__version__ = "1.0.0.dev0"

__all__ = [
    "Compound",
    "Fragment",
    "IdentificationConfig",
    "IdentificationResult",
    "MolecularIon",
    "PeakAssignment",
    "format_formula",
    "formula_mass",
    "identify",
    "identify_compounds",
    "identify_file",
    "parse_formula",
    "read_peaks",
    "save_results",
    "split_compounds",
]
