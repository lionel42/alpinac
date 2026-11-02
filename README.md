# alpinac (identification)

ALgorithmic Process for Identification of Non-targeted Atmospheric Compounds.

Given the peak list of a compound measured by high resolution mass
spectrometry (TOF), alpinac finds the chemical formulas of its fragments and
proposes molecular ion candidates (Guillevic et al. 2021,
[doi:10.1186/s13321-021-00544-w](https://doi.org/10.1186/s13321-021-00544-w)).

This package is a rewrite of the identification part of the original alpinac
(`halosearch/alpinac`). The peak extraction is not part of it.

## Install

```bash
pip install -e .            # numpy, scipy, pandas, numba
pip install -e ".[jdx]"     # + reading NIST .jdx files
```

## Python

```python
import alpinac

# All compounds of a peak file (written by the alpinac extraction)
results = alpinac.identify_file("190424.0810.tank.3.frag.val_HFC134a.txt",
                                target_elements="CHF")
r = results[0]
print(r)
# Compound 0 [ok] rt=1478.6s, 23 peaks, 99.0% of signal explained
#   1. C2H2F4         likelihood  71.25  m/z 102.0087

r.best.formula          # "C2H2F4"
r.molecular_ions_df()   # molecular ion candidates, ranked
r.fragments_df()        # all fragment formulas kept, ranked by likelihood
r.peaks_df()            # for each measured peak: the isotopologues explaining it
r.save("out/")          # molecular_ions.csv, fragments.csv, peaks.csv, summary.json

# One compound from a DataFrame of peaks
r = alpinac.identify(peaks_df, target_elements="CHClF")

# Parameters
cfg = alpinac.IdentificationConfig(target_elements="CHClF", max_opt_signal=90)
results = alpinac.identify_file(path, cfg, n_jobs=4, timeout=300, output_dir="out/")
```

## Command line

```bash
alpinac "data/*.frag.*.txt" -t CHClF -o results/ -j 4 --timeout 300
alpinac run.frag.txt -c 3 -f CCl4        # only compound 3, sub-formulas of CCl4
alpinac run.frag.txt -p                  # also show what explains each measured peak

alpinac-mass CCl3F "CF3+" "[13C]Cl4"    # exact masses
alpinac-isotopes CCl3                    # isotopologue pattern
alpinac-isotopes Cl                      # isotopes of an element
alpinac-knapsack 100.9361 --ppm 10 -t CHClF   # formulas matching an m/z
```

## Input

A whitespace separated text file with a header, one row per peak:

| column | required | meaning |
|---|---|---|
| `mass` | yes | measured m/z |
| `area` | yes | peak area |
| `mass_u_ppm` | yes | mass uncertainty (measurement noise), ppm |
| `mass_cal_u_ppm` | no | mass calibration uncertainty, ppm |
| `peak_width`, `peak_alpha` | yes | pseudo-Voigt peak shape in the mass domain |
| `compound_bin` | no | group of co-eluting peaks (-1: not a compound) |
| `LOD` | no | limit of detection at that mass |
| `RT` | no | retention time |
| `Ionisation` | no | `EI` (default) or `CI` |
| `Adduct` | no | maximal adduct for CI, e.g. `H` |
| `Spectrum_id` | no | spectrum the peak belongs to |

The same columns (lower case names: `rt`, `compound`, `lod`, `ionisation`,
`adduct`, `spectrum`) are used for DataFrames, see `alpinac.io`.

## Output

- `molecular_ions.csv`: rank, formula, m/z, DBE, likelihood, explained signal (%), spectrum, adduct,
  `measured` (False if the molecular ion was reconstructed and not measured).
- `fragments.csv`: all fragment formulas kept, ranked by likelihood, with their fitted intensity.
- `peaks.csv`: one row per (peak, isotopologue) assignment with the mass error and the
  assigned part of the area; unexplained peaks have an empty formula.
- `summary.json`, and `summary.csv` / `config.json` for a whole file.

Formulas are written in Hill order (C, H, then alphabetical), isotopes as `[37Cl]`.

Doubly charged fragments (EI) are found as well: their formulas end with `++`
(e.g. `C6H3++`, `C5[13C]H6++`) and `fragments.csv` / `peaks.csv` have a
`charge` column. They are only proposed when they explain a half-integer peak
(otherwise `C6H6++` cannot be told apart from `C3H3+`) and are a sub-formula of
a singly charged fragment. `max_charge=1` switches this off.

## Differences to the legacy alpinac

The method and its parameters are the same, plus the doubly charged fragments. A list of the bugs fixed and of
the (few) behaviour changes is in [PLAN.md](PLAN.md) ("Behaviour changes").
On the 55 reference compounds of the paper (`tests/data/fragments`), the true
formula is ranked first for 38 compounds (legacy 36) and the identification
is about 45 times faster (12.5 s instead of 573 s for the whole set).

Not supported: unit-mass data (NIST spectra can be read but not identified)
and the "discrete" fitting method of the private `alpinac_sups` package.

## Development

```bash
python -m pytest                          # tests (~20 s)
python benchmarks/run_reference.py -j 8   # reference set vs the legacy baseline
```

`tests/data/legacy/` holds intermediate results of the legacy code, used as
test oracles (`benchmarks/legacy/dump_legacy_oracles.py`).
