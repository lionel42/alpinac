import json

import pandas as pd

from conftest import DATA, FRAGMENTS
from alpinac import identify_compounds, identify_file
from alpinac.cli import identify as cli_identify
from alpinac.cli import isotopes as cli_isotopes
from alpinac.cli import knapsack as cli_knapsack
from alpinac.cli import mass as cli_mass
from alpinac.io import read_peaks
from alpinac.peaks import split_compounds

CFC11 = FRAGMENTS / "200602.1550.std.7.frag.train_CFC11.txt"


def test_save(tmp_path):
    identify_file(CFC11, output_dir=tmp_path)
    assert (tmp_path / "summary.csv").exists()
    assert json.loads((tmp_path / "config.json").read_text())["lod"] == 80.0
    d = tmp_path / "compound_0"
    assert sorted(p.name for p in d.iterdir()) == [
        "fragments.csv", "molecular_ions.csv", "peaks.csv", "summary.json"
    ]
    assert pd.read_csv(d / "molecular_ions.csv")["formula"].iloc[0] == "CCl3F"


def test_parallel_and_timeout():
    compounds = split_compounds(read_peaks(DATA / "frags_Xe.csv"))
    compounds.update({10: split_compounds(read_peaks(CFC11))[0]})
    res = identify_compounds(compounds, n_jobs=2, timeout=60)
    assert res[10].best.formula == "CCl3F"
    res = identify_compounds(compounds, timeout=1e-4)
    assert {r.status for r in res.values()} == {"timeout"}


def test_select_compounds():
    path = FRAGMENTS / "200602.1550.std.7.frag.val_HFO1234zeE.txt"
    keys = sorted(split_compounds(read_peaks(path)))
    assert len(keys) == 2
    res = identify_file(path, compounds=[keys[1]])
    assert list(res) == [keys[1]]


def test_cli_identify(tmp_path, capsys):
    assert cli_identify.main([str(CFC11), "-o", str(tmp_path), "-n", "1"]) == 0
    out = capsys.readouterr().out
    assert "CCl3F" in out
    assert (tmp_path / "compound_0" / "peaks.csv").exists()
    assert cli_identify.main([str(CFC11), "-n", "1", "--peaks"]) == 0
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if "C[37Cl] " in l and l.split()[0] == "48.96478")
    assert line.split()[-2:] == ["+11.8", "96"]
    assert "unidentified" in out


def test_cli_tools(capsys):
    assert cli_mass.main(["CF3+"]) == 0
    assert "68.994661" in capsys.readouterr().out
    assert cli_isotopes.main(["CCl3"]) == 0
    assert "CCl2[37Cl]" in capsys.readouterr().out
    assert cli_knapsack.main(["100.9361", "--ppm", "10", "-t", "CHClF"]) == 0
    assert "CCl2F+" in capsys.readouterr().out
