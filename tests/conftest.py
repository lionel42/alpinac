import json
from functools import cache
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).parent / "data"
FRAGMENTS = DATA / "fragments"
LEGACY = DATA / "legacy"


def reference_files() -> list[Path]:
    """The 55 known-compound peak files of the reference set."""
    return sorted(p for p in FRAGMENTS.glob("*.txt") if p.name != "COPYRIGHT.txt")


@cache
def reference_formulas() -> dict[str, str]:
    df = pd.read_csv(FRAGMENTS / "reference_formulas.csv")
    return dict(zip(df["file"], df["formula"]))


@cache
def legacy_oracle(path: Path) -> list[dict]:
    """Intermediate results of the legacy code for a reference file (one dict per compound)."""
    return json.loads((LEGACY / (path.stem + ".json")).read_text())


@pytest.fixture(params=reference_files(), ids=lambda p: p.stem.split(".frag.")[-1])
def reference_file(request) -> Path:
    return request.param
