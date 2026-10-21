"""``alpinac-mass``: exact mass of formulas."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alpinac-mass",
        description="Exact mass of chemical formulas. Isotopes ([13C]), parentheses and "
        "charges (CF3+) are allowed.",
    )
    parser.add_argument("formulas", nargs="+")
    args = parser.parse_args(argv)

    from alpinac.formula import charge, dbe, format_formula, formula_mass, parse_formula

    print(f"{'formula':<16}{'neutral mass':>16}{'m/z (1+)':>16}{'DBE':>7}")
    status = 0
    for f in args.formulas:
        try:
            neutral = formula_mass(f.split("+")[0])
            ion = formula_mass(f) if charge(f) else formula_mass(f, ionised=True)
            d = dbe(parse_formula(f))
        except ValueError as e:
            print(f"{f}: {e}", file=sys.stderr)
            status = 1
            continue
        label = f if "[" in f else format_formula(parse_formula(f)) + ("+" * min(charge(f), 1))
        print(f"{label:<16}{neutral:>16.6f}{ion:>16.6f}{d:>7.1f}")
    return status


if __name__ == "__main__":
    sys.exit(main())
