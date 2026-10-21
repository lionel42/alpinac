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

    # Without a charge in the formula, the m/z is given for a single charge
    print(f"{'formula':<16}{'neutral mass':>16}{'charge':>8}{'m/z':>16}{'DBE':>7}")
    status = 0
    for f in args.formulas:
        try:
            z = max(charge(f), 1)
            neutral = formula_mass(f.split("+")[0])
            ion = formula_mass(f) if charge(f) else formula_mass(f, ionised=True)
            d = dbe(parse_formula(f))
        except ValueError as e:
            print(f"{f}: {e}", file=sys.stderr)
            status = 1
            continue
        body = f.split("+")[0] if "[" in f else format_formula(parse_formula(f))
        print(f"{body:<16}{neutral:>16.6f}{f'{z}+':>8}{ion:>16.6f}{d:>7.1f}")
    return status


if __name__ == "__main__":
    sys.exit(main())
