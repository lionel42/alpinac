"""``alpinac-knapsack``: formulas matching a measured m/z."""

from __future__ import annotations

import argparse
import sys

from alpinac.cli import add_target_arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alpinac-knapsack",
        description="All formulas (most abundant isotopes, DBE >= 0) of singly charged "
        "ions matching a measured m/z.",
    )
    parser.add_argument("mz", type=float, help="Measured m/z.")
    tol = parser.add_mutually_exclusive_group()
    tol.add_argument("--tol", type=float,
                     help="Tolerance in m/z (default 0.005, or 0.5 if the m/z is an integer).")
    tol.add_argument("--ppm", type=float, help="Tolerance in ppm.")
    add_target_arguments(parser)
    args = parser.parse_args(argv)

    from alpinac.config import DEFAULT_ELEMENTS
    from alpinac.elements import ELECTRON_MASS
    from alpinac.formula import dbe, format_formula, ion_mass
    from alpinac.knapsack import FormulaSpace

    if args.ppm is not None:
        tolerance = args.mz * args.ppm * 1e-6
    elif args.tol is not None:
        tolerance = args.tol
    else:
        # an integer m/z is a nominal (unit) mass
        tolerance = 0.5 if args.mz.is_integer() else 0.005
    neutral = args.mz + ELECTRON_MASS
    elements = args.target_elements or (None if args.target_formula else DEFAULT_ELEMENTS)
    try:
        space = FormulaSpace.from_target(neutral + tolerance, elements, args.target_formula)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    formulas = space.enumerate(neutral - tolerance, neutral + tolerance)
    rows = sorted(((ion_mass(f), format_formula(f), dbe(f)) for f in formulas), key=lambda r: abs(r[0] - args.mz))
    print(f"{len(rows)} formula(s) within {tolerance:.5f} m/z of {args.mz}")
    if rows:
        print(f"{'formula':<16}{'m/z':>14}{'error ppm':>11}{'DBE':>6}")
    for m, name, d in rows:
        print(f"{name + '+':<16}{m:>14.6f}{(m - args.mz) / args.mz * 1e6:>11.2f}{d:>6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
