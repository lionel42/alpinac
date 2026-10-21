"""``alpinac-isotopes``: isotopes of an element, or isotopologue pattern of a formula."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alpinac-isotopes",
        description="Isotopes of an element (e.g. Cl), or the isotopologue pattern of a "
        "formula (e.g. CCl4): m/z and intensity relative to the most intense one.",
    )
    parser.add_argument("formula")
    parser.add_argument("--min", type=float, default=1e-3,
                        help="Smallest relative intensity shown (default 0.001).")
    args = parser.parse_args(argv)

    from alpinac.elements import ELECTRON_MASS, ELEMENTS, INDEX
    from alpinac.formula import format_isotopologue, parse_formula
    from alpinac.isotopes import enumerate_isotopologues

    if args.formula in INDEX:
        e = ELEMENTS[INDEX[args.formula]]
        print(f"{e.symbol}: valence {e.valence}")
        print(f"{'isotope':<10}{'mass':>16}{'abundance':>12}")
        for iso in e.isotopes:
            print(f"{iso.label:<10}{iso.mass:>16.8f}{iso.abundance:>12.6f}")
        return 0
    try:
        f = parse_formula(args.formula)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1
    patterns, rel, mass = enumerate_isotopologues(f, 0.0)
    rel = rel / rel.max()
    order = mass.argsort()
    print(f"{'isotopologue':<24}{'m/z (1+)':>14}{'rel. intensity':>16}")
    for i in order:
        if rel[i] >= args.min:
            print(f"{format_isotopologue(f, patterns[i]):<24}{mass[i] - ELECTRON_MASS:>14.6f}{rel[i]:>16.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
