"""``alpinac``: identify the compounds of peak files."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from alpinac.cli import add_logging_arguments, add_target_arguments, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpinac",
        description="Identify the molecular formula of compounds from their peak lists "
        "(.txt/.csv files of the alpinac extraction).",
    )
    parser.add_argument("files", nargs="+", help="Peak files; glob patterns are allowed (quote them).")
    add_target_arguments(parser)
    parser.add_argument("-c", "--compound", type=int, action="append", dest="compounds",
                        help="Only this compound index (can be repeated).")
    parser.add_argument("-o", "--output-dir",
                        help="Save the results there (one sub-folder per file if several files).")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Compounds identified in parallel.")
    parser.add_argument("--timeout", type=float, help="Maximal time per compound, seconds.")
    parser.add_argument("-n", "--top", type=int, default=5, help="Number of molecular ions shown.")
    parser.add_argument("-p", "--peaks", action="store_true",
                        help="Also show the identification of each measured peak.")
    parser.add_argument("--max-opt-signal", type=float, help="Percentage of the signal to explain (default 95).")
    parser.add_argument("--lod", type=float, help="Limit of detection (default 80).")
    add_logging_arguments(parser)
    return parser


def format_peaks(result) -> str:
    """One line per (peak, isotopologue) assignment, most intense peaks first.

    A peak explained by several isotopologues has several lines (largest
    contribution first); an unexplained peak has one line "unidentified".
    ``% of area`` is the part of the peak area explained by the isotopologue.
    """
    lines = [f"  {'m/z':>11} {'area':>12}  {'formula':<18} {'error ppm':>9} {'% of area':>9}"]
    df = result.peaks_df().sort_values(
        ["area", "peak", "assigned_area"], ascending=[False, True, False], kind="stable"
    )
    for r in df.itertuples():
        head = f"  {r.mass:>11.5f} {r.area:>12.1f}  "
        if r.formula:
            pct = r.assigned_area / r.area * 100 if r.area else 0.0
            pct_str = f"{pct:.2f}" if pct < 1 else f"{pct:.0f}"
            lines.append(head + f"{r.formula:<18} {r.mass_error_ppm:>+9.1f} {pct_str:>9}")
        else:
            lines.append(head + "unidentified")
    return "\n".join(lines)


def _expand(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        files.extend(Path(m) for m in matches)
    return files


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args)

    from alpinac.api import identify_file
    from alpinac.config import IdentificationConfig

    changes = {}
    if args.max_opt_signal is not None:
        changes["max_opt_signal"] = args.max_opt_signal
    if args.lod is not None:
        changes["lod"] = args.lod
    config = IdentificationConfig(
        target_elements=args.target_elements, target_formula=args.target_formula, **changes
    )

    files = _expand(args.files)
    if not files:
        print(f"No file matches {args.files}", file=sys.stderr)
        return 1
    status = 0
    for path in files:
        out = None
        if args.output_dir:
            out = Path(args.output_dir)
            if len(files) > 1:
                out = out / path.stem
        try:
            results = identify_file(path, config, compounds=args.compounds, output_dir=out,
                                    timeout=args.timeout, n_jobs=args.jobs)
        except Exception as e:
            print(f"{path}: {type(e).__name__}: {e}", file=sys.stderr)
            status = 1
            continue
        print(f"== {path}")
        for r in results.values():
            if r.ok:
                lines = str(r).splitlines()
                print("\n".join(lines[: 1 + args.top]))
                if args.peaks:
                    print(format_peaks(r))
            else:
                print(r)
                if r.status == "error":
                    status = 1
        if out is not None:
            print(f"results saved in {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
