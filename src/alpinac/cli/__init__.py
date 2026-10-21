"""Command line tools."""

from __future__ import annotations

import argparse
import logging


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t", "--target-elements",
        help="Elements possibly present, e.g. CHFClBr (default: HCNOSFClBrI).",
    )
    parser.add_argument(
        "-f", "--target-formula",
        help="Maximal formula: only its sub-formulas are considered, e.g. CCl4.",
    )


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_const", dest="loglevel",
                        const=logging.INFO, default=logging.WARNING, help="Show progress.")
    parser.add_argument("-d", "--debug", action="store_const", dest="loglevel",
                        const=logging.DEBUG, help="Show debug information.")


def setup_logging(args: argparse.Namespace) -> None:
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("alpinac").setLevel(args.loglevel)
