"""`forgeff export`."""

from __future__ import annotations

import argparse

from forgeff.io import read_potential, write_lammps_potential


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input potential file, such as initial.toml or final.npy.")
    parser.add_argument(
        "output",
        help="Output LAMMPS-compatible file, such as final.eam.alloy, final.fs, or final.adp.",
    )
    parser.add_argument(
        "--form",
        default=None,
        help="Explicit format hint for ambiguous inputs such as NIST .txt files.",
    )


def run(args: argparse.Namespace) -> None:
    data = read_potential(args.input, form=args.form)
    write_lammps_potential(args.output, data)

