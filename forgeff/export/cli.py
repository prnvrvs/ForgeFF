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
        "--nr",
        type=int,
        default=None,
        help="Number of radial samples to write for exported EAM/ADP tables.",
    )
    parser.add_argument(
        "--nrho",
        type=int,
        default=None,
        help="Number of density samples to write for exported EAM/ADP tables.",
    )


def run(args: argparse.Namespace) -> None:
    data = read_potential(args.input)
    write_lammps_potential(args.output, data, nr=args.nr, nrho=args.nrho)
