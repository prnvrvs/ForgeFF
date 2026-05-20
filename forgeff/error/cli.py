"""`forgeff error`."""

from __future__ import annotations

import argparse
from pathlib import Path

from forgeff.evaluate.evaluator import Evaluator
from forgeff.io import read_potential
from forgeff.io.utils import get_dummy_species, read_images
from forgeff.loss import ErrorPrinter, format_error_statistics
from forgeff.parallel import is_master, world


def analyze_error_statistics(
    potential: str | Path,
    datasets: list[str | Path],
    *,
    species: list[int] | None = None,
    engine: str = "numpy",
    comm=world,
) -> dict[str, dict[str, float]]:
    """Run a potential on a dataset and collect error statistics."""
    species = list(species) if species is not None else None
    dataset_paths = [str(dataset) for dataset in datasets]
    images = read_images(dataset_paths, species=species, comm=comm, title="configurations")
    if species is None:
        species = get_dummy_species(images)

    pot_data = read_potential(str(potential))
    pot_data.species = species
    evaluator = Evaluator(pot_data, engine=engine, comm=comm)
    images_eval = evaluator.evaluate(images)
    return ErrorPrinter(images_eval).calculate()


def print_error_statistics(errors: dict[str, dict[str, float]]) -> None:
    """Print error statistics in a compact table."""
    print(format_error_statistics(errors))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments."""
    parser.add_argument("potential", help="Potential file, such as final.npy or final.toml.")
    parser.add_argument(
        "dataset",
        nargs="+",
        help="Dataset file(s) containing targets, such as .cfg or .traj files.",
    )
    parser.add_argument(
        "--engine",
        default="numpy",
        help="Calculator engine name (for example, numpy or numba).",
    )
    parser.add_argument(
        "--species",
        nargs="*",
        type=int,
        help="Optional atomic-number order for MLIP .cfg files.",
    )


def run(args: argparse.Namespace) -> None:
    """Run the error analysis CLI."""
    errors = analyze_error_statistics(
        args.potential,
        args.dataset,
        species=args.species,
        engine=args.engine,
        comm=world,
    )
    if is_master(world):
        print_error_statistics(errors)


def main() -> None:
    """Command."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    add_arguments(parser)
    args = parser.parse_args()
    run(args)
