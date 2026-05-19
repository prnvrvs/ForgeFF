"""`forgeff train`."""

import argparse

from forgeff.error.cli import analyze_error_statistics, print_error_statistics
from forgeff.train.setting import load_setting_train
from forgeff.parallel import world
from forgeff.utils import measure_time

from .trainer import train_from_setting


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def run(args: argparse.Namespace) -> None:
    """Run."""
    setting = load_setting_train(args.setting)
    with measure_time("total"):
        train_from_setting(args.setting, comm=world)
    errors = analyze_error_statistics(
        setting.potentials.final,
        setting.configurations.training,
        species=setting.common.species or None,
        engine=getattr(setting.common, "engine", None) or "numpy",
        comm=world,
    )
    print_error_statistics(errors)
    print("Training complete.")


def main() -> None:
    """Command."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    add_arguments(parser)
    args = parser.parse_args()
    run(args)
