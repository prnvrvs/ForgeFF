"""`forgeff`."""

import argparse
import logging
from types import SimpleNamespace

import forgeff.error.cli
import forgeff.export.cli
import forgeff.evaluate.cli
import forgeff.grade.cli
import forgeff.template.cli
import forgeff.train.cli

from forgeff.parallel import world
from forgeff.utils import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger("forgeff")


def main() -> None:
    """Command."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    parser.add_argument(
        "-t",
        dest="template",
        action="store_true",
        help="Interactively generate a template.",
    )
    subparsers = parser.add_subparsers(dest="command")

    commands = {
        "error": forgeff.error.cli,
        "export": forgeff.export.cli,
        "train": forgeff.train.cli,
        "evaluate": forgeff.evaluate.cli,
        "grade": forgeff.grade.cli,
    }
    for key, value in commands.items():
        value.add_arguments(subparsers.add_parser(key))

    args = parser.parse_args()

    if args.command is None:
        if getattr(args, "template", False):
            forgeff.template.cli.run(
                SimpleNamespace(
                    family=None,
                    species=None,
                    form=None,
                    expression=None,
                    parameter_names=None,
                    output=None,
                    wizard=True,
                    dataset=None,
                    initial_output="initial.toml",
                    train_output=None,
                    energy_weight=None,
                    forces_weight=None,
                    stress_weight=None,
                    energy_offset_mode=None,
                    optimizer=None,
                    maxiter=None,
                    tol=None,
                    force=False,
                )
            )
            return
        parser.print_help()
        return

    commands[args.command].run(args)


if __name__ == "__main__":
    main()
