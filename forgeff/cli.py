"""`forgeff`."""

import argparse
import logging

import forgeff.error.cli
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
    subparsers = parser.add_subparsers(dest="command")

    commands = {
        "error": forgeff.error.cli,
        "train": forgeff.train.cli,
        "evaluate": forgeff.evaluate.cli,
        "grade": forgeff.grade.cli,
        "template": forgeff.template.cli,
    }
    for key, value in commands.items():
        value.add_arguments(subparsers.add_parser(key))

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands[args.command].run(args)


if __name__ == "__main__":
    main()
