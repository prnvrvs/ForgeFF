"""Run one of the TOML training examples from the command line.

This script is intentionally small and direct. It shows how to point ForgeFF
at a training setting file, let the library do the fit, and then print a
compact error summary for the trained model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from contextlib import contextmanager

from forgeff.error.cli import analyze_error_statistics, print_error_statistics
from forgeff.train.setting import load_setting_train
from forgeff.parallel import world
from forgeff.train.trainer import train_from_setting


ROOT = Path(__file__).resolve().parent
DEFAULT_SETTING = ROOT / "eam" / "alloy" / "forgeff.train.toml"


@contextmanager
def _pushd(path: Path):
    """Temporarily change the working directory."""
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def main() -> None:
    """Run the training example."""
    parser = argparse.ArgumentParser(
        description="Run one ForgeFF TOML training example."
    )
    parser.add_argument(
        "--setting",
        default=str(DEFAULT_SETTING),
        help="Path to a forgeff.train.toml file.",
    )
    args = parser.parse_args()
    setting_path = Path(args.setting).resolve()
    setting = load_setting_train(str(setting_path))

    print(f"Training setting: {setting_path}")
    with _pushd(setting_path.parent):
        train_from_setting(str(setting_path), comm=world)
        errors = analyze_error_statistics(
            setting.potentials.final,
            setting.configurations.training,
            species=setting.common.species or None,
            engine=getattr(setting.common, "engine", None) or "numpy",
            comm=world,
        )
    print_error_statistics(errors)
    print("Training complete.")


if __name__ == "__main__":
    main()
