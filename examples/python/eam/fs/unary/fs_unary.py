"""
Finnis-Sinclair EAM, unary Al
=============================

Run the Finnis-Sinclair EAM training example on a unary Al setup.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from forgeff.parallel import world
from forgeff.train.trainer import train_from_setting


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        setting = candidate / "examples" / "toml" / "eam" / "fs_unary" / "forgeff.train.toml"
        if setting.exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root.")


@contextmanager
def _pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def main() -> None:
    setting_path = _repo_root() / "examples" / "toml" / "eam" / "fs_unary" / "forgeff.train.toml"
    print("Finnis-Sinclair EAM, unary Al")
    print(f"Setting: {setting_path}")
    with _pushd(setting_path.parent):
        train_from_setting(str(setting_path), comm=world)
    print("Done.")


if __name__ == "__main__":
    main()
