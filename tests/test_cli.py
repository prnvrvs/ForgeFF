"""Tests for CLI."""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_repo_root()),
            sysconfig.get_path("purelib") or "",
        ]
    )
    return env


def test_help_main() -> None:
    """Test `forgeff -h`."""
    result = subprocess.run(
        [sys.executable, "-S", "-m", "forgeff", "-h"],
        check=False,
        cwd=_repo_root(),
        env=_subprocess_env(),
    )
    assert result.returncode == 0


@pytest.mark.parametrize("command", ["train", "evaluate", "grade"])
def test_help_sub(command: str) -> None:
    """Test `forgeff command -h`."""
    result = subprocess.run(
        [sys.executable, "-S", "-m", f"forgeff.{command}", "-h"],
        check=False,
        cwd=_repo_root(),
        env=_subprocess_env(),
    )
    assert result.returncode == 0
