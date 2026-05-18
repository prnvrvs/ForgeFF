"""Tests for CLI."""

import subprocess
import sys

import pytest


def test_help_main() -> None:
    """Test `forgeff -h`."""
    result = subprocess.run([sys.executable, "-m", "forgeff", "-h"], check=False)
    assert result.returncode == 0


@pytest.mark.parametrize("command", ["train", "evaluate", "grade"])
def test_help_sub(command: str) -> None:
    """Test `forgeff command -h`."""
    result = subprocess.run([sys.executable, "-m", f"forgeff.{command}", "-h"], check=False)
    assert result.returncode == 0
