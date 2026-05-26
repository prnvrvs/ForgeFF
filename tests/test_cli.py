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


def test_template_subcommand_is_removed() -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-m", "forgeff", "template"],
        check=False,
        cwd=_repo_root(),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "usage:" in result.stderr.lower()


def test_root_template_flag_invokes_wizard(monkeypatch) -> None:
    import forgeff.cli as cli

    captured: dict[str, object] = {}

    def fake_run(args) -> None:
        captured["args"] = args

    monkeypatch.setattr(cli.forgeff.template.cli, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [sys.argv[0], "-t"])

    cli.main()

    assert captured["args"].wizard is True
    assert captured["args"].family is None


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
