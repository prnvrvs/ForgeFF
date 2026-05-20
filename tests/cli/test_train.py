from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import forgeff.train.cli as train_cli


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def test_train_cli_prints_error_statistics() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "forgeff",
            "train",
            str(_repo_root() / "examples/toml/eam/alloy/forgeff.train.toml"),
        ],
        cwd=_repo_root(),
        env=_subprocess_env(),
        check=False,
        text=True,
        capture_output=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Error statistics:" in output, output
    assert "| Metric" in output, output
    assert "Training complete." in output, output


def test_train_cli_prints_only_on_master(monkeypatch, capsys) -> None:
    monkeypatch.setattr(train_cli, "load_setting_train", lambda _: SimpleNamespace(
        potentials=SimpleNamespace(final="final.npy"),
        configurations=SimpleNamespace(training=[]),
        common=SimpleNamespace(species=[], engine="numpy"),
    ))
    monkeypatch.setattr(train_cli, "train_from_setting", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_cli, "analyze_error_statistics", lambda *args, **kwargs: {})
    called = []
    monkeypatch.setattr(train_cli, "print_error_statistics", lambda errors: called.append(errors))
    monkeypatch.setattr(train_cli, "world", SimpleNamespace(rank=1, size=2))

    train_cli.run(SimpleNamespace(setting="dummy.toml"))

    assert called == []
