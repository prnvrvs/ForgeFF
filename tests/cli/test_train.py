from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path


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
