from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_train_cli_prints_error_statistics() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgeff",
            "train",
            str(_repo_root() / "examples/toml/eam/alloy/forgeff.train.toml"),
        ],
        cwd=_repo_root(),
        check=False,
        text=True,
        capture_output=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Error statistics:" in output, output
    assert "| Metric" in output, output
    assert "Training complete." in output, output
