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


def test_export_cli_writes_lammps_alloy(tmp_path: Path) -> None:
    output = tmp_path / "Al99.eam.alloy"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "forgeff",
            "export",
            str(_repo_root() / "tests/data_path/nist/Al99.eam.alloy"),
            str(output),
        ],
        cwd=_repo_root(),
        env=_subprocess_env(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert output.exists()

