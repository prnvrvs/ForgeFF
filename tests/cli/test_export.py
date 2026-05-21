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


def _write_tersoff_toml(path: Path) -> Path:
    path.write_text(
        """
[potential]
family = "tersoff"

[species]
order = ["Si", "C"]

[triplet.SiSiSi]
values = [3.0, 1.0, 0.0, 100390.0, 16.217, -0.59825, 0.78734, 0.0000011, 1.73222, 471.18, 2.85, 0.15, 2.4799, 1830.8]

[triplet.SiSiC]
values = [3.0, 1.0, 0.0, 100390.0, 16.217, -0.59825, 0.0, 0.0, 0.0, 0.0, 2.36, 0.15, 0.0, 0.0]

[triplet.SiCSi]
values = [3.0, 1.0, 0.0, 100390.0, 16.217, -0.59825, 0.0, 0.0, 0.0, 0.0, 2.85, 0.15, 0.0, 0.0]

[triplet.SiCC]
values = [3.0, 1.0, 0.0, 100390.0, 16.217, -0.59825, 0.78734, 0.0000011, 1.97205, 395.126, 2.36, 0.15, 2.9839, 1597.3111]

[triplet.CSiSi]
values = [3.0, 1.0, 0.0, 38049.0, 4.3484, -0.57058, 0.72751, 0.00000015724, 1.97205, 395.126, 2.36, 0.15, 2.9839, 1597.3111]

[triplet.CSiC]
values = [3.0, 1.0, 0.0, 38049.0, 4.3484, -0.57058, 0.0, 0.0, 0.0, 0.0, 1.95, 0.15, 0.0, 0.0]

[triplet.CCSi]
values = [3.0, 1.0, 0.0, 38049.0, 4.3484, -0.57058, 0.0, 0.0, 0.0, 0.0, 2.36, 0.15, 0.0, 0.0]

[triplet.CCC]
values = [3.0, 1.0, 0.0, 38049.0, 4.3484, -0.57058, 0.72751, 0.00000015724, 2.2119, 346.7, 1.95, 0.15, 3.4879, 1393.6]
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_export_cli_writes_lammps_from_toml(tmp_path: Path) -> None:
    sources = [
        ("examples/toml/eam/alloy/initial.toml", "Al99.eam.alloy"),
        ("examples/toml/eam/fs/initial.toml", "alcu.fs"),
        ("examples/toml/adp/alcu/initial.toml", "AlCu.adp"),
        (_write_tersoff_toml(tmp_path / "SiC.toml"), "SiC.tersoff"),
    ]
    for source, output_name in sources:
        output = tmp_path / output_name
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "forgeff",
                "export",
                str(source),
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


def test_export_cli_writes_lammps_from_npy(tmp_path: Path) -> None:
    sources = [
        ("examples/toml/eam/alloy/final.npy", "Al99.eam.alloy"),
        ("examples/toml/eam/fs/final.npy", "alcu.fs"),
        ("examples/toml/adp/alcu/final.npy", "AlCu.adp"),
    ]
    for source, output_name in sources:
        output = tmp_path / output_name
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "forgeff",
                "export",
                str(_repo_root() / source),
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


def test_export_cli_writes_lammps_from_toml_and_npy(tmp_path: Path) -> None:
    output = tmp_path / "SiC.tersoff"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "forgeff",
            "export",
            str(_write_tersoff_toml(tmp_path / "SiC.toml")),
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
