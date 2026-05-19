from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_example(script: str) -> str:
    env = os.environ.copy()
    env["NUMBA_DISABLE_JIT"] = "1"
    result = subprocess.run(
        [sys.executable, str(_repo_root() / script)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout


def test_simulated_annealing_example_runs() -> None:
    stdout = _run_example("examples/python/3.anneal.py")
    initial = float(re.search(r"initial loss: ([0-9.eE+-]+)", stdout).group(1))
    final = float(re.search(r"final loss: ([0-9.eE+-]+)", stdout).group(1))
    assert final < initial


def test_genetic_algorithm_example_runs() -> None:
    stdout = _run_example("examples/python/4.ga.py")
    initial = float(re.search(r"initial best fitness: ([0-9.eE+-]+)", stdout).group(1))
    final = float(re.search(r"final best fitness: ([0-9.eE+-]+)", stdout).group(1))
    assert final <= initial

