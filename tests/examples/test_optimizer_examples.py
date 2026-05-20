from __future__ import annotations

import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_example(script: str) -> str:
    env = os.environ.copy()
    env["NUMBA_DISABLE_JIT"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_repo_root()),
            sysconfig.get_path("purelib") or "",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-S", str(_repo_root() / script)],
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


def test_training_examples_run() -> None:
    stdout = _run_example("examples/python/0.train.py")
    initial = float(re.search(r"initial loss: ([0-9.eE+-]+)", stdout).group(1))
    final = float(re.search(r"final loss: ([0-9.eE+-]+)", stdout).group(1))
    assert final <= initial
    assert "training indices:" in stdout

    stdout = _run_example("examples/python/1.evaluate.py")
    assert "Training set error statistics:" in stdout
    assert "Testing set error statistics:" in stdout

    stdout = _run_example("examples/python/2.grade.py")
    assert "graded training set:" in stdout
    assert "graded testing set:" in stdout
