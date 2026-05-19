from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forgeff.error.cli import analyze_error_statistics


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "potential,dataset,engine",
    [
        ("examples/toml/pairwise/morse/final.npy", "examples/toml/data/unary/training.cfg", "numpy"),
        ("examples/toml/eam/alloy/final.npy", "examples/toml/data/unary/training.cfg", "numpy"),
        ("examples/toml/adp/alcu/final.npy", "examples/toml/data/binary/training.cfg", "numba"),
    ],
)
def test_error_cli_runs_on_example_dataset(potential: str, dataset: str, engine: str) -> None:
    errors = analyze_error_statistics(
        _repo_root() / potential,
        [_repo_root() / dataset],
        engine=engine,
    )

    assert set(errors) == {"energy", "energy_per_atom", "forces", "stress"}
    assert np.isfinite(errors["energy"]["RMS"])
    assert np.isfinite(errors["forces"]["RMS"])
    assert errors["stress"]["RMS"] >= 0.0
