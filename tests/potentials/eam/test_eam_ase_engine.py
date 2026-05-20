from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from forgeff.calculator import make_calculator
from forgeff.io import read_potential


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _al_cell() -> Atoms:
    return Atoms(
        symbols=["Al", "Al"],
        positions=[(0.0, 0.0, 0.0), (2.86, 0.0, 0.0)],
        cell=(12.0, 12.0, 12.0),
        pbc=True,
    )


def test_eam_ase_engine_runs_on_example() -> None:
    path = _repo_root() / "examples/toml/eam/alloy/initial.toml"
    pot = read_potential(str(path))
    calc = make_calculator(pot, engine="ASE")
    atoms = _al_cell()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(stress))
