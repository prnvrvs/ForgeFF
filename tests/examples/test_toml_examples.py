from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
import pytest

from forgeff.calculator import make_calculator
from forgeff.io import read_potential
from forgeff.train.setting import load_setting_train


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _al_supercell() -> Atoms:
    return bulk("Al", cubic=True) * (2, 2, 2)


def _binary_al_cu_cell() -> Atoms:
    return Atoms(
        symbols=["Al", "Cu"],
        positions=[(0.0, 0.0, 0.0), (2.8, 0.0, 0.0)],
        cell=(10.0, 10.0, 10.0),
        pbc=True,
    )


def _assert_calculation_runs(atoms: Atoms, calc) -> None:
    atoms = atoms.copy()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))


def _compare_backend_outputs(path: Path, atoms: Atoms, backend_a: str, backend_b: str) -> None:
    pot_a = read_potential(str(path))
    pot_a.backend = backend_a
    pot_b = read_potential(str(path))
    pot_b.backend = backend_b

    calc_a = make_calculator(pot_a, engine=backend_a)
    calc_b = make_calculator(pot_b, engine=backend_b)

    atoms_a = atoms.copy()
    atoms_a.calc = calc_a
    atoms_b = atoms.copy()
    atoms_b.calc = calc_b

    np.testing.assert_allclose(atoms_a.get_potential_energy(), atoms_b.get_potential_energy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(atoms_a.get_forces(), atoms_b.get_forces(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(atoms_a.get_stress(), atoms_b.get_stress(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "path",
    [
        "examples/toml/pairwise/morse/initial.toml",
        "examples/toml/pairwise/double_morse/initial.toml",
    ],
)
def test_pairwise_builtin_examples_match_on_unary_and_binary(path: str) -> None:
    full_path = _repo_root() / path
    _compare_backend_outputs(full_path, _al_supercell(), "numpy", "numba")
    _compare_backend_outputs(full_path, _binary_al_cu_cell(), "numpy", "numba")


def test_pairwise_custom_example_runs_on_unary_and_binary() -> None:
    path = _repo_root() / "examples/toml/pairwise/custom_expression/initial.toml"
    pot = read_potential(str(path))
    calc = make_calculator(pot, engine="numpy")
    _assert_calculation_runs(_al_supercell(), calc)
    _assert_calculation_runs(_binary_al_cu_cell(), calc)


def test_eam_alloy_example_matches_on_unary() -> None:
    path = _repo_root() / "examples/toml/eam/alloy/initial.toml"
    _compare_backend_outputs(path, _al_supercell(), "numpy", "numba")


def test_eam_alloy_binary_example_matches_on_binary() -> None:
    path = _repo_root() / "examples/toml/eam/alloy_binary/initial.toml"
    _compare_backend_outputs(path, _binary_al_cu_cell(), "numpy", "numba")


def test_eam_fs_example_matches_on_binary() -> None:
    path = _repo_root() / "examples/toml/eam/fs/initial.toml"
    pot = read_potential(str(path))
    calc = make_calculator(pot, engine="numba")
    _assert_calculation_runs(_binary_al_cu_cell(), calc)


def test_eam_fs_unary_example_runs_on_unary() -> None:
    path = _repo_root() / "examples/toml/eam/fs_unary/initial.toml"
    pot = read_potential(str(path))
    calc = make_calculator(pot, engine="numba")
    _assert_calculation_runs(_al_supercell(), calc)


def test_adp_example_runs_on_unary_and_binary() -> None:
    path = _repo_root() / "examples/toml/adp/alcu/initial.toml"
    pot = read_potential(str(path))
    calc = make_calculator(pot, engine="numba")
    _assert_calculation_runs(_al_supercell(), calc)
    _assert_calculation_runs(_binary_al_cu_cell(), calc)


def test_eam_train_settings_resolve_example_paths() -> None:
    morse = load_setting_train(_repo_root() / "examples/toml/pairwise/morse/forgeff.train.toml")
    double_morse = load_setting_train(_repo_root() / "examples/toml/pairwise/double_morse/forgeff.train.toml")
    alloy = load_setting_train(_repo_root() / "examples/toml/eam/alloy/forgeff.train.toml")
    alloy_binary = load_setting_train(_repo_root() / "examples/toml/eam/alloy_binary/forgeff.train.toml")
    fs = load_setting_train(_repo_root() / "examples/toml/eam/fs/forgeff.train.toml")
    fs_unary = load_setting_train(_repo_root() / "examples/toml/eam/fs_unary/forgeff.train.toml")
    adp = load_setting_train(_repo_root() / "examples/toml/adp/alcu/forgeff.train.toml")

    assert morse.potentials.initial.endswith("examples/toml/pairwise/morse/initial.toml")
    assert morse.potentials.final.endswith("examples/toml/pairwise/morse/final.npy")
    assert double_morse.potentials.initial.endswith("examples/toml/pairwise/double_morse/initial.toml")
    assert double_morse.potentials.final.endswith("examples/toml/pairwise/double_morse/final.npy")
    assert alloy.potentials.initial.endswith("examples/toml/eam/alloy/initial.toml")
    assert alloy.potentials.final.endswith("examples/toml/eam/alloy/final.npy")
    assert alloy.configurations.training[0].endswith("examples/toml/data/unary/training.cfg")
    assert alloy_binary.potentials.initial.endswith("examples/toml/eam/alloy_binary/initial.toml")
    assert alloy_binary.potentials.final.endswith("examples/toml/eam/alloy_binary/final.npy")
    assert alloy_binary.configurations.training[0].endswith("examples/toml/data/binary/training.cfg")

    assert fs.potentials.initial.endswith("examples/toml/eam/fs/initial.toml")
    assert fs.potentials.final.endswith("examples/toml/eam/fs/final.npy")
    assert fs.configurations.training[0].endswith("examples/toml/data/binary/training.cfg")
    assert fs_unary.potentials.initial.endswith("examples/toml/eam/fs_unary/initial.toml")
    assert fs_unary.potentials.final.endswith("examples/toml/eam/fs_unary/final.npy")
    assert fs_unary.configurations.training[0].endswith("examples/toml/data/unary/training.cfg")

    assert adp.potentials.initial.endswith("examples/toml/adp/alcu/initial.toml")
    assert adp.potentials.final.endswith("examples/toml/adp/alcu/final.npy")
    assert adp.configurations.training[0].endswith("examples/toml/data/binary/training.cfg")
