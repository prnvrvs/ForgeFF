from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from ase.build import bulk
from ase.calculators.eam import EAM as ASEEAM

from forgeff.io import read_potential, write_lammps_potential


def _assert_same_results(calc_a, calc_b, atoms) -> None:
    atoms_a = atoms.copy()
    atoms_b = atoms.copy()
    atoms_a.calc = calc_a
    atoms_b.calc = calc_b

    assert_allclose(atoms_a.get_potential_energy(), atoms_b.get_potential_energy(), rtol=1e-12, atol=1e-12)
    assert_allclose(atoms_a.get_forces(), atoms_b.get_forces(), rtol=1e-11, atol=1e-11)
    assert_allclose(atoms_a.get_stress(), atoms_b.get_stress(), rtol=1e-10, atol=1e-10)


def test_write_lammps_alloy_roundtrip(tmp_path: Path) -> None:
    source = "tests/data_path/nist/FeNiCrCoCu_with_ZBL.eam.alloy"
    pot = read_potential(source, form="alloy")
    output = tmp_path / "FeNiCrCoCu_with_ZBL.eam.alloy"
    write_lammps_potential(output, pot)

    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (3, 3, 3)
    species = np.array([26, 28, 24, 27, 29], dtype=int)
    counts = np.array([11, 11, 11, 11, 10], dtype=int)
    numbers = np.repeat(species, counts)
    rng = np.random.default_rng(2024)
    rng.shuffle(numbers)
    atoms.numbers = numbers
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)

    _assert_same_results(ASEEAM(potential=source), ASEEAM(potential=str(output)), atoms)


def test_write_lammps_fs_roundtrip(tmp_path: Path) -> None:
    source = "tests/data_path/nist/Fe_H_Kumar2023.eam.fs"
    pot = read_potential(source, form="fs")
    output = tmp_path / "Fe_H_Kumar2023.eam.fs"
    write_lammps_potential(output, pot)

    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (3, 3, 3)
    atoms.numbers = np.array([26] * 54, dtype=int)
    atoms.numbers[0] = 1
    rng = np.random.default_rng(2025)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)

    _assert_same_results(ASEEAM(potential=source, form="fs"), ASEEAM(potential=str(output), form="fs"), atoms)


def test_write_lammps_adp_roundtrip(tmp_path: Path) -> None:
    source = "tests/data_path/nist/AlCu.adp"
    pot = read_potential(source, form="adp")
    output = tmp_path / "AlCu.adp"
    write_lammps_potential(output, pot)

    atoms = bulk("Al", "fcc", a=4.05, cubic=True) * (2, 2, 2)
    atoms.numbers = np.array([13, 29] * 16, dtype=int)
    rng = np.random.default_rng(2026)
    rng.shuffle(atoms.numbers)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)

    _assert_same_results(ASEEAM(potential=source, form="adp"), ASEEAM(potential=str(output), form="adp"), atoms)
