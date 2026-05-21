from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.eam import EAM as ASEEAM

from forgeff.io import read_potential
from forgeff.potentials.eam.numpy.adp_engine import NumpyADPEngine
from forgeff.potentials.eam.numpy.eam_engine import NumpyEAMEngine
from forgeff.potentials.eam.numba.adp_engine import NumbaADPEngine
from forgeff.potentials.eam.numba.eam_engine import NumbaEAMEngine


def test_nist_al99_eam_matches_ase() -> None:
    atoms = bulk("Al", "fcc", a=4.05) * (4, 4, 4)
    ase_calc = ASEEAM(potential="tests/data_path/nist/Al99.eam.alloy")
    numba_engine = NumbaEAMEngine(read_potential("tests/data_path/nist/Al99.eam.alloy"))

    ase_calc.calculate(atoms.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numba_res = numba_engine.calculate(atoms.copy())

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-12, atol=1e-12)


def test_nist_al99_eam_small_periodic_cell_matches_ase_numpy_and_numba() -> None:
    atoms = Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[3.3, 3.3, 3.3], pbc=True)
    pot = "tests/data_path/nist/Al99.eam.alloy"

    ase_calc = ASEEAM(potential=pot)
    numpy_engine = NumpyEAMEngine(read_potential(pot))
    numba_engine = NumbaEAMEngine(read_potential(pot))

    ase_atoms = atoms.copy()
    numpy_atoms = atoms.copy()
    numba_atoms = atoms.copy()

    ase_calc.calculate(ase_atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_res = numpy_engine.calculate(numpy_atoms)
    numba_res = numba_engine.calculate(numba_atoms)

    _assert_site_energies_sum_to_total(numpy_res)
    _assert_site_energies_sum_to_total(numba_res)
    np.testing.assert_allclose(numpy_res["energy"], ase_calc.results["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numpy_res["forces"], ase_calc.results["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numpy_res["stress"], ase_calc.results["stress"], rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-12, atol=1e-12)


def test_nist_alcu_adp_matches_ase_for_pure_al() -> None:
    atoms = bulk("Al", "fcc", a=4.05) * (4, 4, 4)
    ase_calc = ASEEAM(potential="tests/data_path/nist/AlCu.adp")
    numba_engine = NumbaADPEngine(read_potential("tests/data_path/nist/AlCu.adp"))

    ase_calc.calculate(atoms.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numba_res = numba_engine.calculate(atoms.copy())

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-12, atol=1e-12)


def test_nist_alcu_adp_small_periodic_cell_matches_ase_numpy_and_numba() -> None:
    atoms = Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[3.3, 3.3, 3.3], pbc=True)
    pot = "tests/data_path/nist/AlCu.adp"

    ase_calc = ASEEAM(potential=pot, form="adp")
    numpy_engine = NumpyADPEngine(read_potential(pot, form="adp"))
    numba_engine = NumbaADPEngine(read_potential(pot, form="adp"))

    ase_atoms = atoms.copy()
    numpy_atoms = atoms.copy()
    numba_atoms = atoms.copy()

    ase_calc.calculate(ase_atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_res = numpy_engine.calculate(numpy_atoms)
    numba_res = numba_engine.calculate(numba_atoms)

    _assert_site_energies_sum_to_total(numpy_res)
    _assert_site_energies_sum_to_total(numba_res)
    np.testing.assert_allclose(numpy_res["energy"], ase_calc.results["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numpy_res["forces"], ase_calc.results["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numpy_res["stress"], ase_calc.results["stress"], rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-12, atol=1e-12)


def _distorted_bcc_fe_with_h() -> Atoms:
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (2, 2, 2)
    h_position = np.dot(np.array([0.25, 0.50, 0.00]), atoms.cell.array)
    atoms += Atoms("H", positions=[h_position], cell=atoms.cell, pbc=True)
    rng = np.random.default_rng(1234)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    return atoms


def _distorted_bcc_fe_with_c() -> Atoms:
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (2, 2, 2)
    c_position = np.dot(np.array([0.25, 0.50, 0.00]), atoms.cell.array)
    atoms += Atoms("C", positions=[c_position], cell=atoms.cell, pbc=True)
    rng = np.random.default_rng(1234)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    return atoms


def _distorted_bcc_fe_cr_with_h() -> Atoms:
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (3, 3, 3)
    numbers = np.array([26] * 27 + [24] * 27, dtype=int)
    rng = np.random.default_rng(2025)
    rng.shuffle(numbers)
    atoms.numbers = numbers
    h_position = np.dot(np.array([0.25, 0.50, 0.00]), atoms.cell.array)
    atoms += Atoms("H", positions=[h_position], cell=atoms.cell, pbc=True)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    return atoms


def _distorted_bcc_fe_ni_cr_co_cu() -> Atoms:
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (3, 3, 3)
    species = np.array([26, 28, 24, 27, 29], dtype=int)
    counts = np.array([11, 11, 11, 11, 10], dtype=int)
    numbers = np.repeat(species, counts)
    rng = np.random.default_rng(2024)
    rng.shuffle(numbers)
    atoms.numbers = numbers
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    return atoms


def _assert_site_energies_sum_to_total(result: dict) -> None:
    assert "energies" in result
    np.testing.assert_allclose(np.sum(result["energies"]), result["energy"], rtol=1e-12, atol=1e-12)


def test_nist_fe_h_eam_fs_matches_ase_numpy_and_numba() -> None:
    atoms = _distorted_bcc_fe_with_h()
    pot = "tests/data_path/nist/Fe_H_Kumar2023.eam.fs"

    ase_calc = ASEEAM(potential=pot)
    numpy_engine = NumpyEAMEngine(read_potential(pot))
    numba_engine = NumbaEAMEngine(read_potential(pot))

    ase_atoms = atoms.copy()
    numpy_atoms = atoms.copy()
    numba_atoms = atoms.copy()

    ase_calc.calculate(ase_atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_res = numpy_engine.calculate(numpy_atoms)
    numba_res = numba_engine.calculate(numba_atoms)

    _assert_site_energies_sum_to_total(numpy_res)
    _assert_site_energies_sum_to_total(numba_res)
    np.testing.assert_allclose(numpy_res["energy"], ase_calc.results["energy"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_res["forces"], ase_calc.results["forces"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_res["stress"], ase_calc.results["stress"], rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-10, atol=1e-10)


def test_nist_fe_c_eam_alloy_matches_ase_numpy_and_numba() -> None:
    atoms = _distorted_bcc_fe_with_c()
    pot = "tests/data_path/nist/FeC.eam"

    ase_calc = ASEEAM(potential=pot, form="alloy")
    numpy_engine = NumpyEAMEngine(read_potential(pot))
    numba_engine = NumbaEAMEngine(read_potential(pot))

    ase_atoms = atoms.copy()
    numpy_atoms = atoms.copy()
    numba_atoms = atoms.copy()

    ase_calc.calculate(ase_atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_res = numpy_engine.calculate(numpy_atoms)
    numba_res = numba_engine.calculate(numba_atoms)

    _assert_site_energies_sum_to_total(numpy_res)
    _assert_site_energies_sum_to_total(numba_res)
    np.testing.assert_allclose(numpy_res["energy"], ase_calc.results["energy"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_res["forces"], ase_calc.results["forces"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_res["stress"], ase_calc.results["stress"], rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-10, atol=1e-10)


def test_nist_fe_ni_cr_co_cu_hea_eam_alloy_matches_ase_numpy_and_numba() -> None:
    atoms = _distorted_bcc_fe_ni_cr_co_cu()
    pot = "tests/data_path/nist/FeNiCrCoCu_with_ZBL.eam.alloy"

    ase_calc = ASEEAM(potential=pot)
    numpy_engine = NumpyEAMEngine(read_potential(pot))
    numba_engine = NumbaEAMEngine(read_potential(pot))

    ase_atoms = atoms.copy()
    numpy_atoms = atoms.copy()
    numba_atoms = atoms.copy()

    ase_calc.calculate(ase_atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_res = numpy_engine.calculate(numpy_atoms)
    numba_res = numba_engine.calculate(numba_atoms)

    _assert_site_energies_sum_to_total(numpy_res)
    _assert_site_energies_sum_to_total(numba_res)
    np.testing.assert_allclose(numpy_res["energy"], ase_calc.results["energy"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_res["forces"], ase_calc.results["forces"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_res["stress"], ase_calc.results["stress"], rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-10, atol=1e-10)


def test_nist_fe_cr_h_adp_matches_ase_numpy_and_numba() -> None:
    atoms = _distorted_bcc_fe_cr_with_h()
    pot = "tests/data_path/nist/Fe_Cr_H.adp.txt"

    ase_calc = ASEEAM(potential=pot, form="adp")
    numpy_engine = NumpyADPEngine(read_potential(pot, form="adp"))
    numba_engine = NumbaADPEngine(read_potential(pot, form="adp"))

    ase_atoms = atoms.copy()
    numpy_atoms = atoms.copy()
    numba_atoms = atoms.copy()

    ase_calc.calculate(ase_atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_res = numpy_engine.calculate(numpy_atoms)
    numba_res = numba_engine.calculate(numba_atoms)

    _assert_site_energies_sum_to_total(numpy_res)
    _assert_site_energies_sum_to_total(numba_res)
    np.testing.assert_allclose(numpy_res["energy"], ase_calc.results["energy"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_res["forces"], ase_calc.results["forces"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_res["stress"], ase_calc.results["stress"], rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-10, atol=1e-10)
