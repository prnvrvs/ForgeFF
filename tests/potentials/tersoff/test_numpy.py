from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.tersoff import Tersoff as ASETersoff

from forgeff.potentials.tersoff.data import TersoffParameters
from forgeff.potentials.tersoff.numpy import NumpyTersoffCalculator
from forgeff.potentials.tersoff.numba import NumbaTersoffCalculator


def _parameters() -> dict[tuple[str, str, str], TersoffParameters]:
    return {
        ("Si", "Si", "Si"): TersoffParameters(
            m=1.0,
            gamma=1.0,
            lambda3=1.2,
            c=1.0,
            d=0.5,
            h=-0.3,
            n=1.0,
            beta=1.0,
            lambda2=1.5,
            B=0.8,
            R=3.0,
            D=0.2,
            lambda1=2.6,
            A=1.4,
        )
    }


def _distorted_si_cell() -> Atoms:
    rng = np.random.default_rng(20260521)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    cell = atoms.cell.array.copy()
    cell += np.array(
        [
            [0.06, 0.02, -0.01],
            [0.00, -0.04, 0.03],
            [0.01, -0.02, 0.05],
        ]
    )
    atoms.set_cell(cell, scale_atoms=True)
    atoms.positions += rng.normal(scale=0.03, size=atoms.positions.shape)
    atoms.wrap()
    return atoms


@pytest.mark.parametrize(
    "atoms",
    [
        _distorted_si_cell(),
        Atoms("Si", cell=np.eye(3) * 12.0, pbc=True),
    ],
)
def test_numpy_tersoff_matches_ase_and_numba(atoms: Atoms) -> None:
    parameters = _parameters()
    ase = ASETersoff(parameters=parameters, skin=0.0)
    numpy = NumpyTersoffCalculator(parameters=parameters, skin=0.0)
    numba = NumbaTersoffCalculator(parameters=parameters, skin=0.0)

    atoms_ase = atoms.copy()
    atoms_ase.calc = ase
    ase_energy = atoms_ase.get_potential_energy()
    ase_forces = atoms_ase.get_forces()
    ase_stress = atoms_ase.get_stress() if len(atoms) else np.zeros(6)
    ase_energies = atoms_ase.get_potential_energies()

    atoms_numpy = atoms.copy()
    atoms_numpy.calc = numpy
    numpy_energy = atoms_numpy.get_potential_energy()
    numpy_forces = atoms_numpy.get_forces()
    numpy_stress = atoms_numpy.get_stress() if len(atoms) else np.zeros(6)
    numpy_energies = atoms_numpy.get_potential_energies()

    atoms_numba = atoms.copy()
    atoms_numba.calc = numba
    numba_energy = atoms_numba.get_potential_energy()
    numba_forces = atoms_numba.get_forces()
    numba_stress = atoms_numba.get_stress() if len(atoms) else np.zeros(6)
    numba_energies = atoms_numba.get_potential_energies()

    assert numpy_energy == pytest.approx(ase_energy, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(numpy_forces, ase_forces, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_stress, ase_stress, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_energies, ase_energies, rtol=1e-10, atol=1e-10)

    assert numba_energy == pytest.approx(ase_energy, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(numba_forces, ase_forces, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_stress, ase_stress, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_energies, ase_energies, rtol=1e-10, atol=1e-10)
