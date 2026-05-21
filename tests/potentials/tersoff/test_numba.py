from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.tersoff import Tersoff as ASETersoff

from forgeff.potentials.tersoff.data import TersoffParameters
from forgeff.potentials.tersoff.numba import NumbaTersoffCalculator


def _distorted_si_cell():
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


def test_numba_tersoff_matches_ase_reference() -> None:
    parameters = {
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

    atoms = _distorted_si_cell()

    ref = ASETersoff(parameters=parameters, skin=0.0)
    numba = NumbaTersoffCalculator(parameters=parameters, skin=0.0)

    atoms.calc = ref
    ref_energy = atoms.get_potential_energy()
    ref_forces = atoms.get_forces()
    ref_stress = atoms.get_stress()
    ref_energies = atoms.get_potential_energies()

    atoms.calc = numba
    numba_energy = atoms.get_potential_energy()
    numba_forces = atoms.get_forces()
    numba_stress = atoms.get_stress()
    numba_energies = atoms.get_potential_energies()

    assert numba_energy == pytest.approx(ref_energy, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(numba_forces, ref_forces, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_stress, ref_stress, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numba_energies, ref_energies, rtol=1e-10, atol=1e-10)
