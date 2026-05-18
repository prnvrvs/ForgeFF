from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from forgeff.potentials.ase.custom import CustomPairPotential
from forgeff.potentials.ase.numba_pair import NumbaPairPotential


@pytest.mark.parametrize(
    ("form", "expression", "parameter_names", "parameters"),
    [
        ("bornmayer", "A * exp(-r / rho)", ["A", "rho"], [2.0, 1.5]),
        ("constant", "c", ["c"], [0.3]),
        ("coul", "14.3996454784255 * q1 * q2 / r", ["q1", "q2"], [1.2, -0.8]),
        ("exponential", "A * r**n", ["A", "n"], [0.7, -1.5]),
        ("hbnd", "A / r**12 - B / r**10", ["A", "B"], [0.2, 0.15]),
        ("zero", "0.0", [], []),
    ],
)
def test_builtin_aliases_match_custom_expression(
    form: str,
    expression: str,
    parameter_names: list[str],
    parameters: list[float],
) -> None:
    atoms = Atoms(
        "Al2",
        positions=[[0.0, 0.0, 0.0], [1.9, 0.1, 0.2]],
        cell=np.eye(3) * 12.0,
        pbc=True,
    )

    custom_kwargs = {"expression": expression, "cutoff": 8.0}
    numba_kwargs = {"form": form, "cutoff": 8.0}
    if parameter_names:
        custom_kwargs["parameter_names"] = parameter_names
        for name, value in zip(parameter_names, parameters, strict=True):
            custom_kwargs[name] = value
            numba_kwargs[name] = value

    custom = CustomPairPotential(**custom_kwargs)
    numba = NumbaPairPotential(**numba_kwargs)

    atoms.calc = custom
    custom_energy = atoms.get_potential_energy()
    custom_forces = atoms.get_forces()
    custom_stress = atoms.get_stress()

    atoms.calc = numba
    numba_energy = atoms.get_potential_energy()
    numba_forces = atoms.get_forces()
    numba_stress = atoms.get_stress()

    assert custom_energy == pytest.approx(numba_energy, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(custom_forces, numba_forces, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(custom_stress, numba_stress, rtol=1e-12, atol=1e-12)
