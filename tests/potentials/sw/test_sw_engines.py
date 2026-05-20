from __future__ import annotations

import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk

from forgeff.potentials.sw.data import SWData
from forgeff.potentials.sw.numpy import NumpySWEngine
from forgeff.potentials.sw.numba import NumbaSWEngine


def _make_sw_data() -> SWData:
    return SWData(
        species=["Si"],
        epsilon=2.1683,
        sigma=2.0951,
        costheta0=1.0 / 3.0,
        A=7.049556277,
        B=0.6022245584,
        p=4.0,
        a=1.8,
        lambda1=21.0,
        gamma=1.2,
    )


def _make_binary_sw_data() -> SWData:
    pair_parameters = np.asarray(
        [
            [
                [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 7.0],
                [1.1, 2.1, 3.1, 0.0, 4.1, 5.1, 6.1, 7.1],
            ],
            [
                [1.1, 2.1, 3.1, 0.0, 4.1, 5.1, 6.1, 7.1],
                [1.2, 2.2, 3.2, 0.0, 4.2, 5.2, 6.2, 7.2],
            ],
        ],
        dtype=float,
    )
    lambda_values = np.asarray(
        [
            [
                [0.1, 0.2],
                [0.2, 0.3],
            ],
            [
                [0.4, 0.5],
                [0.5, 0.6],
            ],
        ],
        dtype=float,
    )
    return SWData(species=["Al", "Cu"], pair_parameters=pair_parameters, lambda_values=lambda_values, costheta0=1.0 / 3.0)


def _reference_matscipy_calc():
    matscipy = pytest.importorskip("matscipy")
    from matscipy.calculators.manybody import StillingerWeber
    from matscipy.calculators.manybody.calculator import Manybody

    params = {
        "el": "Si",
        "epsilon": 2.1683,
        "sigma": 2.0951,
        "costheta0": 1.0 / 3.0,
        "A": 7.049556277,
        "B": 0.6022245584,
        "p": 4.0,
        "a": 1.8,
        "lambda1": 21.0,
        "gamma": 1.2,
    }
    sw = StillingerWeber(params)
    return Manybody(
        sw["atom_type"],
        sw["pair_type"],
        sw["F"],
        sw["G"],
        sw["d1F"],
        sw["d2F"],
        sw["d11F"],
        sw["d22F"],
        sw["d12F"],
        sw["d1G"],
        sw["d11G"],
        sw["d2G"],
        sw["d22G"],
        sw["d12G"],
        sw["cutoff"],
    )


def test_numpy_sw_matches_matscipy_reference() -> None:
    atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
    ref = _reference_matscipy_calc()
    atoms_ref = atoms.copy()
    atoms_ref.calc = ref

    engine = NumpySWEngine(_make_sw_data())
    res = engine.calculate(atoms)

    np.testing.assert_allclose(res["energy"], atoms_ref.get_potential_energy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res["forces"], atoms_ref.get_forces(), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(res["stress"], atoms_ref.get_stress(), rtol=1e-10, atol=1e-10)


def test_numba_sw_matches_numpy_sw() -> None:
    atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
    numpy_engine = NumpySWEngine(_make_sw_data())
    numba_engine = NumbaSWEngine(_make_sw_data())

    numpy_res = numpy_engine.calculate(atoms)
    numba_res = numba_engine.calculate(atoms)

    np.testing.assert_allclose(numba_res["energy"], numpy_res["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["forces"], numpy_res["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["stress"], numpy_res["stress"], rtol=1e-12, atol=1e-12)


def test_numba_sw_matches_numpy_sw_multispecies() -> None:
    atoms = Atoms(
        symbols=["Al", "Cu", "Al"],
        positions=np.asarray([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [1.2, 1.9, 0.0]], dtype=float),
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    numpy_engine = NumpySWEngine(_make_binary_sw_data())
    numba_engine = NumbaSWEngine(_make_binary_sw_data())

    numpy_res = numpy_engine.calculate(atoms)
    numba_res = numba_engine.calculate(atoms)

    np.testing.assert_allclose(numba_res["energy"], numpy_res["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["forces"], numpy_res["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["stress"], numpy_res["stress"], rtol=1e-12, atol=1e-12)
