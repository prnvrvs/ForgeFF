from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.eam import EAM as ASEEAM
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.calculators.tersoff import Tersoff as ASETersoff

from forgeff.calculator import make_calculator
from forgeff.io import read_potential
from forgeff.potentials.ase.custom import CustomPairPotential
from forgeff.potentials.ase.numba_pair import NumbaPairPotential
from forgeff.potentials.eam.numpy.adp_engine import NumpyADPEngine
from forgeff.potentials.eam.numpy.eam_engine import NumpyEAMEngine
from forgeff.potentials.eam.numba.adp_engine import NumbaADPEngine
from forgeff.potentials.eam.numba.eam_engine import NumbaEAMEngine
from forgeff.potentials.sw.data import SWData
from forgeff.potentials.sw.numpy import NumpySWEngine
from forgeff.potentials.sw.numba import NumbaSWEngine
from forgeff.potentials.tersoff.data import TersoffData, TersoffParameters


ROOT = Path(__file__).resolve().parents[2]


def _strain_and_perturb(atoms: Atoms, seed: int, *, scale: float = 0.01) -> Atoms:
    rng = np.random.default_rng(seed)
    strained = atoms.copy()
    strain = np.eye(3) + rng.uniform(-0.015, 0.015, size=(3, 3))
    strained.set_cell(strained.cell.array @ strain, scale_atoms=True)
    strained.positions += rng.normal(scale=scale, size=strained.positions.shape)
    strained.wrap()
    return strained


def _assert_results_match(result: dict, energy: float, forces: np.ndarray, stress: np.ndarray, *, tol: float) -> None:
    np.testing.assert_allclose(result["energy"], energy, rtol=tol, atol=tol)
    np.testing.assert_allclose(result["forces"], forces, rtol=tol, atol=tol)
    np.testing.assert_allclose(result["stress"], stress, rtol=tol, atol=tol)


def _ase_eam_results(calc: ASEEAM, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
    calc.calculate(atoms.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    return calc.results["energy"], calc.results["forces"], calc.results["stress"]


@pytest.mark.parametrize(
    ("label", "potential", "form", "atoms", "seed"),
    [
        (
            "alloy",
            ROOT / "tests" / "data_path" / "nist" / "FeC.eam",
            "alloy",
            bulk("Fe", "bcc", a=2.86, cubic=True) * (2, 2, 2)
            + Atoms("C", positions=[[1.43, 1.43, 0.0]]),
            1101,
        ),
        (
            "fs",
            ROOT / "tests" / "data_path" / "nist" / "Fe_H_Kumar2023.eam.fs",
            "fs",
            bulk("Fe", "bcc", a=2.86, cubic=True) * (2, 2, 2)
            + Atoms("H", positions=[[1.43, 1.43, 0.0]]),
            1102,
        ),
        (
            "adp",
            ROOT / "tests" / "data_path" / "nist" / "AlCu.adp",
            "adp",
            bulk("Al", "fcc", a=4.05, cubic=True) * (2, 2, 2),
            1103,
        ),
    ],
)
def test_eam_family_numpy_numba_and_ase_match_random_cells(
    label: str,
    potential: Path,
    form: str,
    atoms: Atoms,
    seed: int,
) -> None:
    atoms = _strain_and_perturb(atoms, seed)
    pot = read_potential(str(potential), form=form)
    ase_energy, ase_forces, ase_stress = _ase_eam_results(ASEEAM(potential=str(potential), form=form), atoms)

    if label == "adp":
        numpy_res = NumpyADPEngine(pot).calculate(atoms.copy())
        numba_res = NumbaADPEngine(pot).calculate(atoms.copy())
    else:
        numpy_res = NumpyEAMEngine(pot).calculate(atoms.copy())
        numba_res = NumbaEAMEngine(pot).calculate(atoms.copy())

    _assert_results_match(numpy_res, ase_energy, ase_forces, ase_stress, tol=1e-10)
    _assert_results_match(numba_res, ase_energy, ase_forces, ase_stress, tol=1e-10)


@pytest.mark.parametrize(
    ("label", "potential", "form", "seed"),
    [
        ("alloy_small_periodic", ROOT / "tests" / "data_path" / "nist" / "Al99.eam.alloy", "alloy", 2101),
        ("adp_small_periodic", ROOT / "tests" / "data_path" / "nist" / "AlCu.adp", "adp", 2102),
    ],
)
def test_eam_family_numpy_numba_and_ase_match_small_random_periodic_cells(
    label: str,
    potential: Path,
    form: str,
    seed: int,
) -> None:
    del label
    atoms = Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=np.diag([3.3, 3.5, 3.7]), pbc=True)
    atoms = _strain_and_perturb(atoms, seed, scale=0.02)
    pot = read_potential(str(potential), form=form)
    ase_energy, ase_forces, ase_stress = _ase_eam_results(ASEEAM(potential=str(potential), form=form), atoms)

    if form == "adp":
        numpy_res = NumpyADPEngine(pot).calculate(atoms.copy())
        numba_res = NumbaADPEngine(pot).calculate(atoms.copy())
    else:
        numpy_res = NumpyEAMEngine(pot).calculate(atoms.copy())
        numba_res = NumbaEAMEngine(pot).calculate(atoms.copy())

    _assert_results_match(numpy_res, ase_energy, ase_forces, ase_stress, tol=1e-10)
    _assert_results_match(numba_res, ase_energy, ase_forces, ase_stress, tol=1e-10)


@pytest.mark.parametrize("form", ["lj", "morse"])
def test_pair_numpy_numba_and_ase_match_random_cells(form: str) -> None:
    atoms = _strain_and_perturb(
        bulk("Al", "fcc", a=4.05, cubic=True) * (2, 2, 2),
        1200,
    )
    if form == "lj":
        # ASE's LennardJones shifts the energy by u(rc). A large non-periodic
        # cutoff keeps the tested potential effectively identical to ForgeFF's
        # unshifted analytical LJ form while preserving force/stress coverage.
        atoms.pbc = False
        cutoff = 1000.0
        kwargs = {"epsilon": 0.20, "sigma": 2.40}
        numpy_calc = CustomPairPotential(
            expression="4*epsilon*((sigma/r)**12 - (sigma/r)**6)",
            parameter_names=["epsilon", "sigma"],
            cutoff=cutoff,
            **kwargs,
        )
        numba_calc = NumbaPairPotential(form="lj", cutoff=cutoff, **kwargs)
        ase_calc = LennardJones(epsilon=kwargs["epsilon"], sigma=kwargs["sigma"], rc=cutoff)
    else:
        cutoff = 3.2
        kwargs = {"De": 0.25, "a": 4.0, "re": 2.3}
        numpy_calc = CustomPairPotential(
            expression="De * (exp(-2.0 * a * (r - re)) - 2.0 * exp(-a * (r - re)))",
            parameter_names=["De", "a", "re"],
            cutoff=cutoff,
            **kwargs,
        )
        numba_calc = NumbaPairPotential(form="morse", cutoff=cutoff, **kwargs)
        ase_calc = MorsePotential(
            epsilon=kwargs["De"],
            rho0=kwargs["a"] * kwargs["re"],
            r0=kwargs["re"],
            rcut1=cutoff / kwargs["re"],
            rcut2=(cutoff + 0.1) / kwargs["re"],
        )

    atoms_numpy = atoms.copy()
    atoms_numba = atoms.copy()
    atoms_ase = atoms.copy()
    atoms_numpy.calc = numpy_calc
    atoms_numba.calc = numba_calc
    atoms_ase.calc = ase_calc

    energy_tol = 1e-10 if form == "lj" else 1e-12
    np.testing.assert_allclose(
        atoms_numpy.get_potential_energy(),
        atoms_ase.get_potential_energy(),
        rtol=energy_tol,
        atol=energy_tol,
    )
    np.testing.assert_allclose(atoms_numpy.get_forces(), atoms_ase.get_forces(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(atoms_numpy.get_stress(), atoms_ase.get_stress(), rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(
        atoms_numba.get_potential_energy(),
        atoms_ase.get_potential_energy(),
        rtol=energy_tol,
        atol=energy_tol,
    )
    np.testing.assert_allclose(atoms_numba.get_forces(), atoms_ase.get_forces(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(atoms_numba.get_stress(), atoms_ase.get_stress(), rtol=1e-12, atol=1e-12)


def _tersoff_data() -> TersoffData:
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
    return TersoffData.from_parameter_dict(parameters, species=["Si"])


def test_tersoff_numpy_numba_and_ase_match_random_cell() -> None:
    atoms = _strain_and_perturb(
        bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2),
        1300,
        scale=0.03,
    )
    data = _tersoff_data()
    ase_calc = ASETersoff(
        parameters={("Si", "Si", "Si"): TersoffParameters.from_list(data.parameter_table[0, 0, 0])},
        skin=0.0,
    )
    numpy_calc = make_calculator(data, engine="numpy", skin=0.0)
    numba_calc = make_calculator(data, engine="numba", skin=0.0)

    atoms_ase = atoms.copy()
    atoms_numpy = atoms.copy()
    atoms_numba = atoms.copy()
    atoms_ase.calc = ase_calc
    atoms_numpy.calc = numpy_calc
    atoms_numba.calc = numba_calc

    np.testing.assert_allclose(
        atoms_numpy.get_potential_energy(),
        atoms_ase.get_potential_energy(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(atoms_numpy.get_forces(), atoms_ase.get_forces(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(atoms_numpy.get_stress(), atoms_ase.get_stress(), rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(
        atoms_numba.get_potential_energy(),
        atoms_ase.get_potential_energy(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(atoms_numba.get_forces(), atoms_ase.get_forces(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(atoms_numba.get_stress(), atoms_ase.get_stress(), rtol=1e-10, atol=1e-10)


def _sw_data() -> SWData:
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


def test_sw_numpy_numba_and_matscipy_match_random_cell() -> None:
    pytest.importorskip("matscipy")
    from matscipy.calculators.manybody import StillingerWeber
    from matscipy.calculators.manybody.calculator import Manybody

    atoms = _strain_and_perturb(
        bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2),
        1400,
        scale=0.02,
    )
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
    matscipy_calc = Manybody(
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

    atoms_ref = atoms.copy()
    atoms_ref.calc = matscipy_calc
    numpy_res = NumpySWEngine(_sw_data()).calculate(atoms.copy())
    numba_res = NumbaSWEngine(_sw_data()).calculate(atoms.copy())

    _assert_results_match(
        numpy_res,
        atoms_ref.get_potential_energy(),
        atoms_ref.get_forces(),
        atoms_ref.get_stress(),
        tol=1e-10,
    )
    _assert_results_match(
        numba_res,
        atoms_ref.get_potential_energy(),
        atoms_ref.get_forces(),
        atoms_ref.get_stress(),
        tol=1e-10,
    )


def test_sw_numpy_numba_and_matscipy_match_small_periodic_cell() -> None:
    pytest.importorskip("matscipy")
    from matscipy.calculators.manybody import StillingerWeber
    from matscipy.calculators.manybody.calculator import Manybody

    atoms = _strain_and_perturb(
        Atoms(
            "Si2",
            positions=[
                [0.0, 0.0, 0.0],
                [1.05, 0.85, 1.20],
            ],
            cell=np.diag([3.0, 3.0, 3.0]),
            pbc=True,
        ),
        1401,
        scale=0.015,
    )
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
    matscipy_calc = Manybody(
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

    atoms_ref = atoms.copy()
    atoms_ref.calc = matscipy_calc
    numpy_res = NumpySWEngine(_sw_data()).calculate(atoms.copy())
    numba_res = NumbaSWEngine(_sw_data()).calculate(atoms.copy())

    _assert_results_match(
        numpy_res,
        atoms_ref.get_potential_energy(),
        atoms_ref.get_forces(),
        atoms_ref.get_stress(),
        tol=1e-10,
    )
    _assert_results_match(
        numba_res,
        atoms_ref.get_potential_energy(),
        atoms_ref.get_forces(),
        atoms_ref.get_stress(),
        tol=1e-10,
    )
