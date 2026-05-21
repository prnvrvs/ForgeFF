from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
import pytest

from forgeff.calculator import make_calculator
from forgeff.io import read_potential
from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.tersoff.data import TersoffData, TersoffParameters


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _al_dimer() -> Atoms:
    return Atoms(
        symbols=["Al", "Al"],
        positions=[(0.0, 0.0, 0.0), (2.2, 0.0, 0.0)],
        cell=(12.0, 12.0, 12.0),
        pbc=True,
    )


def _compare(calc_a, calc_b, atoms: Atoms) -> None:
    atoms_a = atoms.copy()
    atoms_b = atoms.copy()
    atoms_a.calc = calc_a
    atoms_b.calc = calc_b
    np.testing.assert_allclose(atoms_a.get_potential_energy(), atoms_b.get_potential_energy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(atoms_a.get_forces(), atoms_b.get_forces(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(atoms_a.get_stress(), atoms_b.get_stress(), rtol=1e-12, atol=1e-12)


def _strained_perturbed_al(seed: int) -> Atoms:
    rng = np.random.default_rng(seed)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True) * (2, 2, 2)
    strain = np.eye(3) + rng.uniform(-0.02, 0.02, size=(3, 3))
    atoms.set_cell(atoms.cell.array @ strain, scale_atoms=True)
    atoms.positions += rng.uniform(-0.05, 0.05, size=atoms.positions.shape)
    atoms.wrap()
    return atoms


def test_lj_engine_ase_matches_ase_calculator(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "lj.toml",
        """
[potential]
family = "analytical"
form = "lj"
engine = "ASE"
cutoff = 8.0
initial = [0.20, 3.00]
""".lstrip(),
    )
    pot = read_potential(str(path))
    calc = make_calculator(pot, engine="ASE")
    direct = LennardJones(epsilon=0.20, sigma=3.00)
    _compare(calc, direct, _al_dimer())


def test_morse_engine_ase_matches_ase_calculator(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "morse.toml",
        """
[potential]
family = "analytical"
form = "morse"
engine = "ASE"
cutoff = 8.0
initial = [0.25, 4.0, 2.3]
""".lstrip(),
    )
    pot = read_potential(str(path))
    calc = make_calculator(pot, engine="ASE")
    direct = MorsePotential(epsilon=0.25, rho0=4.0 * 2.3, r0=2.3)
    _compare(calc, direct, _al_dimer())


def test_morse_numpy_numba_and_ase_match_on_perturbed_cells(tmp_path: Path) -> None:
    cutoff = 3.2
    de = 0.25
    a = 4.0
    re = 2.3
    path = _write_text(
        tmp_path / "morse.toml",
        f"""
[potential]
family = "analytical"
form = "morse"
cutoff = {cutoff}
initial = [{de}, {a}, {re}]

[potential.calculator_kwargs]
rcut1 = {cutoff / re}
rcut2 = {(cutoff + 0.1) / re}
""".lstrip(),
    )

    pot_numpy = read_potential(str(path))
    pot_numba = read_potential(str(path))
    pot_ase = read_potential(str(path))
    calc_numpy = make_calculator(pot_numpy, engine="numpy")
    calc_numba = make_calculator(pot_numba, engine="numba")
    calc_ase = make_calculator(pot_ase, engine="ASE")

    for seed in range(5):
        atoms = _strained_perturbed_al(seed)
        atoms_numpy = atoms.copy()
        atoms_numba = atoms.copy()
        atoms_ase = atoms.copy()

        atoms_numpy.calc = calc_numpy
        atoms_numba.calc = calc_numba
        atoms_ase.calc = calc_ase

        np.testing.assert_allclose(
            atoms_numpy.get_potential_energy(),
            atoms_ase.get_potential_energy(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(atoms_numpy.get_forces(), atoms_ase.get_forces(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(atoms_numpy.get_stress(), atoms_ase.get_stress(), rtol=1e-12, atol=1e-12)

        np.testing.assert_allclose(
            atoms_numba.get_potential_energy(),
            atoms_ase.get_potential_energy(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(atoms_numba.get_forces(), atoms_ase.get_forces(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(atoms_numba.get_stress(), atoms_ase.get_stress(), rtol=1e-12, atol=1e-12)


def test_emt_is_rejected_as_a_fitting_target() -> None:
    pot = ASEData(engine="ASE", calculator_kwargs={"calculator": "EMT"})
    with pytest.raises(ValueError, match="EMT"):
        make_calculator(pot, engine="ASE")


def test_tersoff_falls_back_from_ase_to_numba() -> None:
    data = TersoffData.from_parameter_dict(
            {
                ("Si", "Si", "Si"): TersoffParameters(
                    m=1.0,
                    gamma=1.0,
                    lambda3=0.0,
                    c=1.0,
                    d=1.0,
                    h=-1.0 / 3.0,
                    n=1.0,
                    beta=1.0,
                    lambda2=1.0,
                    B=1.0,
                    R=2.0,
                    D=0.5,
                    lambda1=1.0,
                    A=1.0,
                )
            }
        )
    with pytest.warns(RuntimeWarning, match="ASE does not support Tersoff fitting"):
        calc = make_calculator(data, engine="ASE")
    assert calc.__class__.__name__ == "NumbaTersoffCalculator"


def test_adp_falls_back_from_ase_to_numba() -> None:
    pot = read_potential(str(_repo_root() / "examples/toml/adp/alcu/initial.toml"))
    with pytest.warns(RuntimeWarning, match="ASE does not support ADP fitting"):
        calc = make_calculator(pot, engine="ASE")
    assert calc.__class__.__name__ == "ADP"


def test_multispecies_lj_numpy_and_numba_match(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "binary_lj.toml",
        """
[potential]
family = "analytical"
form = "lj"
cutoff = 8.0

[species]
order = ["Al", "Cu"]

[pair.AlAl]
initial = [0.20, 2.60]

[pair.AlCu]
initial = [0.18, 2.55]

[pair.CuCu]
initial = [0.25, 2.70]
""".lstrip(),
    )
    pot_numpy = read_potential(str(path))
    pot_numba = read_potential(str(path))
    pot_numpy.engine = "numpy"
    pot_numba.engine = "numba"
    calc_numpy = make_calculator(pot_numpy, engine="numpy")
    calc_numba = make_calculator(pot_numba, engine="numba")

    atoms = Atoms(
        symbols=["Al", "Cu", "Al"],
        positions=[(0.0, 0.0, 0.0), (2.2, 0.0, 0.0), (0.0, 2.3, 0.0)],
        cell=(12.0, 12.0, 12.0),
        pbc=True,
    )

    atoms_numpy = atoms.copy()
    atoms_numpy.calc = calc_numpy
    atoms_numba = atoms.copy()
    atoms_numba.calc = calc_numba

    np.testing.assert_allclose(atoms_numpy.get_potential_energy(), atoms_numba.get_potential_energy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(atoms_numpy.get_forces(), atoms_numba.get_forces(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(atoms_numpy.get_stress(), atoms_numba.get_stress(), rtol=1e-12, atol=1e-12)
