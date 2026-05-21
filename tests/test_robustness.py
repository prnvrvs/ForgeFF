from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms
import pytest

from forgeff.calculator import make_calculator
from forgeff.io import read_potential
from forgeff.io import write_potential
from forgeff.loss import ErrorPrinter, LossFunctionStress
from forgeff.loss import LossSetting
from forgeff.loss import _resolve_species_energy_offsets
from forgeff.potentials.sw.data import SWData


def test_empty_atoms_stress_loss_skips_non_3d_cells() -> None:
    atoms = Atoms()
    atoms.calc = SimpleNamespace(
        results={"stress": np.zeros(6, dtype=float)},
        targets={"stress": np.zeros(6, dtype=float)},
    )

    pot_data = SimpleNamespace(number_of_parameters_optimized=1)
    loss = LossFunctionStress([atoms], pot_data)

    assert loss.idcs_str.size == 0
    assert loss.calculate() == 0.0


def test_empty_atoms_error_printer_skips_non_3d_cells() -> None:
    atoms = Atoms()
    atoms.calc = SimpleNamespace(
        results={"energy": 0.0, "stress": np.zeros(6, dtype=float)},
        targets={"energy": 0.0, "stress": np.zeros(6, dtype=float)},
    )

    printer = ErrorPrinter([atoms])

    assert printer.idcs_str.size == 0
    errors = printer.calculate()
    assert errors["stress"]["N"] == 0


def test_manual_species_energy_offsets_are_added_to_predictions() -> None:
    pot_data = SWData(species=["Si"])
    pot_data.species_energy_offsets = {"Si": -1.5}

    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.calc = make_calculator(pot_data, engine="numba")

    assert atoms.get_potential_energy() == pytest.approx(-1.5)
    np.testing.assert_allclose(atoms.calc.results["energies"], np.array([-1.5]))


def test_regression_species_energy_offsets_are_fitted_from_totals() -> None:
    images = []
    for symbols, energy in [("Fe2", -8.0), ("FeC", -6.0), ("C2", -4.0)]:
        atoms = Atoms(symbols, positions=np.zeros((len(Atoms(symbols)), 3)))
        atoms.calc = SimpleNamespace(targets={"energy": energy})
        images.append(atoms)

    pot_data = SimpleNamespace(species=[26, 6], species_energy_offsets={})
    setting = LossSetting(species_energy_offset_mode="regression")

    offsets = _resolve_species_energy_offsets(images, pot_data, setting)

    assert offsets["Fe"] == pytest.approx(-4.0)
    assert offsets["C"] == pytest.approx(-2.0)


def test_species_energy_offsets_roundtrip_through_npy(tmp_path) -> None:
    pot_data = SWData(species=["Si"])
    pot_data.species_energy_offsets = {"Si": -1.5}
    path = tmp_path / "offsets.npy"

    write_potential(str(path), pot_data)
    loaded = read_potential(str(path))

    assert isinstance(loaded, SWData)
    assert loaded.species_energy_offsets == {"Si": -1.5}
