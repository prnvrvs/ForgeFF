from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms

from forgeff.loss import ErrorPrinter, LossFunctionStress


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
