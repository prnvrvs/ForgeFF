from __future__ import annotations

import numpy as np

from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.eam.data import EAMData


def test_eam_data_provides_default_bounds() -> None:
    pot = EAMData(
        species_count=1,
        r_grid=np.linspace(0.1, 1.0, 3),
        rho_grid=np.linspace(0.1, 1.0, 3),
        phi_values=np.zeros((1, 1, 3)),
        rho_values=np.zeros((1, 1, 3)),
        emb_values=np.zeros((1, 3)),
    )
    pot.species = [13]

    bounds = pot.get_bounds()

    assert bounds is not None
    assert len(bounds) == pot.number_of_parameters_optimized
    assert all(bound == (-10.0, 10.0) for bound in bounds)


def test_ase_data_provides_default_bounds() -> None:
    pot = ASEData()
    pot.add_parameter("epsilon", (), 1.0)
    pot.add_parameter("sigma", (), 2.0)

    bounds = pot.get_bounds()

    assert bounds is not None
    assert len(bounds) == pot.number_of_parameters_optimized
    assert all(bound == (-10.0, 10.0) for bound in bounds)
