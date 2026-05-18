from __future__ import annotations

from forgeff.io import read_potential
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData


def test_read_nist_al99_eam() -> None:
    data = read_potential("tests/data_path/nist/Al99.eam.alloy")
    assert isinstance(data, EAMData)
    assert data.species.tolist() == [13]
    assert data.r_grid is not None and data.r_grid.size > 0
    assert data.rho_grid is not None and data.rho_grid.size > 0


def test_read_nist_alcu_adp() -> None:
    data = read_potential("tests/data_path/nist/AlCu.adp")
    assert isinstance(data, ADPData)
    assert data.species.tolist() == [13, 29]
    assert data.r_grid is not None and data.r_grid.size > 0
    assert data.rho_grid is not None and data.rho_grid.size > 0
