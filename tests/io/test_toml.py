"""Tests for TOML potential loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forgeff.io import read_potential
from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_read_custom_builtin_form_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "custom.toml",
        """
[potential]
family = "analytical"
form = "double_morse"
backend = "custom"
cutoff = 8.0
initial = [0.03, 5.0, 2.75, 0.01, 2.0, 3.5, 0.0]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, ASEData)
    assert data.calculator_name == "numpy"
    assert data.number_of_parameters_optimized == 7
    np.testing.assert_allclose(
        data.parameters,
        np.array([0.03, 5.0, 2.75, 0.01, 2.0, 3.5, 0.0]),
    )
    assert data.calculator_kwargs["expression"]
    assert data.calculator_kwargs["parameter_names"] == [
        "E1",
        "a1",
        "r1",
        "E2",
        "a2",
        "r2",
        "delta",
    ]


def test_read_custom_legacy_alias_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "custom_legacy.toml",
        """
[potential]
family = "custom"
form = "double_morse"
calculator_name = "custom"
cutoff = 8.0
initial = [0.03, 5.0, 2.75, 0.01, 2.0, 3.5, 0.0]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, ASEData)
    assert data.backend == "numpy"
    assert data.number_of_parameters_optimized == 7


def test_read_custom_numba_backend_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "custom_numba.toml",
        """
[potential]
family = "analytical"
form = "morse"
backend = "numba"
cutoff = 8.0
initial = [0.5, 2.0, 2.8]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, ASEData)
    assert data.backend == "numba"
    assert data.calculator_name == "numba"
    assert data.number_of_parameters_optimized == 3


def test_read_tabulated_eam_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "eam.toml",
        """
[potential]
family = "eam"
form = "alloy"
backend = "numpy"

[species]
order = ["Al"]

[grids]
r = [0.1, 0.2, 0.3]
rho = [0.0, 1.0, 2.0]

[pair.AlAl]
values = [0.1, 0.2, 0.3]

[density.Al]
values = [0.4, 0.5, 0.6]

[embedding.Al]
values = [0.7, 0.8, 0.9]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, EAMData)
    assert data.species_count == 1
    assert data.backend == "numpy"
    np.testing.assert_allclose(data.phi_values[0, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(data.rho_values[0, 0], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(data.emb_values[0], [0.7, 0.8, 0.9])


def test_read_tabulated_adp_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "adp.toml",
        """
[potential]
family = "adp"
form = "alloy"
backend = "numba"

[species]
order = ["Al"]

[grids]
r = [0.1, 0.2, 0.3]
rho = [0.0, 1.0, 2.0]

[pair.AlAl]
values = [0.1, 0.2, 0.3]

[density.Al]
values = [0.4, 0.5, 0.6]

[embedding.Al]
values = [0.7, 0.8, 0.9]

[dipole.AlAl]
values = [1.0, 1.1, 1.2]

[quadrupole.AlAl]
values = [1.3, 1.4, 1.5]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, ADPData)
    assert data.backend == "numba"
    np.testing.assert_allclose(data.dipole_values[0, 0], [1.0, 1.1, 1.2])
    np.testing.assert_allclose(data.quadrupole_values[0, 0], [1.3, 1.4, 1.5])


def test_read_toml_missing_required_terms_raises(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "invalid.toml",
        """
[potential]
family = "eam"
form = "alloy"

[species]
order = ["Al"]

[grids]
r = [0.1, 0.2]
rho = [0.0, 1.0]

[pair.AlAl]
values = [0.0, 0.0]
""".lstrip(),
    )

    with pytest.raises(ValueError, match="missing required density terms"):
        read_potential(str(path))
