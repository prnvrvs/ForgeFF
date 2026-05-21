"""Tests for TOML potential loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forgeff.io import read_potential
from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.tersoff.data import TersoffData
from forgeff.potentials.sw.data import SWData


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
cutoff = 8.0
initial = [0.03, 5.0, 2.75, 0.01, 2.0, 3.5, 0.0]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, ASEData)
    assert data.engine == "numpy"
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


@pytest.mark.parametrize(
    ("form", "engine", "expected_calculator", "initial"),
    [
        ("lj", "ASE", "LennardJones", "[0.03, 5.0]"),
        ("morse", "ASE", "MorsePotential", "[0.03, 5.0, 2.75]"),
    ],
)
def test_read_custom_ase_builtin_forms_toml(
    tmp_path: Path,
    form: str,
    engine: str,
    expected_calculator: str,
    initial: str,
) -> None:
    path = _write_text(
        tmp_path / f"{form}.toml",
        f"""
[potential]
family = "analytical"
form = "{form}"
engine = "{engine}"
cutoff = 8.0
initial = {initial}
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, ASEData)
    assert data.engine == "ASE"
    assert data.calculator_kwargs["calculator"] == expected_calculator


def test_read_custom_ase_unsupported_form_falls_back_to_numpy(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "double_morse.toml",
        """
[potential]
family = "analytical"
form = "double_morse"
engine = "ASE"
cutoff = 8.0
initial = [0.03, 5.0, 2.75, 0.01, 2.0, 3.5, 0.0]
""".lstrip(),
    )

    with pytest.warns(RuntimeWarning, match="ASE does not support analytical form"):
        data = read_potential(str(path))

    assert isinstance(data, ASEData)
    assert data.engine == "numpy"
    assert data.calculator_kwargs["expression"]


@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_read_multispecies_pairwise_lj_toml(tmp_path: Path, engine: str) -> None:
    path = _write_text(
        tmp_path / f"multispecies_{engine}.toml",
        f"""
[potential]
family = "analytical"
form = "lj"
engine = "{engine}"
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

    data = read_potential(str(path))
    assert isinstance(data, ASEData)
    assert data.engine == engine
    assert data.number_of_parameters_optimized == 6
    assert len(data.calculator_kwargs["pair_terms"]) == 3
    assert data.calculator_kwargs["pair_terms"][0]["parameter_names"] == ["epsilon", "sigma"]


@pytest.mark.parametrize(
    ("path", "expected_parameters"),
    [
        ("examples/toml/pairwise/morse/binary/initial.toml", 9),
        ("examples/toml/pairwise/double_morse/binary/initial.toml", 21),
        ("examples/toml/pairwise/custom_expression/binary/initial.toml", 12),
    ],
)
def test_read_binary_pairwise_examples_have_per_pair_parameters(path: str, expected_parameters: int) -> None:
    data = read_potential(str(Path(__file__).resolve().parents[2] / path))
    assert isinstance(data, ASEData)
    assert len(data.calculator_kwargs["pair_terms"]) == 3
    assert data.number_of_parameters_optimized == expected_parameters


@pytest.mark.parametrize(
    ("path", "expected_parameters"),
    [
        ("examples/toml/pairwise/morse/unary/initial.toml", 3),
        ("examples/toml/pairwise/double_morse/unary/initial.toml", 7),
        ("examples/toml/pairwise/custom_expression/unary/initial.toml", 4),
    ],
)
def test_read_unary_pairwise_examples_use_pair_blocks(path: str, expected_parameters: int) -> None:
    data = read_potential(str(Path(__file__).resolve().parents[2] / path))
    assert isinstance(data, ASEData)
    assert len(data.calculator_kwargs["pair_terms"]) == 1
    assert data.number_of_parameters_optimized == expected_parameters


def test_read_multispecies_pairwise_ase_is_rejected(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "multispecies_ase.toml",
        """
[potential]
family = "analytical"
form = "lj"
engine = "ASE"
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

    with pytest.warns(RuntimeWarning, match="multispecies analytical pair fitting"):
        with pytest.raises(ValueError, match="multispecies analytical pair fitting"):
            read_potential(str(path))


def test_read_tersoff_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "tersoff.toml",
        """
[potential]
family = "tersoff"
cutoff_skin = 0.4

[species]
order = ["Si"]

[triplet.SiSiSi]
initial = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, TersoffData)
    assert data.species == ["Si"]
    assert data.cutoff_skin == 0.4
    assert data.number_of_parameters_optimized == 14
    np.testing.assert_allclose(
        data.parameters,
        np.arange(1.0, 15.0, dtype=float),
    )


def test_read_tersoff_toml_preserves_species_order(tmp_path: Path) -> None:
    triplets = [
        "SiSiSi",
        "SiSiC",
        "SiCSi",
        "SiCC",
        "CSiSi",
        "CSiC",
        "CCSi",
        "CCC",
    ]
    blocks = "\n".join(
        f"[triplet.{triplet}]\ninitial = [{idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0, {idx}.0]"
        for idx, triplet in enumerate(triplets, start=1)
    )
    path = _write_text(
        tmp_path / "tersoff_order.toml",
        f"""
[potential]
family = "tersoff"

[species]
order = ["C", "Si"]

{blocks}
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, TersoffData)
    assert data.species == ["C", "Si"]
    np.testing.assert_allclose(data.parameter_table[1, 1, 1], np.full(14, 1.0))
    np.testing.assert_allclose(data.parameter_table[0, 0, 0], np.full(14, 8.0))


def test_read_tabulated_eam_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "eam.toml",
        """
[potential]
family = "eam"
form = "alloy"

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
    assert data.engine == "numpy"
    np.testing.assert_allclose(data.phi_values[0, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(data.rho_values[0, 0], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(data.emb_values[0], [0.7, 0.8, 0.9])


def test_read_sw_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "sw.toml",
        """
[potential]
family = "sw"
species = ["Si"]
epsilon = 2.1683
sigma = 2.0951
costheta0 = 0.3333333333333333
A = 7.049556277
B = 0.6022245584
p = 4.0
a = 1.8
lambda1 = 21.0
gamma = 1.2
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, SWData)
    assert data.species == ["Si"]
    np.testing.assert_allclose(
        data.pair_parameters[0, 0],
        np.array(
            [
                2.1683 * 7.049556277 * 0.6022245584 * 2.0951**4,
                2.1683 * 7.049556277,
                4.0,
                0.0,
                2.0951,
                1.8 * 2.0951,
                1.2 * 2.0951,
                1.8 * 2.0951,
            ],
            dtype=float,
        ),
    )


def test_read_multispecies_sw_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "sw_binary.toml",
        """
[potential]
family = "sw"
costheta0 = 0.3333333333333333

[species]
order = ["Al", "Cu"]

[pair.AlAl]
initial = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 7.0]

[pair.AlCu]
initial = [1.1, 2.1, 3.1, 0.0, 4.1, 5.1, 6.1, 7.1]

[pair.CuCu]
initial = [1.2, 2.2, 3.2, 0.0, 4.2, 5.2, 6.2, 7.2]

[lambda.AlAlAl]
initial = [0.1]

[lambda.AlAlCu]
initial = [0.2]

[lambda.AlCuCu]
initial = [0.3]

[lambda.CuAlAl]
initial = [0.4]

[lambda.CuAlCu]
initial = [0.5]

[lambda.CuCuCu]
initial = [0.6]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, SWData)
    assert data.species == ["Al", "Cu"]
    assert data.number_of_parameters_optimized == 30
    np.testing.assert_allclose(data.pair_parameters[0, 1], [1.1, 2.1, 3.1, 0.0, 4.1, 5.1, 6.1, 7.1])
    np.testing.assert_allclose(data.pair_parameters[1, 0], [1.1, 2.1, 3.1, 0.0, 4.1, 5.1, 6.1, 7.1])
    np.testing.assert_allclose(data.lambda_values[0, 0, 1], 0.2)
    np.testing.assert_allclose(data.lambda_values[0, 1, 0], 0.2)
    np.testing.assert_allclose(data.lambda_values[1, 0, 1], 0.5)
    np.testing.assert_allclose(data.lambda_values[0, 0, 0], 0.1)
    np.testing.assert_allclose(
        data.parameters,
        np.array(
            [
                1.0,
                2.0,
                3.0,
                0.0,
                4.0,
                5.0,
                6.0,
                7.0,
                1.1,
                2.1,
                3.1,
                0.0,
                4.1,
                5.1,
                6.1,
                7.1,
                1.2,
                2.2,
                3.2,
                0.0,
                4.2,
                5.2,
                6.2,
                7.2,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
            ],
            dtype=float,
        ),
    )


def test_read_tabulated_eam_alloy_uses_diagonal_density_parameters(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "eam_alloy.toml",
        """
[potential]
family = "eam"
form = "alloy"

[species]
order = ["Al", "Cu"]

[grids]
r = [0.1, 0.2, 0.3]
rho = [0.0, 1.0, 2.0]

[pair.AlAl]
values = [0.1, 0.2, 0.3]

[pair.AlCu]
values = [0.4, 0.5, 0.6]

[pair.CuCu]
values = [0.7, 0.8, 0.9]

[density.Al]
values = [1.0, 1.1, 1.2]

[density.Cu]
values = [1.3, 1.4, 1.5]

[embedding.Al]
values = [2.0, 2.1, 2.2]

[embedding.Cu]
values = [2.3, 2.4, 2.5]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, EAMData)
    assert data.number_of_parameters_optimized == 24
    assert data.parameters.size == 24
    np.testing.assert_allclose(data.rho_values[:, 0, :], [[1.0, 1.1, 1.2], [1.0, 1.1, 1.2]])
    np.testing.assert_allclose(data.rho_values[:, 1, :], [[1.3, 1.4, 1.5], [1.3, 1.4, 1.5]])


def test_read_tabulated_fs_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "fs.toml",
        """
[potential]
family = "eam"
form = "fs"

[species]
order = ["Al", "Cu"]

[grids]
r = [0.1, 0.2, 0.3]
rho = [0.0, 1.0, 2.0]

[pair.AlAl]
values = [0.1, 0.2, 0.3]

[pair.AlCu]
values = [0.4, 0.5, 0.6]

[pair.CuCu]
values = [0.7, 0.8, 0.9]

[density.AlAl]
values = [1.0, 1.1, 1.2]

[density.AlCu]
values = [1.3, 1.4, 1.5]

[density.CuAl]
values = [1.3, 1.4, 1.5]

[density.CuCu]
values = [1.6, 1.7, 1.8]

[embedding.Al]
values = [2.0, 2.1, 2.2]

[embedding.Cu]
values = [2.3, 2.4, 2.5]
""".lstrip(),
    )

    data = read_potential(str(path))
    assert isinstance(data, EAMData)
    assert data.form == "fs"
    assert data.engine == "numba"
    np.testing.assert_allclose(data.rho_values[0, 1], [1.3, 1.4, 1.5])


def test_eam_fs_initialize_preserves_asymmetry() -> None:
    data = EAMData(
        form="fs",
        species_count=2,
        r_grid=np.array([0.1, 0.2, 0.3]),
        rho_grid=np.array([0.0, 1.0, 2.0]),
        phi_values=np.zeros((2, 2, 3)),
        rho_values=None,
        emb_values=np.zeros((2, 3)),
    )
    data.species = np.array([13, 29], dtype=np.int32)

    data.initialize(np.random.default_rng(42))

    assert not np.allclose(data.rho_values[0, 1], data.rho_values[1, 0])


def test_eam_fs_parameter_setter_preserves_asymmetry() -> None:
    data = EAMData(
        form="fs",
        species_count=2,
        r_grid=np.array([0.1, 0.2, 0.3]),
        rho_grid=np.array([0.0, 1.0, 2.0]),
        phi_values=np.zeros((2, 2, 3)),
        rho_values=np.zeros((2, 2, 3)),
        emb_values=np.zeros((2, 3)),
    )
    data.species = np.array([13, 29], dtype=np.int32)

    params = np.arange(data.number_of_parameters_optimized, dtype=float)
    data.parameters = params

    rho_offset = 2 * 2 * 3
    expected_rho = params[rho_offset : rho_offset + 12].reshape(2, 2, 3)
    np.testing.assert_allclose(data.rho_values, expected_rho)
    assert not np.allclose(data.rho_values[0, 1], data.rho_values[1, 0])


def test_read_tabulated_adp_toml(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "adp.toml",
        """
[potential]
family = "adp"
form = "alloy"

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
    assert data.engine == "numba"
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
