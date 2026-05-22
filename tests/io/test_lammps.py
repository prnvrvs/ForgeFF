from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from ase import Atoms
from ase.build import bulk
from ase.calculators.eam import EAM as ASEEAM
from scipy.interpolate import CubicSpline

from forgeff.calculator import make_calculator
from forgeff.io import read_potential, write_lammps_potential
from forgeff.io.lammps import _resample_values
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.tersoff.data import TersoffData, TersoffParameters


def _assert_same_results(calc_a, calc_b, atoms) -> None:
    atoms_a = atoms.copy()
    atoms_b = atoms.copy()
    atoms_a.calc = calc_a
    atoms_b.calc = calc_b

    assert_allclose(atoms_a.get_potential_energy(), atoms_b.get_potential_energy(), rtol=1e-12, atol=1e-12)
    assert_allclose(atoms_a.get_forces(), atoms_b.get_forces(), rtol=1e-11, atol=1e-11)
    assert_allclose(atoms_a.get_stress(), atoms_b.get_stress(), rtol=1e-10, atol=1e-10)


def _fs_test_atoms(seed: int = 2024) -> Atoms:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True) * (2, 2, 2)
    numbers = np.array([13, 29] * (len(atoms) // 2), dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(numbers)
    atoms.numbers = numbers
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    return atoms


def _fs_tabulated_potential() -> EAMData:
    r_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    rho_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    phi_values = np.array(
        [
            [[0.10, 0.20, 0.30, 0.40, 0.50], [0.15, 0.25, 0.35, 0.45, 0.55]],
            [[0.15, 0.25, 0.35, 0.45, 0.55], [0.20, 0.30, 0.40, 0.50, 0.60]],
        ],
        dtype=float,
    )
    rho_values = np.array(
        [
            [[0.05, 0.10, 0.15, 0.20, 0.25], [0.07, 0.14, 0.21, 0.28, 0.35]],
            [[0.07, 0.14, 0.21, 0.28, 0.35], [0.11, 0.22, 0.33, 0.44, 0.55]],
        ],
        dtype=float,
    )
    emb_values = np.array(
        [[0.00, -0.10, -0.20, -0.30, -0.40], [0.10, 0.00, -0.10, -0.20, -0.30]],
        dtype=float,
    )
    pot = EAMData(
        form="fs",
        species_count=2,
        r_grid=r_grid,
        rho_grid=rho_grid,
        phi_values=phi_values,
        rho_values=rho_values,
        emb_values=emb_values,
        rphi_values=r_grid[None, None, :] * phi_values,
    )
    pot.species = np.array([13, 29], dtype=np.int32)
    return pot


def _fs_analytical_toml() -> str:
    return """
[potential]
family = "eam"
form = "fs"

[species]
order = ["Al", "Cu"]

[grids]
r = { start = 0.0, stop = 4.0, step = 1.0 }
rho = { start = 0.0, stop = 4.0, step = 1.0 }

[pair.AlAl]
expression = "A + B * r"
parameter_names = ["A", "B"]
initial = [0.10, 0.10]

[pair.AlCu]
expression = "A + B * r"
parameter_names = ["A", "B"]
initial = [0.15, 0.10]

[pair.CuCu]
expression = "A + B * r"
parameter_names = ["A", "B"]
initial = [0.20, 0.10]

[density.AlAl]
expression = "A + B * r"
parameter_names = ["A", "B"]
initial = [0.05, 0.05]

[density.AlCu]
expression = "A + B * r"
parameter_names = ["A", "B"]
initial = [0.07, 0.07]

[density.CuAl]
expression = "A + B * r"
parameter_names = ["A", "B"]
initial = [0.07, 0.07]

[density.CuCu]
expression = "A + B * r"
parameter_names = ["A", "B"]
initial = [0.11, 0.11]

[embedding.Al]
expression = "A + B * rho"
variable = "rho"
parameter_names = ["A", "B"]
initial = [0.00, -0.10]

[embedding.Cu]
expression = "A + B * rho"
variable = "rho"
parameter_names = ["A", "B"]
initial = [0.10, -0.10]
""".strip() + "\n"


def test_write_lammps_alloy_roundtrip(tmp_path: Path) -> None:
    source = "tests/data_path/nist/FeNiCrCoCu_with_ZBL.eam.alloy"
    pot = read_potential(source, form="alloy")
    output = tmp_path / "FeNiCrCoCu_with_ZBL.eam.alloy"
    write_lammps_potential(output, pot)

    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (3, 3, 3)
    species = np.array([26, 28, 24, 27, 29], dtype=int)
    counts = np.array([11, 11, 11, 11, 10], dtype=int)
    numbers = np.repeat(species, counts)
    rng = np.random.default_rng(2024)
    rng.shuffle(numbers)
    atoms.numbers = numbers
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)

    _assert_same_results(ASEEAM(potential=source), ASEEAM(potential=str(output)), atoms)


def test_write_lammps_fs_roundtrip(tmp_path: Path) -> None:
    source = "tests/data_path/nist/Fe_H_Kumar2023.eam.fs"
    pot = read_potential(source, form="fs")
    output = tmp_path / "Fe_H_Kumar2023.eam.fs"
    write_lammps_potential(output, pot)

    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (3, 3, 3)
    atoms.numbers = np.array([26] * 54, dtype=int)
    atoms.numbers[0] = 1
    rng = np.random.default_rng(2025)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)

    _assert_same_results(ASEEAM(potential=source, form="fs"), ASEEAM(potential=str(output), form="fs"), atoms)


def test_write_lammps_adp_roundtrip(tmp_path: Path) -> None:
    source = "tests/data_path/nist/AlCu.adp"
    pot = read_potential(source, form="adp")
    output = tmp_path / "AlCu.adp"
    write_lammps_potential(output, pot)

    atoms = bulk("Al", "fcc", a=4.05, cubic=True) * (2, 2, 2)
    atoms.numbers = np.array([13, 29] * 16, dtype=int)
    rng = np.random.default_rng(2026)
    rng.shuffle(atoms.numbers)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)

    _assert_same_results(ASEEAM(potential=source, form="adp"), ASEEAM(potential=str(output), form="adp"), atoms)


def test_write_lammps_alloy_resamples_grid(tmp_path: Path) -> None:
    pot = EAMData(
        form="alloy",
        species_count=1,
        r_grid=np.array([0.0, 1.0, 2.0]),
        rho_grid=np.array([0.0, 1.0, 2.0]),
        phi_values=np.zeros((1, 1, 3)),
        rho_values=np.array([[[0.0, 1.0, 2.0]]]),
        emb_values=np.array([[0.0, 1.0, 2.0]]),
        rphi_values=np.array([[[0.0, 1.0, 2.0]]]),
    )
    pot.species = np.array([13], dtype=np.int32)

    output = tmp_path / "Al99.eam.alloy"
    write_lammps_potential(output, pot, nr=5, nrho=4)

    lines = [line for line in output.read_text().splitlines() if line and line[0] in "-.0123456789"]
    assert lines[0] == "1 Al"
    assert lines[1].startswith("4 ")
    assert lines[1].split()[0] == "4"
    assert lines[1].split()[2] == "5"

    embedding = [float(value) for value in lines[3:7]]
    density = [float(value) for value in lines[7:12]]
    pair = [float(value) for value in lines[12:17]]
    assert_allclose(embedding, [0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0], rtol=0, atol=1e-12)
    assert_allclose(density, [0.0, 0.5, 1.0, 1.5, 2.0], rtol=0, atol=1e-12)
    assert_allclose(pair, [0.0, 0.5, 1.0, 1.5, 2.0], rtol=0, atol=1e-12)


def test_resample_values_uses_clamped_boundary_conditions() -> None:
    source_grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    values = source_grid**3
    target_grid = np.array([0.5, 1.5, 2.5], dtype=float)

    actual = _resample_values(source_grid, values, target_grid)
    deriv = np.gradient(values, source_grid, edge_order=2)
    expected = np.asarray(
        CubicSpline(source_grid, values, bc_type=((1, deriv[0]), (1, deriv[-1])))(target_grid),
        dtype=float,
    )
    default = np.asarray(CubicSpline(source_grid, values)(target_grid), dtype=float)

    assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert not np.allclose(actual, default, rtol=0, atol=1e-12)


def test_write_lammps_alloy_uses_custom_header_metadata_in_body(tmp_path: Path) -> None:
    pot = EAMData(
        form="alloy",
        species_count=1,
        r_grid=np.array([0.0, 1.0], dtype=float),
        rho_grid=np.array([0.0, 1.0], dtype=float),
        phi_values=np.zeros((1, 1, 2), dtype=float),
        rho_values=np.zeros((1, 1, 2), dtype=float),
        emb_values=np.zeros((1, 2), dtype=float),
    )
    pot.species = np.array([41], dtype=np.int32)

    output = tmp_path / "Nb.eam.alloy"
    write_lammps_potential(
        output,
        pot,
        mass=[92.906],
        a=[3.3008],
        lattice=["bcc"],
    )

    lines = output.read_text().splitlines()
    body_line = next(line for line in lines if line.startswith("41 "))
    assert body_line == "41 92.906000 3.300800 bcc"


def test_write_lammps_adp_uses_custom_header_metadata_in_body(tmp_path: Path) -> None:
    pot = ADPData(
        form="alloy",
        species_count=1,
        r_grid=np.array([0.0, 1.0], dtype=float),
        rho_grid=np.array([0.0, 1.0], dtype=float),
        phi_values=np.zeros((1, 1, 2), dtype=float),
        rho_values=np.zeros((1, 1, 2), dtype=float),
        emb_values=np.zeros((1, 2), dtype=float),
        dipole_values=np.zeros((1, 1, 2), dtype=float),
        quadrupole_values=np.zeros((1, 1, 2), dtype=float),
    )
    pot.species = np.array([41], dtype=np.int32)

    output = tmp_path / "Nb.adp"
    write_lammps_potential(
        output,
        pot,
        mass=[92.906],
        a=[3.3008],
        lattice=["bcc"],
    )

    lines = output.read_text().splitlines()
    body_line = next(line for line in lines if line.startswith("41 "))
    assert body_line == "41 92.906000 3.300800 bcc"


def test_write_lammps_fs_resampled_grid_matches_source(tmp_path: Path) -> None:
    pot = _fs_tabulated_potential()
    atoms = _fs_test_atoms()

    coarse = tmp_path / "synthetic.fs"
    fine = tmp_path / "synthetic_fine.fs"
    write_lammps_potential(coarse, pot)
    write_lammps_potential(fine, pot, nr=7, nrho=9)

    source_calc = make_calculator(pot, engine="numba")
    coarse_calc = ASEEAM(potential=str(coarse), form="fs")
    fine_calc = ASEEAM(potential=str(fine), form="fs")

    _assert_same_results(source_calc, coarse_calc, atoms)
    _assert_same_results(source_calc, fine_calc, atoms)
    _assert_same_results(coarse_calc, fine_calc, atoms)


def test_write_lammps_fs_analytical_resampled_grid_matches_source(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.toml"
    path.write_text(_fs_analytical_toml(), encoding="utf-8")
    pot = read_potential(str(path))
    atoms = _fs_test_atoms(seed=2025)

    coarse = tmp_path / "synthetic_analytical.fs"
    fine = tmp_path / "synthetic_analytical_fine.fs"
    write_lammps_potential(coarse, pot)
    write_lammps_potential(fine, pot, nr=7, nrho=9)

    source_calc = make_calculator(pot, engine="numba")
    coarse_calc = ASEEAM(potential=str(coarse), form="fs")
    fine_calc = ASEEAM(potential=str(fine), form="fs")

    _assert_same_results(source_calc, coarse_calc, atoms)
    _assert_same_results(source_calc, fine_calc, atoms)
    _assert_same_results(coarse_calc, fine_calc, atoms)


def test_write_lammps_tersoff_roundtrip(tmp_path: Path) -> None:
    species = ["Si", "C"]
    parameters = {}
    for left in species:
        for middle in species:
            for right in species:
                parameters[(left, middle, right)] = TersoffParameters.from_list(
                    [
                        3.0,
                        1.0,
                        0.0,
                        38049.0,
                        4.3484,
                        -0.57058,
                        0.72751,
                        0.00000015724,
                        2.2119,
                        346.7,
                        1.95,
                        0.15,
                        3.4879,
                        1393.6,
                    ]
                )

    pot = TersoffData.from_parameter_dict(parameters, species=species)
    output = tmp_path / "SiC.tersoff"
    write_lammps_potential(output, pot)
    read_back = read_potential(str(output))

    assert isinstance(read_back, TersoffData)
    assert read_back.species == pot.species
    assert_allclose(read_back.parameter_table, pot.parameter_table, rtol=0, atol=0)
