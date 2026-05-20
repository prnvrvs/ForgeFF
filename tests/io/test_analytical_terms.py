from __future__ import annotations

from pathlib import Path

import pytest

from forgeff.io import read_potential
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData


def _species(count: int) -> list[str]:
    labels = ["Al", "Cu", "Ni"]
    return labels[:count]


def _pair_labels(species: list[str]) -> list[tuple[str, str]]:
    return [(species[i], species[j]) for i in range(len(species)) for j in range(i, len(species))]


def _all_pairs(species: list[str]) -> list[tuple[str, str]]:
    return [(left, right) for left in species for right in species]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _analytical_eam_toml(species: list[str], form: str) -> str:
    blocks = [
        "[potential]",
        'family = "eam"',
        f'form = "{form}"',
        "",
        "[species]",
        f"order = {species!r}".replace("'", '"'),
        "",
        "[grids]",
        "r = [0.1, 0.2, 0.3]",
        "rho = [0.0, 1.0, 2.0]",
        "",
    ]
    for left, right in _pair_labels(species):
        blocks.extend(
            [
                f"[pair.{left}{right}]",
                'expression = "A * exp(-r)"',
                'parameter_names = ["A"]',
                "initial = [1.0]",
                "",
            ]
        )
    if form == "alloy":
        density_terms = species
    else:
        density_terms = [f"{left}{right}" for left, right in _all_pairs(species)]
    for name in density_terms:
        blocks.extend(
            [
                f"[density.{name}]",
                'expression = "B * exp(-r)"',
                'parameter_names = ["B"]',
                "initial = [2.0]",
                "",
            ]
        )
    for item in species:
        blocks.extend(
            [
                f"[embedding.{item}]",
                'expression = "C * sqrt(rho)"',
                'variable = "rho"',
                'parameter_names = ["C"]',
                "initial = [3.0]",
                "",
            ]
        )
    return "\n".join(blocks).rstrip() + "\n"


def _analytical_adp_toml(species: list[str]) -> str:
    blocks = [
        "[potential]",
        'family = "adp"',
        'form = "alloy"',
        "",
        "[species]",
        f"order = {species!r}".replace("'", '"'),
        "",
        "[grids]",
        "r = [0.1, 0.2, 0.3]",
        "rho = [0.0, 1.0, 2.0]",
        "",
    ]
    for left, right in _pair_labels(species):
        blocks.extend(
            [
                f"[pair.{left}{right}]",
                'expression = "A * exp(-r)"',
                'parameter_names = ["A"]',
                "initial = [1.0]",
                "",
            ]
        )
    for item in species:
        blocks.extend(
            [
                f"[density.{item}]",
                'expression = "B * exp(-r)"',
                'parameter_names = ["B"]',
                "initial = [2.0]",
                "",
                f"[embedding.{item}]",
                'expression = "C * sqrt(rho)"',
                'variable = "rho"',
                'parameter_names = ["C"]',
                "initial = [3.0]",
                "",
            ]
        )
    for left, right in _pair_labels(species):
        blocks.extend(
            [
                f"[dipole.{left}{right}]",
                'expression = "D * exp(-r)"',
                'parameter_names = ["D"]',
                "initial = [4.0]",
                "",
                f"[quadrupole.{left}{right}]",
                'expression = "Q * exp(-r)"',
                'parameter_names = ["Q"]',
                "initial = [5.0]",
                "",
            ]
        )
    return "\n".join(blocks).rstrip() + "\n"


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("form", ["alloy", "fs"])
def test_analytical_eam_roundtrip(tmp_path: Path, count: int, form: str) -> None:
    species = _species(count)
    path = _write(tmp_path / f"eam_{form}_{count}.toml", _analytical_eam_toml(species, form))

    data = read_potential(str(path))
    assert isinstance(data, EAMData)
    assert data.form == form
    assert data.species_count == count
    assert data.phi_values.shape == (count, count, 3)
    assert data.emb_values.shape == (count, 3)
    if form == "alloy":
        assert data.rho_values.shape == (count, count, 3)
    else:
        assert data.rho_values.shape == (count, count, 3)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_analytical_adp_roundtrip(tmp_path: Path, count: int) -> None:
    species = _species(count)
    path = _write(tmp_path / f"adp_{count}.toml", _analytical_adp_toml(species))

    data = read_potential(str(path))
    assert isinstance(data, ADPData)
    assert data.form == "alloy"
    assert data.species_count == count
    assert data.dipole_values.shape == (count, count, 3)
    assert data.quadrupole_values.shape == (count, count, 3)
