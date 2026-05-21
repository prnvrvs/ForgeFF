from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from forgeff.io import read_potential
from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.sw.data import SWData
from forgeff.potentials.tersoff.data import TersoffData
from forgeff.template.cli import build_template
from forgeff.template.cli import run


def _species(count: int) -> list[str]:
    labels = ["Al", "Cu", "Ni"]
    return labels[:count]


def _pair_count(count: int) -> int:
    return count * (count + 1) // 2


def _lambda_count(count: int) -> int:
    return count * count * (count + 1) // 2


def _triplet_count(count: int) -> int:
    return count**3


def _roundtrip_template(tmp_path: Path, template: str, name: str) -> object:
    path = tmp_path / name
    path.write_text(template, encoding="utf-8")
    return read_potential(str(path))


def test_template_stdout_for_analytical(capsys) -> None:
    run(
        Namespace(
            family="analytical",
            species=["Al", "Cu"],
            form="morse",
            expression=None,
            parameter_names=None,
            output=None,
            force=False,
        )
    )
    out = capsys.readouterr().out
    assert '[potential]' in out
    assert 'family = "analytical"' in out
    assert 'form = "morse"' in out
    assert "[pair.AlAl]" in out
    assert "[pair.AlCu]" in out
    assert "[pair.CuCu]" in out


def test_template_stdout_for_sw(capsys) -> None:
    run(
        Namespace(
            family="sw",
            species=["Al", "Cu"],
            form=None,
            expression=None,
            parameter_names=None,
            output=None,
            force=False,
        )
    )
    out = capsys.readouterr().out
    assert 'family = "sw"' in out
    assert "[pair.AlCu]" in out
    assert "[lambda.AlAlCu]" in out
    assert "[lambda.CuCuCu]" in out


def test_template_writes_file(tmp_path: Path) -> None:
    output = tmp_path / "initial.toml"
    run(
        Namespace(
            family="eam",
            species=["Al", "Cu"],
            form="alloy",
            expression=None,
            parameter_names=None,
            output=str(output),
            force=False,
        )
    )
    text = output.read_text(encoding="utf-8")
    assert 'family = "eam"' in text
    assert 'form = "alloy"' in text
    assert "[density.Al]" in text
    assert "[density.Cu]" in text
    assert "[embedding.Al]" in text


@pytest.mark.parametrize("count", [1, 2, 3])
def test_template_matrix_analytical_builtin_roundtrip(tmp_path: Path, count: int) -> None:
    species = _species(count)
    template = build_template("analytical", species, form="morse")

    assert 'family = "analytical"' in template
    assert 'form = "morse"' in template
    assert template.count("[pair.") == _pair_count(count)

    data = _roundtrip_template(tmp_path, template, f"analytical_{count}.toml")
    assert isinstance(data, ASEData)
    assert data.engine == "numpy"
    assert len(data.calculator_kwargs["pair_terms"]) == _pair_count(count)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_template_matrix_analytical_expression_roundtrip(tmp_path: Path, count: int) -> None:
    species = _species(count)
    expression = "A * exp(-a * r) + B / r**12"
    parameter_names = ["A", "a", "B"]
    template = build_template(
        "analytical",
        species,
        expression=expression,
        parameter_names=parameter_names,
    )

    assert 'family = "analytical"' in template
    assert 'expression = "A * exp(-a * r) + B / r**12"' in template
    assert template.count("[pair.") == _pair_count(count)

    data = _roundtrip_template(tmp_path, template, f"analytical_expr_{count}.toml")
    assert isinstance(data, ASEData)
    assert data.engine == "numpy"
    assert data.calculator_kwargs["expression"] == expression
    assert len(data.calculator_kwargs["pair_terms"]) == _pair_count(count)


@pytest.mark.parametrize("form", ["alloy", "fs"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_template_matrix_eam_roundtrip(tmp_path: Path, count: int, form: str) -> None:
    species = _species(count)
    template = build_template("eam", species, form=form)

    assert 'family = "eam"' in template
    assert f'form = "{form}"' in template
    assert template.count("[pair.") == _pair_count(count)
    assert template.count("[embedding.") == count
    assert template.count("[density.") == (count if form == "alloy" else count * count)

    data = _roundtrip_template(tmp_path, template, f"eam_{form}_{count}.toml")
    assert isinstance(data, EAMData)
    assert data.form == form
    assert data.species_count == count


@pytest.mark.parametrize("count", [1, 2, 3])
def test_template_matrix_adp_roundtrip(tmp_path: Path, count: int) -> None:
    species = _species(count)
    template = build_template("adp", species)

    assert 'family = "adp"' in template
    assert template.count("[pair.") == _pair_count(count)
    assert template.count("[density.") == count
    assert template.count("[embedding.") == count
    assert template.count("[dipole.") == _pair_count(count)
    assert template.count("[quadrupole.") == _pair_count(count)

    data = _roundtrip_template(tmp_path, template, f"adp_{count}.toml")
    assert isinstance(data, ADPData)
    assert data.species_count == count


@pytest.mark.parametrize("count", [1, 2, 3])
def test_template_matrix_sw_roundtrip(tmp_path: Path, count: int) -> None:
    species = _species(count)
    template = build_template("sw", species)

    assert 'family = "sw"' in template
    assert template.count("[pair.") == _pair_count(count)
    assert template.count("[lambda.") == _lambda_count(count)

    data = _roundtrip_template(tmp_path, template, f"sw_{count}.toml")
    assert isinstance(data, SWData)
    assert data.species == species


@pytest.mark.parametrize("count", [1, 2, 3])
def test_template_matrix_tersoff_roundtrip(tmp_path: Path, count: int) -> None:
    species = _species(count)
    template = build_template("tersoff", species)

    assert 'family = "tersoff"' in template
    assert template.count("[triplet.") == _triplet_count(count)

    data = _roundtrip_template(tmp_path, template, f"tersoff_{count}.toml")
    assert isinstance(data, TersoffData)
    assert data.species == species
    assert data.number_of_parameters_optimized == _triplet_count(count) * 14
