from __future__ import annotations

import builtins
from argparse import Namespace
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write
import pytest

from forgeff.io import read_potential
from forgeff.io.mlip.cfg import write_cfg
from forgeff.io.potfit import write_force
from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.sw.data import SWData
from forgeff.potentials.tersoff.data import TersoffData
from forgeff.template.cli import build_template
from forgeff.template.cli import _wizard_generate
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


def test_wizard_generates_initial_and_training_templates(tmp_path: Path) -> None:
    dataset = tmp_path / "training.xyz"
    atoms = Atoms("CHO", positions=np.zeros((3, 3)))
    atoms.info["energy"] = -1.23
    ase_write(dataset, [atoms])

    initial_output = tmp_path / "initial.toml"
    train_output = tmp_path / "forgeff.train.toml"
    args = Namespace(
        family="analytical",
        species=None,
        form="morse",
        expression=None,
        parameter_names=None,
        output=str(train_output),
        wizard=True,
        dataset=str(dataset),
        initial_output=str(initial_output),
        train_output=None,
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.001,
        energy_offset_mode="off",
        train_fraction=1.0,
        test_fraction=0.0,
        optimizer="L-BFGS-B",
        maxiter=100,
        tol=1.0e-6,
        force=False,
    )

    _wizard_generate(args)

    initial_text = initial_output.read_text(encoding="utf-8")
    train_text = train_output.read_text(encoding="utf-8")

    assert '[potential]' in initial_text
    assert 'family = "analytical"' in initial_text
    assert 'order = ["C", "H", "O"]' in initial_text
    assert 'species = ["C", "H", "O"]' in train_text
    assert 'energy_weight = 1' in train_text
    assert 'forces_weight = 0.01' in train_text
    assert 'stress_weight = 0.001' in train_text
    assert 'method = "L-BFGS-B"' in train_text


def test_wizard_prompts_for_cfg_species_labels(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "training.cfg"
    atoms = Atoms("AlCu", positions=np.zeros((2, 3)), cell=np.eye(3), pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.23,
        forces=np.zeros((2, 3)),
        stress=np.zeros(6),
    )
    write_cfg(dataset, [atoms], species=["Al", "Cu"])

    responses = iter(["Al Cu"])
    monkeypatch.setattr(builtins, "input", lambda _: next(responses))

    initial_output = tmp_path / "initial.toml"
    train_output = tmp_path / "forgeff.train.toml"
    args = Namespace(
        family="analytical",
        species=None,
        form="morse",
        expression=None,
        parameter_names=None,
        output=str(train_output),
        wizard=True,
        dataset=str(dataset),
        initial_output=str(initial_output),
        train_output=None,
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.001,
        energy_offset_mode="off",
        train_fraction=1.0,
        test_fraction=0.0,
        optimizer="L-BFGS-B",
        maxiter=100,
        tol=1.0e-6,
        force=False,
    )

    _wizard_generate(args)

    initial_text = initial_output.read_text(encoding="utf-8")
    train_text = train_output.read_text(encoding="utf-8")

    assert 'order = ["Al", "Cu"]' in initial_text
    assert 'species = ["Al", "Cu"]' in train_text


def test_wizard_infers_species_from_potfit_force(tmp_path: Path) -> None:
    dataset = tmp_path / "training.force"
    atoms = Atoms("AlCu", positions=np.zeros((2, 3)), cell=np.eye(3), pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-2.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros(6),
    )
    write_force(dataset, [atoms])

    initial_output = tmp_path / "initial.toml"
    train_output = tmp_path / "forgeff.train.toml"
    args = Namespace(
        family="analytical",
        species=None,
        form="morse",
        expression=None,
        parameter_names=None,
        output=str(train_output),
        wizard=True,
        dataset=str(dataset),
        initial_output=str(initial_output),
        train_output=None,
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.001,
        energy_offset_mode="off",
        train_fraction=1.0,
        test_fraction=0.0,
        optimizer="L-BFGS-B",
        maxiter=100,
        tol=1.0e-6,
        force=False,
    )

    _wizard_generate(args)

    initial_text = initial_output.read_text(encoding="utf-8")
    train_text = train_output.read_text(encoding="utf-8")

    assert 'order = ["Al", "Cu"]' in initial_text
    assert 'species = ["Al", "Cu"]' in train_text


def test_wizard_keeps_existing_outputs_and_continues(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "training.xyz"
    atoms = Atoms("Al", positions=np.zeros((1, 3)))
    ase_write(dataset, [atoms])

    initial_output = tmp_path / "initial.toml"
    train_output = tmp_path / "forgeff.train.toml"
    initial_output.write_text("keep", encoding="utf-8")
    train_output.write_text("keep", encoding="utf-8")

    responses = iter(["", ""])
    monkeypatch.setattr(builtins, "input", lambda _: next(responses))

    args = Namespace(
        family="analytical",
        species=None,
        form="morse",
        expression=None,
        parameter_names=None,
        output=None,
        wizard=True,
        dataset=str(dataset),
        initial_output=str(initial_output),
        train_output=str(train_output),
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.001,
        energy_offset_mode="off",
        train_fraction=1.0,
        test_fraction=0.0,
        optimizer="L-BFGS-B",
        maxiter=100,
        tol=1.0e-6,
        force=False,
    )

    _wizard_generate(args)

    assert initial_output.read_text(encoding="utf-8") == "keep"
    assert train_output.read_text(encoding="utf-8") == "keep"


def test_wizard_allows_skipping_dataset_prompt(tmp_path: Path, monkeypatch) -> None:
    initial_output = tmp_path / "initial.toml"
    train_output = tmp_path / "forgeff.train.toml"

    responses = iter(["", ""])
    monkeypatch.setattr(builtins, "input", lambda _: next(responses))

    args = Namespace(
        family="analytical",
        species=["Al"],
        form="morse",
        expression=None,
        parameter_names=None,
        output=str(train_output),
        wizard=True,
        dataset=None,
        initial_output=str(initial_output),
        train_output=str(train_output),
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.001,
        energy_offset_mode="off",
        train_fraction=1.0,
        test_fraction=0.0,
        optimizer="L-BFGS-B",
        maxiter=100,
        tol=1.0e-6,
        force=False,
    )

    _wizard_generate(args)

    assert initial_output.exists()
    assert train_output.exists()
