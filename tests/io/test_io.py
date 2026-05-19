"""Tests for IO."""

import pathlib

import numpy as np

import forgeff.io
from forgeff.io.mlip.cfg import read_cfg, write_cfg
from forgeff.train.trainer import read_images


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _current_cfg_path(_: pathlib.Path) -> pathlib.Path:
    return _repo_root() / "tests/test.cfg"


def _example_cfg_path(_: pathlib.Path) -> pathlib.Path:
    return _repo_root() / "examples/toml/data/unary/training.cfg"


def _example_binary_cfg_path(_: pathlib.Path) -> pathlib.Path:
    return _repo_root() / "examples/toml/data/binary/training.cfg"


def test_read_path(data_path: pathlib.Path) -> None:
    """Test if the `pathlib.Path` object can be read directly."""
    path = _current_cfg_path(data_path)
    read_cfg(path)


def test_index(data_path: pathlib.Path) -> None:
    """Test if `read_cfg` can accept a flexible `index`."""
    path = _current_cfg_path(data_path)
    images = read_cfg(path, index="0:2")
    assert len(images) == 2


def test_parse_filename(data_path: pathlib.Path) -> None:
    """Test if the ASE at-mark syntax works."""
    path = _current_cfg_path(data_path)
    images = forgeff.io.read(str(path) + "@0")
    assert len(images) == 1
    images = forgeff.io.read(str(path) + "@0:2")
    assert len(images) == 2


def test_read_multiple_files(data_path: pathlib.Path) -> None:
    """Test if multiple files can be read."""
    configurations = [
        str(_current_cfg_path(data_path)),
        str(_example_cfg_path(data_path)),
        str(_example_binary_cfg_path(data_path)),
    ]
    n_ref = sum(len(read_cfg(_, index=":")) for _ in configurations)
    assert len(read_images(configurations)) == n_ref


def test_example_datasets_have_ten_configurations(data_path: pathlib.Path) -> None:
    unary = forgeff.io.read(str(_example_cfg_path(data_path)), species=[13])
    binary = forgeff.io.read(str(_example_binary_cfg_path(data_path)), species=[13, 29])

    assert len(unary) == 10
    assert len(binary) == 10
    assert all(set(atoms.numbers) == {13} for atoms in unary)
    assert all(set(atoms.numbers) == {13, 29} for atoms in binary)


def test_read_ase_file(data_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test if the ASE-recognized file can be read."""
    path = _current_cfg_path(data_path)
    atoms = read_cfg(path)
    fd = tmp_path / "test.xyz"
    atoms.write(fd)
    assert forgeff.io.read(fd)


def test_roundtrip(data_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test if `write_cfg` works as expected."""
    path = _current_cfg_path(data_path)
    atoms_ref = read_cfg(path)
    atoms_ref.calc.results.pop("free_energy")
    fd = tmp_path / "test.cfg"
    write_cfg(fd, atoms_ref)
    atoms = read_cfg(fd)
    assert atoms == atoms_ref
    assert atoms.info == atoms_ref.info
    for k, v in atoms_ref.calc.results.items():
        assert np.all(atoms.calc.results[k] == v)
