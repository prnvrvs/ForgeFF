from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io.vasp import iread_vasp_out

import forgeff.io
from forgeff.io.potfit import read_force, write_force


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _potfit_ase_dir() -> Path:
    return _repo_root() / "tests" / "data_path" / "potfit" / "ase"


def _read_outcar(path: Path) -> list:
    with path.open("r") as handle:
        return list(iread_vasp_out(handle, index=":"))


def _assert_images_match(reference, candidate) -> None:
    assert candidate.numbers.tolist() == reference.numbers.tolist()
    assert candidate.pbc.tolist() == reference.pbc.tolist()
    np.testing.assert_allclose(candidate.cell.array, reference.cell.array, atol=1e-10)
    np.testing.assert_allclose(candidate.get_positions(), reference.get_positions(), atol=5e-7)
    np.testing.assert_allclose(candidate.get_forces(), reference.get_forces(), atol=5e-7)
    np.testing.assert_allclose(candidate.get_stress(), reference.get_stress(), atol=1e-8)
    np.testing.assert_allclose(candidate.get_potential_energy(), reference.get_potential_energy(), atol=1e-6)


def test_read_outcar_matches_ase() -> None:
    path = _potfit_ase_dir() / "OUTCAR"
    ase_images = _read_outcar(path)
    forgeff_images = forgeff.io.read(path)

    assert len(forgeff_images) == len(ase_images)
    for reference, candidate in zip(ase_images, forgeff_images, strict=True):
        _assert_images_match(reference, candidate)


def test_write_force_roundtrip_against_ase_outcar(tmp_path: Path) -> None:
    path = _potfit_ase_dir() / "OUTCAR"
    ase_images = _read_outcar(path)

    force_path = tmp_path / "outcar.force"
    write_force(force_path, ase_images)

    roundtrip = read_force(force_path, index=":")
    assert len(roundtrip) == len(ase_images)
    for reference, candidate in zip(ase_images, roundtrip, strict=True):
        _assert_images_match(reference, candidate)
        assert candidate.info["potfit_useforce"] == 1
        assert candidate.info["potfit_weight"] == 1.0


def test_public_io_helpers_handle_potfit_force(tmp_path: Path) -> None:
    path = _potfit_ase_dir() / "OUTCAR.005"
    ase_images = _read_outcar(path)

    force_path = tmp_path / "outcar.005.potfit"
    forgeff.io.write(force_path, ase_images)

    roundtrip = forgeff.io.read(force_path)
    assert len(roundtrip) == len(ase_images)
    for reference, candidate in zip(ase_images, roundtrip, strict=True):
        _assert_images_match(reference, candidate)
