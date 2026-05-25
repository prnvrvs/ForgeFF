"""IO."""

from typing import Any
import gzip
import os

import ase.io
import numpy as np
from ase import Atoms
from ase.io.formats import parse_filename

from forgeff.io.toml import read_potential_toml, write_potential_toml
from forgeff.io.lammps import read_lammps_tersoff_potential, write_lammps_potential
from forgeff.io.potfit import read_force, write_force
from forgeff.io.nist import read_nist_potential
from forgeff.io.mlip.cfg import read_cfg, write_cfg
from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.sw.data import SWData


def read(filename: str, species: list[int] | None = None) -> list[Atoms]:
    """Read images.

    Parameters
    ----------
    filename : str
        File name to be read.
        Both the MLIP `.cfg` format, potfit force configurations, and the
        ASE-recognized formats can be parsed.

        To select a part of images, the ASE `@` syntax can be used as follows.

        https://wiki.fysik.dtu.dk/ase/ase/gui/basics.html#selecting-part-of-a-trajectory

        - `x.traj@0:10:1`: first 10 images
        - `x.traj@0:10`: first 10 images
        - `x.traj@:10`: first 10 images
        - `x.traj@-10:`: last 10 images
        - `x.traj@0`: first image
        - `x.traj@-1`: last image
        - `x.traj@::2`: every second image

        Further, for the ASE database format, i.e., `.json` and `.db`,
        the extended ASE syntax can also be used as follows.

        https://wiki.fysik.dtu.dk/ase/ase/db/db.html#integration-with-other-parts-of-ase

        https://wiki.fysik.dtu.dk/ase/ase/db/db.html#querying

        - `x.db@H>0`: images with hydrogen atoms

    species : list[int]
        List of atomic numbers for the atomic types in the MLIP `.cfg` and
        potfit force formats.

    Returns
    -------
    list[Atoms]
        List of ASE `Atoms` objects.

    """
    filename = os.fspath(filename)
    filename_parsed, index = parse_filename(filename)
    index = ":" if index is None else index
    if isinstance(filename_parsed, str) and filename_parsed.endswith(".cfg"):
        images = read_cfg(filename_parsed, index=index, species=species)
    elif isinstance(filename_parsed, str) and _looks_like_potfit_force(filename_parsed):
        images = read_force(filename_parsed, index=index, species=species)
    else:
        images = ase.io.read(filename_parsed, index=index, parallel=False)
    return [images] if isinstance(images, Atoms) else images


def write(filename: str, images: list[Atoms], species: list[int] | None = None) -> None:
    """Write images.

    Parameters
    ----------
    filename : str
        File name to be written.
        Both the MLIP `.cfg` format, potfit force configurations, and the
        ASE-recognized formats can be written.
    images : list[Atoms]
        List of ASE `Atoms` objects.
    species : list[int]
        List of atomic numbers for the atomic types in the MLIP `.cfg` and
        potfit force formats.

    """
    filename = os.fspath(filename)
    if filename.endswith(".cfg"):
        return write_cfg(filename, images, species=species)
    if filename.endswith(".force") or filename.endswith(".potfit"):
        return write_force(filename, images, species=species)
    return ase.io.write(filename, images)


def read_potential(filename: str, form: str | None = None):
    """Read a potential file into a potential data object."""
    filename = os.fspath(filename)
    if filename.endswith(".toml"):
        return read_potential_toml(filename)

    if (
        filename.endswith(".eam")
        or filename.endswith(".eam.alloy")
        or filename.endswith(".adp")
        or filename.endswith(".fs")
        or filename.endswith(".txt")
    ):
        return read_nist_potential(filename, form=form)

    if filename.endswith(".tersoff"):
        return read_lammps_tersoff_potential(filename)

    if filename.endswith(".npy"):
        data = np.load(filename, allow_pickle=True).item()
        if "dipole_values" in data or "quadrupole_values" in data:
            data = dict(data)
            backend = data.pop("backend", None)
            if "engine" not in data and backend is not None:
                data["engine"] = backend
            return ADPData(**data)
        if {"phi_values", "rho_values", "emb_values"} <= data.keys():
            data = dict(data)
            backend = data.pop("backend", None)
            if "engine" not in data and backend is not None:
                data["engine"] = backend
            return EAMData(**data)
        if "pair_parameters" in data and "lambda_values" in data:
            data = dict(data)
            if "species" in data and data["species"] is not None:
                data["species"] = list(data["species"])
            return SWData(**data)
        if {"epsilon", "sigma", "costheta0", "A", "B", "p", "a", "lambda1", "gamma"} <= data.keys():
            data = dict(data)
            backend = data.pop("backend", None)
            if "species" in data and data["species"] is not None:
                data["species"] = list(data["species"])
            if "engine" not in data and backend is not None:
                data["engine"] = backend
            data.pop("engine", None)
            return SWData(**data)
        if {"params", "info", "kwargs"} <= data.keys():
            kwargs = dict(data.get("kwargs", {}))
            engine = kwargs.pop("engine", "numpy")
            ase_data = ASEData(
                calculator_kwargs=kwargs,
                engine=engine,
            )
            ase_data.parameters = np.asarray(data.get("params", []), dtype=float)
            ase_data.parameter_info = data.get("info", {})
            ase_data.optimized = list(ase_data.parameter_info)
            ase_data.species_energy_offsets = dict(data.get("species_energy_offsets", {}))
            return ase_data
        return data

    raise ValueError(f"Unsupported potential format: {filename}")


def write_potential(filename: str, data: Any) -> None:
    """Write a potential data object."""
    filename = os.fspath(filename)
    if filename.endswith(".toml") and isinstance(data, (ASEData, EAMData, ADPData)):
        write_potential_toml(filename, data)
        return
    if filename.endswith(".npy") and hasattr(data, "write"):
        data.write(filename)
        return
    raise ValueError(f"Unsupported potential object: {type(data).__name__}")


def _looks_like_potfit_force(filename: str) -> bool:
    opener = gzip.open if filename.endswith(".gz") else open
    try:
        with opener(filename, "rt") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                return stripped.startswith("#N")
    except OSError:
        return False
    return False
