"""Potfit-style force configuration I/O."""

from __future__ import annotations

from typing import TextIO

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers, chemical_symbols
from ase.utils import reader, string2index, writer


def _convert_species(species: list[int] | list[str] | None) -> list[int] | None:
    if species is None:
        return None
    if isinstance(species, list) and species and isinstance(species[0], str):
        return [int(atomic_numbers[str(symbol)]) for symbol in species]
    return [int(item) for item in species]


def _species_numbers_from_labels(labels: list[str], types: list[int]) -> list[int]:
    numbers = []
    for type_index in types:
        label = labels[type_index]
        if label in atomic_numbers:
            numbers.append(int(atomic_numbers[label]))
        else:
            raise ValueError(f"Unknown chemical symbol in potfit configuration: {label!r}")
    return numbers


def _species_labels(species: list[int]) -> list[str]:
    return [chemical_symbols[int(num)] for num in species]


def _stress_from_potfit(values: list[float]) -> np.ndarray:
    # potfit format order: xx yy zz xy yz xz
    # ASE Voigt order:     xx yy zz yz xz xy
    return np.asarray([values[0], values[1], values[2], values[4], values[5], values[3]], dtype=float)


def _stress_to_potfit(values: np.ndarray) -> np.ndarray:
    # ASE Voigt order:     xx yy zz yz xz xy
    # potfit format order: xx yy zz xy yz xz
    return np.asarray([values[0], values[1], values[2], values[5], values[3], values[4]], dtype=float)


def _get_singlepoint(atoms: Atoms, key: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        value = atoms.calc.results[key]
    except Exception:
        if shape is None:
            return np.array([], dtype=float)
        return np.zeros(shape, dtype=float)
    return np.asarray(value, dtype=float)


def _read_potfit_config(lines: list[str], start: int, species: list[int] | None) -> tuple[Atoms, int]:
    header = lines[start].split()
    if len(header) < 3 or header[0] != "#N":
        raise ValueError("Potfit configuration must start with a #N line")

    natoms = int(header[1])
    useforce = int(header[2])

    i = start + 1
    labels: list[str] = []
    cell_rows: dict[str, list[float]] = {}
    weight = 1.0
    energy_per_atom = None
    stress = None
    box_of_contributing_particles: list[str] = []

    while i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw:
            continue
        if raw.startswith("#F"):
            break
        if not raw.startswith("#"):
            continue

        tokens = raw.split()
        tag = tokens[0]
        if tag == "#C":
            labels = tokens[1:]
        elif tag in {"#X", "#Y", "#Z"}:
            cell_rows[tag[1:]] = [float(x) for x in tokens[1:4]]
        elif tag == "#W":
            weight = float(tokens[1])
        elif tag == "#E":
            energy_per_atom = float(tokens[1])
        elif tag == "#S":
            stress = [float(x) for x in tokens[1:7]]
        elif tag.startswith("#B"):
            box_of_contributing_particles.append(raw)

    if i > len(lines):
        raise ValueError("Missing #F line in potfit configuration")

    atom_rows: list[list[str]] = []
    while len(atom_rows) < natoms and i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw:
            continue
        if raw.startswith("#"):
            raise ValueError("Unexpected header line inside potfit atom data")
        atom_rows.append(raw.split())

    if len(atom_rows) != natoms:
        raise ValueError("Incomplete atom data in potfit configuration")

    type_ids = [int(row[0]) for row in atom_rows]
    if species is None:
        if not labels:
            raise ValueError("Potfit configuration requires a #C chemical-species header or explicit species mapping")
        numbers = _species_numbers_from_labels(labels, type_ids)
    else:
        numbers = [species[idx] for idx in type_ids]

    positions = [[float(row[1]), float(row[2]), float(row[3])] for row in atom_rows]
    forces = [[float(row[4]), float(row[5]), float(row[6])] for row in atom_rows]

    if not {"X", "Y", "Z"} <= set(cell_rows):
        raise ValueError("Potfit configuration is missing one or more cell vectors (#X, #Y, #Z)")

    cell = [cell_rows["X"], cell_rows["Y"], cell_rows["Z"]]
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    calc = SinglePointCalculator(atoms)
    calc.results["forces"] = np.asarray(forces, dtype=float)
    if energy_per_atom is not None:
        calc.results["energy"] = float(energy_per_atom) * len(atoms)
        calc.results["free_energy"] = float(energy_per_atom) * len(atoms)
    if stress is not None:
        calc.results["stress"] = _stress_from_potfit(stress)
    atoms.calc = calc
    atoms.info["potfit_useforce"] = useforce
    atoms.info["potfit_weight"] = weight
    if box_of_contributing_particles:
        atoms.info["potfit_box"] = box_of_contributing_particles
    return atoms, i


@reader
def read_force(
    fd: TextIO,
    index: int = -1,
    species: list[int] | list[str] | None = None,
) -> Atoms | list[Atoms]:
    """Read potfit-style force configurations.

    Parameters
    ----------
    fd : TextIO
        TextIO object.
    index : int
        Index of images.
    species : list[int] | list[str], optional
        Optional atom-type mapping. If omitted, the chemical symbols are read
        from the ``#C`` header line.

    Returns
    -------
    Atoms | list[Atoms]
    """
    species = _convert_species(species)
    lines = fd.readlines()
    atoms_list: list[Atoms] = []
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        if not raw.startswith("#N"):
            i += 1
            continue
        atoms, i = _read_potfit_config(lines, i, species)
        atoms_list.append(atoms)

    if isinstance(index, str):
        index = string2index(index)
    return atoms_list[index]


@writer
def write_force(
    fd: TextIO,
    images: Atoms | list[Atoms],
    species: list[int] | list[str] | None = None,
    *,
    weight: float = 1.0,
    useforce: int = 1,
) -> None:
    """Write potfit-style force configurations."""
    if isinstance(images, Atoms):
        images = [images]

    species = _convert_species(species)
    if species is None:
        numbers: list[int] = []
        for atoms in images:
            for number in atoms.numbers.tolist():
                if number not in numbers:
                    numbers.append(int(number))
        species = numbers

    labels = _species_labels(species)
    for atoms in images:
        if atoms.calc is None:
            atoms.calc = SinglePointCalculator(atoms)
            atoms.calc.results["forces"] = np.zeros((len(atoms), 3), dtype=float)
            atoms.calc.results["energy"] = 0.0
            atoms.calc.results["free_energy"] = 0.0

        forces = _get_singlepoint(atoms, "forces", shape=(len(atoms), 3))
        energy = float(atoms.get_potential_energy()) / float(len(atoms))

        try:
            stress = np.asarray(atoms.get_stress(), dtype=float)
        except Exception:
            stress = np.array([], dtype=float)

        fd.write(f"#N {len(atoms)} {int(useforce)}\n")
        fd.write(f"#C {' '.join(labels)}\n")
        fd.write("## force file generated by ForgeFF\n")
        cell = np.asarray(atoms.cell.array, dtype=float)
        fd.write(f"#X {cell[0][0]:13.8f} {cell[0][1]:13.8f} {cell[0][2]:13.8f}\n")
        fd.write(f"#Y {cell[1][0]:13.8f} {cell[1][1]:13.8f} {cell[1][2]:13.8f}\n")
        fd.write(f"#Z {cell[2][0]:13.8f} {cell[2][1]:13.8f} {cell[2][2]:13.8f}\n")
        fd.write(f"#W {float(weight):f}\n")
        fd.write(f"#E {energy:.10f}\n")
        if stress.size == 6:
            stress_potfit = _stress_to_potfit(stress)
            fd.write("#S")
            for value in stress_potfit:
                fd.write(f" {value:8.7g}")
            fd.write("\n")
        fd.write("#F\n")

        symbol_to_type = {int(number): idx for idx, number in enumerate(species)}
        for index, number in enumerate(atoms.numbers.tolist()):
            type_index = symbol_to_type[int(number)]
            position = atoms.positions[index]
            force = forces[index]
            fd.write(
                f"{type_index} {position[0]:11.7g} {position[1]:11.7g} {position[2]:11.7g} "
                f"{force[0]:11.7g} {force[1]:11.7g} {force[2]:11.7g}\n"
            )
