"""Helpers for optional per-species energy offsets."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols


def normalize_species_label(value: Any) -> str:
    """Normalize a species label to a chemical-symbol string."""
    if isinstance(value, (int, np.integer)):
        return str(chemical_symbols[int(value)])
    return str(value)


def apply_species_energy_offsets(
    results: dict[str, Any],
    atoms: Atoms,
    pot_data: Any,
) -> dict[str, Any]:
    """Add per-species offsets to calculator outputs."""
    offsets = getattr(pot_data, "species_energy_offsets", {}) or {}
    if not offsets:
        return results

    normalized = {normalize_species_label(key).lower(): float(value) for key, value in offsets.items()}
    per_atom = np.zeros(len(atoms), dtype=float)
    total = 0.0
    for idx, number in enumerate(atoms.numbers.tolist()):
        label = normalize_species_label(number).lower()
        value = normalized.get(label, 0.0)
        per_atom[idx] = value
        total += value

    adjusted = dict(results)
    adjusted["energy"] = float(adjusted["energy"]) + total
    if "free_energy" in adjusted:
        adjusted["free_energy"] = float(adjusted["free_energy"]) + total
    else:
        adjusted["free_energy"] = adjusted["energy"]
    if "energies" in adjusted:
        adjusted["energies"] = np.asarray(adjusted["energies"], dtype=float) + per_atom
    return adjusted
