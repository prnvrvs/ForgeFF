"""Reusable neighbor-list cache for ASE-backed pair calculators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


@dataclass
class NeighborCache:
    """Cache the structural neighbor data for a fixed Atoms configuration."""

    numbers: np.ndarray | None = None
    positions: np.ndarray | None = None
    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None
    cutoff: float | None = None
    i_list: np.ndarray | None = None
    j_list: np.ndarray | None = None
    shifts: np.ndarray | None = None
    dist: np.ndarray | None = None
    vectors: np.ndarray | None = None

    def _matches(self, atoms: Atoms, cutoff: float) -> bool:
        if self.cutoff != cutoff:
            return False
        if self.numbers is None or self.positions is None or self.cell is None or self.pbc is None:
            return False
        if not np.array_equal(self.numbers, atoms.numbers):
            return False
        if not np.array_equal(self.positions, atoms.positions):
            return False
        if not np.array_equal(self.cell, atoms.cell.array):
            return False
        return np.array_equal(self.pbc, atoms.pbc)

    def get(self, atoms: Atoms, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return cached neighbor arrays, rebuilding them if the structure changed."""
        if not self._matches(atoms, cutoff):
            i_list, j_list, shifts, dist = neighbor_list("ijSd", atoms, cutoff)
            vectors = atoms.positions[j_list] + shifts @ atoms.cell.array - atoms.positions[i_list]
            self.numbers = np.asarray(atoms.numbers, dtype=np.int64).copy()
            self.positions = np.asarray(atoms.positions, dtype=float).copy()
            self.cell = np.asarray(atoms.cell.array, dtype=float).copy()
            self.pbc = np.asarray(atoms.pbc, dtype=bool).copy()
            self.cutoff = float(cutoff)
            self.i_list = np.asarray(i_list, dtype=np.int64)
            self.j_list = np.asarray(j_list, dtype=np.int64)
            self.shifts = np.asarray(shifts, dtype=np.int64)
            self.dist = np.asarray(dist, dtype=float)
            self.vectors = np.asarray(vectors, dtype=float)
        assert self.i_list is not None
        assert self.j_list is not None
        assert self.shifts is not None
        assert self.dist is not None
        assert self.vectors is not None
        return self.i_list, self.j_list, self.shifts, self.dist, self.vectors
