"""Tersoff data structures for ForgeFF."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class TersoffParameters:
    """Parameters for a 3-element Tersoff interaction."""

    m: float
    gamma: float
    lambda3: float
    c: float
    d: float
    h: float
    n: float
    beta: float
    lambda2: float
    B: float
    R: float
    D: float
    lambda1: float
    A: float

    @classmethod
    def from_list(cls, params: list[float] | tuple[float, ...]) -> "TersoffParameters":
        if len(params) != 14:
            raise ValueError(f"Expected 14 Tersoff parameters, got {len(params)}")
        return cls(*map(float, params))

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.m,
                self.gamma,
                self.lambda3,
                self.c,
                self.d,
                self.h,
                self.n,
                self.beta,
                self.lambda2,
                self.B,
                self.R,
                self.D,
                self.lambda1,
                self.A,
            ],
            dtype=float,
        )


def _default_parameter_table() -> npt.NDArray[np.float64]:
    return np.zeros((0, 0, 0, 14), dtype=float)


@dataclass
class TersoffData:
    """Dense Tersoff parameter table with a canonical species order."""

    species: list[str] = field(default_factory=list)
    parameter_table: npt.NDArray[np.float64] = field(default_factory=_default_parameter_table)
    cutoff_skin: float = 0.3
    species_energy_offsets: dict[str, float] = field(default_factory=dict)
    optimized: list[str] = field(default_factory=lambda: ["parameter_table"])

    @classmethod
    def from_parameter_dict(
        cls,
        parameters: dict[tuple[str, str, str], TersoffParameters | list[float] | tuple[float, ...]],
        *,
        species: list[str] | None = None,
        cutoff_skin: float = 0.3,
    ) -> "TersoffData":
        if species is None:
            species = []
            for key in parameters:
                for symbol in key:
                    if symbol not in species:
                        species.append(symbol)
        else:
            species = [str(symbol) for symbol in species]

        if not species:
            raise ValueError("Tersoff parameter dictionary must not be empty.")

        spc = len(species)
        table = np.zeros((spc, spc, spc, 14), dtype=float)
        index = {symbol: idx for idx, symbol in enumerate(species)}

        for key, value in parameters.items():
            if len(key) != 3:
                raise ValueError(f"Tersoff parameter key must have three symbols, got {key!r}")
            i, j, k = (index[symbol] for symbol in key)
            params = value if isinstance(value, TersoffParameters) else TersoffParameters.from_list(value)
            table[i, j, k] = params.as_array()

        return cls(species=species, parameter_table=table, cutoff_skin=float(cutoff_skin))

    @property
    def species_count(self) -> int:
        return len(self.species)

    @property
    def parameters(self) -> np.ndarray:
        return np.asarray(self.parameter_table, dtype=float).reshape(-1)

    @parameters.setter
    def parameters(self, parameters: npt.ArrayLike) -> None:
        arr = np.asarray(parameters, dtype=float).reshape(-1)
        if self.species_count == 0:
            raise ValueError("Cannot set flattened parameters before species are defined.")
        expected = self.species_count * self.species_count * self.species_count * 14
        if arr.size != expected:
            raise ValueError(f"Expected {expected} flattened parameters, got {arr.size}")
        self.parameter_table = arr.reshape(self.species_count, self.species_count, self.species_count, 14)

    @property
    def number_of_parameters_optimized(self) -> int:
        return int(self.parameters.size)

    def get_bounds(self) -> list[tuple[float, float]] | None:
        return None

    def initialize(self, rng: np.random.Generator) -> None:
        if self.parameter_table.size == 0:
            return
        if np.all(self.parameter_table == 0):
            self.parameter_table = rng.uniform(-0.1, 0.1, self.parameter_table.shape)

    def log(self) -> None:
        pass

    def write(self, filename: str | Path) -> None:
        np.save(
            filename,
            {
                "species": self.species,
                "parameter_table": self.parameter_table,
                "cutoff_skin": self.cutoff_skin,
                "species_energy_offsets": self.species_energy_offsets,
            },
            allow_pickle=True,
        )

    @classmethod
    def from_file(cls, filename: str | Path) -> "TersoffData":
        data = np.load(filename, allow_pickle=True).item()
        return cls(**data)
