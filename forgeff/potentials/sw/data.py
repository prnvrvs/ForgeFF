"""Stillinger-Weber data for ForgeFF."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from ase.data import atomic_numbers, chemical_symbols


PAIR_PARAMETER_NAMES = ("A", "B", "p", "q", "delta", "a1", "gamma", "a2")
PAIR_PARAMETER_COUNT = len(PAIR_PARAMETER_NAMES)

_PAIR_BOUNDS = (
    (0.0, 200.0),
    (0.0, 200.0),
    (0.0, 20.0),
    (0.0, 20.0),
    (0.0, 50.0),
    (0.1, 50.0),
    (0.0, 50.0),
    (0.1, 50.0),
)
_LAMBDA_BOUNDS = (0.0, 200.0)


def _normalize_species_label(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(chemical_symbols[int(value)])
    return str(value)


def _species_number(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    label = str(value)
    if label not in atomic_numbers:
        raise ValueError(f"Unknown species label in SW data: {value!r}")
    return int(atomic_numbers[label])


def _canonical_pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i <= j else (j, i)


def _canonical_lambda(i: int, j: int, k: int) -> tuple[int, int, int]:
    left, right = sorted((j, k))
    return i, left, right


def _as_array(value: npt.ArrayLike | None, *, ndim: int, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required SW array: {name}")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != ndim:
        raise ValueError(f"Expected {name} to have {ndim} dimensions, got {arr.ndim}.")
    return arr


def _legacy_pair_parameters(epsilon: float, sigma: float, A: float, B: float, p: float, a: float, gamma: float) -> np.ndarray:
    return np.asarray(
        [
            epsilon * A * B * sigma**p,
            epsilon * A,
            p,
            0.0,
            sigma,
            a * sigma,
            gamma * sigma,
            a * sigma,
        ],
        dtype=float,
    )


@dataclass
class SWData:
    """Potfit-style multispecies Stillinger-Weber parameter set."""

    species: list[str] = field(default_factory=lambda: ["Si"])
    pair_parameters: npt.ArrayLike | None = None
    lambda_values: npt.ArrayLike | None = None
    epsilon: float = 2.1683
    sigma: float = 2.0951
    costheta0: float = 1.0 / 3.0
    A: float = 7.049556277
    B: float = 0.6022245584
    p: float = 4.0
    a: float = 1.8
    lambda1: float = 21.0
    gamma: float = 1.2
    species_energy_offsets: dict[str, float] = field(default_factory=dict)
    optimized: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.species = [_normalize_species_label(item) for item in self.species]
        species_count = len(self.species)
        if species_count == 0:
            raise ValueError("Stillinger-Weber requires at least one species.")

        if self.pair_parameters is None and self.lambda_values is None:
            if species_count != 1:
                raise ValueError(
                    "Multispecies Stillinger-Weber requires explicit pair_parameters and lambda_values."
                )
            pair = np.zeros((1, 1, PAIR_PARAMETER_COUNT), dtype=float)
            pair[0, 0] = _legacy_pair_parameters(self.epsilon, self.sigma, self.A, self.B, self.p, self.a, self.gamma)
            lambdas = np.zeros((1, 1, 1), dtype=float)
            lambdas[0, 0, 0] = float(self.lambda1) * float(self.epsilon)
            self.pair_parameters = pair
            self.lambda_values = lambdas
        elif self.pair_parameters is None or self.lambda_values is None:
            raise ValueError("SWData requires both pair_parameters and lambda_values when using the multispecies layout.")
        else:
            pair = _as_array(self.pair_parameters, ndim=3, name="pair_parameters")
            lambdas = _as_array(self.lambda_values, ndim=3, name="lambda_values")
            if pair.shape != (species_count, species_count, PAIR_PARAMETER_COUNT):
                raise ValueError(
                    "SW pair_parameters must have shape "
                    f"({species_count}, {species_count}, {PAIR_PARAMETER_COUNT}), got {pair.shape}."
                )
            if lambdas.shape != (species_count, species_count, species_count):
                raise ValueError(
                    "SW lambda_values must have shape "
                    f"({species_count}, {species_count}, {species_count}), got {lambdas.shape}."
                )
            self.pair_parameters = pair.copy()
            self.lambda_values = lambdas.copy()

        self._enforce_symmetry()
        if not self.optimized:
            self.optimized = self._default_optimized_names()

    def _enforce_symmetry(self) -> None:
        pair = np.asarray(self.pair_parameters, dtype=float)
        lambdas = np.asarray(self.lambda_values, dtype=float)
        spc = len(self.species)
        for i in range(spc):
            for j in range(i, spc):
                pair[j, i] = pair[i, j]
        for i in range(spc):
            for j in range(spc):
                for k in range(j, spc):
                    if j <= k:
                        lambdas[i, k, j] = lambdas[i, j, k]
        self.pair_parameters = pair
        self.lambda_values = lambdas

    @property
    def species_count(self) -> int:
        return len(self.species)

    @property
    def species_numbers(self) -> np.ndarray:
        return np.asarray([_species_number(item) for item in self.species], dtype=np.int32)

    @property
    def pair_count(self) -> int:
        return self.species_count * (self.species_count + 1) // 2

    @property
    def lambda_count(self) -> int:
        return self.species_count * self.pair_count

    @property
    def max_cutoff(self) -> float:
        pair = np.asarray(self.pair_parameters, dtype=float)
        return float(np.max(np.maximum(pair[:, :, 5], pair[:, :, 7])))

    def species_index(self, atomic_number: int) -> int:
        numbers = self.species_numbers.tolist()
        try:
            return numbers.index(int(atomic_number))
        except ValueError as exc:
            raise ValueError(f"Atom type {atomic_number} is not present in SW species {self.species}.") from exc

    def pair_parameter_block(self, i: int, j: int) -> np.ndarray:
        i, j = _canonical_pair(i, j)
        return np.asarray(self.pair_parameters[i, j], dtype=float)

    def lambda_parameter(self, i: int, j: int, k: int) -> float:
        i, j, k = _canonical_lambda(i, j, k)
        return float(np.asarray(self.lambda_values, dtype=float)[i, j, k])

    def _default_optimized_names(self) -> list[str]:
        names: list[str] = []
        for i in range(self.species_count):
            for j in range(i, self.species_count):
                left = self.species[i]
                right = self.species[j]
                names.extend(f"{left}{right}_{name}" for name in PAIR_PARAMETER_NAMES)
        for i in range(self.species_count):
            for j in range(self.species_count):
                for k in range(j, self.species_count):
                    names.append(f"lambda_{self.species[i]}{self.species[j]}{self.species[k]}")
        return names

    @property
    def parameters(self) -> np.ndarray:
        values: list[float] = []
        pair = np.asarray(self.pair_parameters, dtype=float)
        lambdas = np.asarray(self.lambda_values, dtype=float)
        for i in range(self.species_count):
            for j in range(i, self.species_count):
                values.extend(pair[i, j].tolist())
        for i in range(self.species_count):
            for j in range(self.species_count):
                for k in range(j, self.species_count):
                    values.append(float(lambdas[i, j, k]))
        return np.asarray(values, dtype=float)

    @parameters.setter
    def parameters(self, parameters: npt.ArrayLike) -> None:
        arr = np.asarray(parameters, dtype=float).reshape(-1)
        expected = self.number_of_parameters_optimized
        if arr.size != expected:
            raise ValueError(f"Expected {expected} SW parameters, got {arr.size}")

        pair = np.asarray(self.pair_parameters, dtype=float).copy()
        lambdas = np.asarray(self.lambda_values, dtype=float).copy()
        offset = 0
        for i in range(self.species_count):
            for j in range(i, self.species_count):
                pair[i, j] = arr[offset : offset + PAIR_PARAMETER_COUNT]
                pair[j, i] = pair[i, j]
                offset += PAIR_PARAMETER_COUNT
        for i in range(self.species_count):
            for j in range(self.species_count):
                for k in range(j, self.species_count):
                    lambdas[i, j, k] = arr[offset]
                    lambdas[i, k, j] = arr[offset]
                    offset += 1
        self.pair_parameters = pair
        self.lambda_values = lambdas

    @property
    def number_of_parameters_optimized(self) -> int:
        return self.pair_count * PAIR_PARAMETER_COUNT + self.lambda_count

    def get_bounds(self) -> list[tuple[float, float]] | None:
        bounds = [*(_PAIR_BOUNDS)] * self.pair_count
        bounds.extend([_LAMBDA_BOUNDS] * self.lambda_count)
        return bounds

    def initialize(self, rng: np.random.Generator) -> None:
        if np.allclose(self.parameters, 0.0):
            bounds = self.get_bounds() or []
            values = np.asarray([rng.uniform(low, high) for low, high in bounds], dtype=float)
            self.parameters = values

    def log(self) -> None:
        pass

    def write(self, filename: str | Path) -> None:
        np.save(
            filename,
            {
                "species": self.species,
                "pair_parameters": np.asarray(self.pair_parameters, dtype=float),
                "lambda_values": np.asarray(self.lambda_values, dtype=float),
                "epsilon": self.epsilon,
                "sigma": self.sigma,
                "costheta0": self.costheta0,
                "A": self.A,
                "B": self.B,
                "p": self.p,
                "a": self.a,
                "lambda1": self.lambda1,
                "gamma": self.gamma,
                "species_energy_offsets": self.species_energy_offsets,
                "optimized": self.optimized,
            },
            allow_pickle=True,
        )

    @classmethod
    def from_file(cls, filename: str | Path) -> "SWData":
        data = np.load(filename, allow_pickle=True).item()
        if "pair_parameters" in data and "lambda_values" in data:
            return cls(**data)
        return cls(
            species=data.get("species", ["Si"]),
            epsilon=float(data.get("epsilon", 2.1683)),
            sigma=float(data.get("sigma", 2.0951)),
            costheta0=float(data.get("costheta0", 1.0 / 3.0)),
            A=float(data.get("A", 7.049556277)),
            B=float(data.get("B", 0.6022245584)),
            p=float(data.get("p", 4.0)),
            a=float(data.get("a", 1.8)),
            lambda1=float(data.get("lambda1", 21.0)),
            gamma=float(data.get("gamma", 1.2)),
            species_energy_offsets=dict(data.get("species_energy_offsets", {})),
        )
