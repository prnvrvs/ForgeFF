"""Custom analytic pair potential for ASE-backed fitting."""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress
from sympy import Symbol, diff, lambdify, sympify


_RESERVED_KEYS = {
    "expression",
    "parameter_names",
    "variable",
    "cutoff",
    "rc",
}


@dataclass
class _CompiledExpression:
    energy: Any
    derivative: Any


class CustomPairPotential(Calculator):
    """Analytic pair potential defined by a user expression.

    The expression is interpreted as a function of one interatomic distance
    variable, typically ``r``, and any number of scalar parameters.
    """

    implemented_properties = (
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    )

    def __init__(
        self,
        *,
        expression: str,
        parameter_names: list[str] | tuple[str, ...] | None = None,
        variable: str = "r",
        cutoff: float | None = None,
        rc: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(expression=expression, parameter_names=parameter_names, variable=variable, cutoff=cutoff, rc=rc, **kwargs)
        self.expression = expression
        self.variable = variable
        self.cutoff = cutoff if cutoff is not None else rc
        if self.cutoff is None:
            raise ValueError("CustomPairPotential requires a finite 'cutoff' (or 'rc').")

        if parameter_names is None:
            parameter_names = [key for key in kwargs if key not in _RESERVED_KEYS]
        self.parameter_names = list(parameter_names)

        missing = [name for name in self.parameter_names if name not in kwargs]
        if missing:
            raise ValueError(f"Missing values for custom parameters: {missing}")

        self._parameter_values = np.array([float(kwargs[name]) for name in self.parameter_names], dtype=float)
        self._parameter_index = {name: idx for idx, name in enumerate(self.parameter_names)}
        self._compile()

    @staticmethod
    @lru_cache(maxsize=None)
    def _compiled_expression(
        expression: str,
        variable: str,
        parameter_names: tuple[str, ...],
    ) -> _CompiledExpression:
        var = Symbol(variable)
        param_symbols = [Symbol(name) for name in parameter_names]
        local_dict = {variable: var}
        local_dict.update({name: sym for name, sym in zip(parameter_names, param_symbols, strict=True)})
        expr = sympify(expression, locals=local_dict)
        return _CompiledExpression(
            energy=lambdify((var, *param_symbols), expr, modules="numpy"),
            derivative=lambdify((var, *param_symbols), diff(expr, var), modules="numpy"),
        )

    def _compile(self) -> None:
        self._compiled = self._compiled_expression(
            self.expression,
            self.variable,
            tuple(self.parameter_names),
        )

    def update_parameters(self, **kwargs: Any) -> None:
        """Update parameter values without recompiling the expression."""
        for name, value in kwargs.items():
            idx = self._parameter_index.get(name)
            if idx is not None:
                self._parameter_values[idx] = float(value)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):  # noqa: B006
        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)
        energy = 0.0
        forces = np.zeros((natoms, 3), dtype=float)
        virial = np.zeros((3, 3), dtype=float)

        i_list, j_list, shifts, dist = neighbor_list("ijSd", self.atoms, float(self.cutoff))
        if len(i_list):
            vec = self.atoms.positions[j_list] + shifts @ self.atoms.cell.array - self.atoms.positions[i_list]
            args = (np.asarray(dist, dtype=float), *self._parameter_values)
            pair_energy = np.asarray(self._compiled.energy(*args), dtype=float)
            d_v_dr = np.asarray(self._compiled.derivative(*args), dtype=float)
            if pair_energy.ndim == 0:
                pair_energy = np.full(dist.shape, float(pair_energy), dtype=float)
            else:
                pair_energy = np.asarray(pair_energy, dtype=float)
            if d_v_dr.ndim == 0:
                d_v_dr = np.full(dist.shape, float(d_v_dr), dtype=float)
            else:
                d_v_dr = np.asarray(d_v_dr, dtype=float)
            unit = vec / dist[:, None]
            fij = -d_v_dr[:, None] * unit

            energy = float(np.sum(pair_energy))
            np.add.at(forces, i_list, fij)
            np.add.at(forces, j_list, -fij)
            virial = np.einsum("ni,nj->ij", vec, fij, optimize=True)

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        local = np.zeros(natoms, dtype=float)
        if len(i_list):
            pair_local = 0.5 * np.asarray(pair_energy, dtype=float)
            np.add.at(local, i_list, pair_local)
            np.add.at(local, j_list, pair_local)
        self.results["energies"] = local
        self.results["forces"] = forces

        if self.atoms.cell.rank == 3 and self.atoms.get_volume() != 0.0:
            self.results["stress"] = full_3x3_to_voigt_6_stress(-virial / self.atoms.get_volume())
