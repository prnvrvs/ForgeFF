"""Built-in analytical form registry for ForgeFF."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sympy import Symbol, diff, lambdify, sympify


FORMULA_LIBRARY: dict[str, dict[str, Any]] = {
    "lj": {
        "formula": "4*epsilon*((sigma/r)**12 - (sigma/r)**6)",
        "params": ["epsilon", "sigma"],
        "default_bounds": [(0.0, 10.0), (0.1, 10.0)],
        "variable": "r",
    },
    "bornmayer": {
        "formula": "A * exp(-r / rho)",
        "params": ["A", "rho"],
        "default_bounds": [(0.0, 1000.0), (0.1, 10.0)],
        "variable": "r",
    },
    "morse": {
        "formula": "De * (exp(-2.0 * a * (r - re)) - 2.0 * exp(-a * (r - re)))",
        "params": ["De", "a", "re"],
        "default_bounds": [(0.0, 10.0), (0.1, 10.0), (1.0, 6.0)],
        "variable": "r",
    },
    "double_morse": {
        "formula": "E1 * (exp(-2.0 * a1 * (r - r1)) - 2.0 * exp(-a1 * (r - r1))) + "
        "E2 * (exp(-2.0 * a2 * (r - r2)) - 2.0 * exp(-a2 * (r - r2))) + delta",
        "params": ["E1", "a1", "r1", "E2", "a2", "r2", "delta"],
        "default_bounds": [(0.0, 5.0), (0.1, 5.0), (1.0, 6.0), (0.0, 5.0), (0.1, 5.0), (1.0, 6.0), (-2.0, 2.0)],
        "variable": "r",
    },
    "power_decay": {
        "formula": "alpha * (1.0 / r)**beta",
        "params": ["alpha", "beta"],
        "default_bounds": [(0.01, 20.0), (0.1, 15.0)],
        "variable": "r",
    },
    "exp_decay": {
        "formula": "alpha * exp(-beta * r)",
        "params": ["alpha", "beta"],
        "default_bounds": [(0.0, 20.0), (0.1, 10.0)],
        "variable": "r",
    },
    "constant": {
        "formula": "c",
        "params": ["c"],
        "default_bounds": [(-10.0, 10.0)],
        "variable": "r",
    },
    "coul": {
        "formula": "14.3996454784255 * q1 * q2 / r",
        "params": ["q1", "q2"],
        "default_bounds": [(-10.0, 10.0), (-10.0, 10.0)],
        "variable": "r",
    },
    "exponential": {
        "formula": "A * r**n",
        "params": ["A", "n"],
        "default_bounds": [(-100.0, 100.0), (-10.0, 10.0)],
        "variable": "r",
    },
    "hbnd": {
        "formula": "A / r**12 - B / r**10",
        "params": ["A", "B"],
        "default_bounds": [(0.0, 100.0), (0.0, 100.0)],
        "variable": "r",
    },
    "sqrt": {
        "formula": "alpha * sqrt(rho / beta)",
        "params": ["alpha", "beta"],
        "default_bounds": [(0.0, 100.0), (0.01, 10.0)],
        "variable": "rho",
    },
    "buck": {
        "formula": "A * exp(-r / rho) - C / r**6",
        "params": ["A", "rho", "C"],
        "default_bounds": [(0.0, 1000.0), (0.1, 1.0), (0.0, 100.0)],
        "variable": "r",
    },
    "eopp": {
        "formula": "C1 / r**eta1 + (C2 / r**eta2) * cos(k * r + phi)",
        "params": ["C1", "eta1", "C2", "eta2", "k", "phi"],
        "default_bounds": [(1.0, 10000.0), (1.0, 20.0), (-100.0, 100.0), (1.0, 10.0), (0.0, 10.0), (0.0, 6.3)],
        "variable": "r",
    },
    "csw": {
        "formula": "(1.0 + c1 * cos(k * r) + c2 * sin(k * r)) / r**power",
        "params": ["c1", "c2", "k", "power"],
        "default_bounds": [(-10.0, 10.0), (-10.0, 10.0), (0.0, 10.0), (1.0, 15.0)],
        "variable": "r",
    },
    "csw2": {
        "formula": "(1.0 + c1 * cos(k * r + phi)) / r**power",
        "params": ["c1", "k", "phi", "power"],
        "default_bounds": [(-10.0, 10.0), (0.0, 10.0), (0.0, 6.3), (1.0, 15.0)],
        "variable": "r",
    },
    "ms": {
        "formula": "De * (exp(a * (1.0 - r / r0)) - 2.0 * exp(0.5 * a * (1.0 - r / r0)))",
        "params": ["De", "a", "r0"],
        "default_bounds": [(0.0, 10.0), (0.1, 10.0), (1.0, 10.0)],
        "variable": "r",
    },
    "born": {
        "formula": "A * exp((r0 - r) / sigma) - C / r**6 + D / r**8",
        "params": ["A", "sigma", "r0", "C", "D"],
        "default_bounds": [(0.0, 100.0), (0.1, 10.0), (1.0, 8.0), (0.0, 100.0), (0.0, 100.0)],
        "variable": "r",
    },
    "softshell": {
        "formula": "(alpha / r)**beta",
        "params": ["alpha", "beta"],
        "default_bounds": [(0.1, 10.0), (1.0, 10.0)],
        "variable": "r",
    },
    "exp_plus": {
        "formula": "alpha * exp(-beta * r) + c",
        "params": ["alpha", "beta", "c"],
        "default_bounds": [(0.0, 20.0), (0.0, 10.0), (-10.0, 10.0)],
        "variable": "r",
    },
    "mexp_decay": {
        "formula": "alpha * exp(-beta * (r - r0))",
        "params": ["alpha", "beta", "r0"],
        "default_bounds": [(0.0, 20.0), (0.0, 10.0), (0.0, 10.0)],
        "variable": "r",
    },
    "strmm": {
        "formula": "2.0 * alpha * exp(-beta / 2.0 * (r - r0)) - gamma * (1.0 + delta * (r - r0)) * exp(-delta * (r - r0))",
        "params": ["alpha", "beta", "gamma", "delta", "r0"],
        "default_bounds": [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
        "variable": "r",
    },
    "poly_5": {
        "formula": "p0 + 0.5 * p1 * (r - 1.0)**2 + p2 * (r - 1.0)**3 + p3 * (r - 1.0)**4 + p4 * (r - 1.0)**5",
        "params": ["p0", "p1", "p2", "p3", "p4"],
        "default_bounds": [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
        "variable": "r",
    },
    "zero": {
        "formula": "0.0",
        "params": [],
        "default_bounds": [],
        "variable": "r",
    },
}


def get_form_spec(name: str) -> dict[str, Any]:
    """Return a built-in analytical form specification."""
    key = name.lower()
    if key not in FORMULA_LIBRARY:
        raise KeyError(f"Unknown analytical form: {name!r}")
    return FORMULA_LIBRARY[key]


@lru_cache(maxsize=None)
def _compile_expression(expression: str, variable: str, parameter_names: tuple[str, ...]):
    var = Symbol(variable)
    params = [Symbol(name) for name in parameter_names]
    local_dict = {variable: var, **{name: sym for name, sym in zip(parameter_names, params, strict=True)}}
    expr = sympify(expression, locals=local_dict)
    return lambdify((var, *params), expr, modules="numpy")


def evaluate_expression(
    expression: str,
    *,
    variable: str,
    parameter_names: list[str] | tuple[str, ...],
    x: np.ndarray,
    parameters: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """Evaluate a symbolic expression on a grid."""
    fn = _compile_expression(expression, variable, tuple(parameter_names))
    values = fn(np.asarray(x, dtype=float), *np.asarray(parameters, dtype=float))
    return np.asarray(values, dtype=float)


def evaluate_form(
    name: str,
    *,
    x: np.ndarray,
    parameters: list[float] | tuple[float, ...] | np.ndarray,
    variable: str | None = None,
) -> np.ndarray:
    """Evaluate a built-in analytical form on a grid."""
    spec = get_form_spec(name)
    if spec["params"] == []:
        return np.zeros_like(np.asarray(x, dtype=float))
    return evaluate_expression(
        spec["formula"],
        variable=variable or spec.get("variable", "r"),
        parameter_names=spec["params"],
        x=x,
        parameters=parameters,
    )
