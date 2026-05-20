"""ASE-backed potential helpers."""

from __future__ import annotations

from .data import ASEData
from .engine import GenericASEEngine
from .forms import FORMULA_LIBRARY, evaluate_form, get_form_spec

__all__ = [
    "ASEData",
    "FORMULA_LIBRARY",
    "GenericASEEngine",
    "CustomPairPotential",
    "NumbaPairPotential",
    "evaluate_form",
    "get_form_spec",
]


def __getattr__(name: str):
    if name == "CustomPairPotential":
        from .custom import CustomPairPotential

        return CustomPairPotential
    if name == "NumbaPairPotential":
        from .numba_pair import NumbaPairPotential

        return NumbaPairPotential
    raise AttributeError(name)
