"""ASE Calculators."""

from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.tersoff.data import TersoffData

_EAM_ENGINES = {
    "numpy": "forgeff.potentials.eam.numpy.engine.ASEAMEngine",
    "ase": "forgeff.potentials.eam.numpy.engine.ASEAMEngine",
    "numba": "forgeff.potentials.eam.numba.engine.NumbaEAMEngine",
}

_ADP_ENGINES = {
    "numba": "forgeff.potentials.eam.numba.adp_engine.NumbaADPEngine",
}

_TERSOFF_ENGINES = {
    "numba": "forgeff.potentials.tersoff.numba.NumbaTersoffCalculator",
}

_ASE_ENGINES = {
    "generic": "forgeff.potentials.ase.engine.GenericASEEngine",
}


def make_eam_engine(
    engine: str = "numpy",
) -> type:
    """Return the engine class for the given backend name."""
    import importlib

    if engine not in _EAM_ENGINES:
        raise ValueError(f"Unknown EAM engine {engine!r}. Supported: {sorted(_EAM_ENGINES)}")
    module_path, _, class_name = _EAM_ENGINES[engine].rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def make_adp_engine(
    engine: str = "numba",
) -> type:
    """Return the engine class for the given backend name."""
    import importlib

    if engine not in _ADP_ENGINES:
        raise ValueError(f"Unknown ADP engine {engine!r}. Supported: {sorted(_ADP_ENGINES)}")
    module_path, _, class_name = _ADP_ENGINES[engine].rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def make_tersoff_engine(
    engine: str = "numba",
) -> type:
    """Return the engine class for the given Tersoff backend name."""
    import importlib

    if engine not in _TERSOFF_ENGINES:
        raise ValueError(f"Unknown Tersoff engine {engine!r}. Supported: {sorted(_TERSOFF_ENGINES)}")
    module_path, _, class_name = _TERSOFF_ENGINES[engine].rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def make_ase_engine(
    engine: str = "generic",
) -> type:
    """Return the engine class for the given backend name."""
    import importlib

    if engine not in _ASE_ENGINES:
        raise ValueError(f"Unknown ASE engine {engine!r}. Supported: {sorted(_ASE_ENGINES)}")
    module_path, _, class_name = _ASE_ENGINES[engine].rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def make_calculator(
    pot_data: Any,
    engine: str = "cext",
    mode: str = "run",
    **kwargs: dict,
) -> "EAM | ADP | ASECalculatorWrapper":
    """Create the appropriate calculator based on data type.

    Parameters
    ----------
    pot_data : Any
        Potential data.
    engine : str
        Backend name (``"cext"``, ``"numba"``, etc.).
    mode : str
        Operation mode.
    **kwargs
        Forwarded to the calculator constructor.

    Returns
    -------
    Calculator for the matching potential family.

    """
    if isinstance(pot_data, ASEData):
        return ASECalculatorWrapper(pot_data, engine="generic", mode=mode, **kwargs)
    if isinstance(pot_data, ADPData):
        if engine not in _ADP_ENGINES:
            engine = "numba"
        return ADP(pot_data, engine=engine, mode=mode, **kwargs)
    if isinstance(pot_data, TersoffData):
        if engine not in _TERSOFF_ENGINES:
            engine = "numba"
        engine_cls = make_tersoff_engine(engine)
        return engine_cls(pot_data, **kwargs)
    if isinstance(pot_data, EAMData):
        if engine not in _EAM_ENGINES:
            engine = "numpy"
        return EAM(pot_data, engine=engine, mode=mode, **kwargs)
    raise TypeError(f"Unsupported potential data: {type(pot_data).__name__}")


class EAM(Calculator):
    """ASE Calculator of the EAM potential."""

    implemented_properties = (
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    )

    def __init__(
        self,
        eam_data: EAMData,
        *args: tuple,
        engine: str = "numpy",
        mode: str = "run",
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        engine_cls = make_eam_engine(engine)
        self.engine = engine_cls(eam_data, mode=mode)

    def update_parameters(self, eam_data: EAMData) -> None:
        """Update EAM parameters."""
        self.engine.update(eam_data)
        self.results = {}  # trigger new calculation

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate."""
        super().calculate(atoms, properties, system_changes)

        self.results = self.engine.calculate(self.atoms)

        self.results["free_energy"] = self.results["energy"]

        if self.atoms.cell.rank != 3 and "stress" in self.results:
            del self.results["stress"]


class ADP(Calculator):
    """ASE Calculator of the ADP potential."""

    implemented_properties = (
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    )

    def __init__(
        self,
        adp_data: ADPData,
        *args: tuple,
        engine: str = "numba",
        mode: str = "run",
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        engine_cls = make_adp_engine(engine)
        self.engine = engine_cls(adp_data, mode=mode)

    def update_parameters(self, adp_data: ADPData) -> None:
        """Update ADP parameters."""
        self.engine.update(adp_data)
        self.results = {}  # trigger new calculation

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate."""
        super().calculate(atoms, properties, system_changes)

        self.results = self.engine.calculate(self.atoms)

        self.results["free_energy"] = self.results["energy"]

        if self.atoms.cell.rank != 3 and "stress" in self.results:
            del self.results["stress"]


class ASECalculatorWrapper(Calculator):
    """ASE Calculator wrapper for any ASE potential."""

    implemented_properties = (
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    )

    def __init__(
        self,
        ase_data: ASEData,
        *args: tuple,
        engine: str = "generic",
        mode: str = "run",
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        engine_cls = make_ase_engine(engine)
        self.engine = engine_cls(ase_data, mode=mode)

    def update_parameters(self, ase_data: ASEData) -> None:
        """Update parameters."""
        self.engine.update(ase_data)
        self.results = {}  # trigger new calculation

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate."""
        super().calculate(atoms, properties, system_changes)

        self.results = self.engine.calculate(self.atoms)

        self.results["free_energy"] = self.results["energy"]

        if self.atoms.cell.rank != 3 and "stress" in self.results:
            del self.results["stress"]


ASEPotentialCalculator = ASECalculatorWrapper

__all__ = [
    "ADP",
    "ASEPotentialCalculator",
    "EAM",
    "TersoffData",
    "make_calculator",
]
