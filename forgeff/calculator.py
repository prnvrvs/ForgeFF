"""ASE Calculators."""

import warnings
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.ase.engine import GenericASEEngine
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.sw.data import SWData
from forgeff.potentials.tersoff.data import TersoffData

_EAM_ENGINES = {
    "ase": "forgeff.potentials.eam.numpy.engine.NumpyEAMEngine",
    "numpy": "forgeff.potentials.eam.numpy.engine.NumpyEAMEngine",
    "numba": "forgeff.potentials.eam.numba.engine.NumbaEAMEngine",
}

_ADP_ENGINES = {
    "numpy": "forgeff.potentials.eam.numpy.adp_engine.NumpyADPEngine",
    "numba": "forgeff.potentials.eam.numba.adp_engine.NumbaADPEngine",
}

_TERSOFF_ENGINES = {
    "numba": "forgeff.potentials.tersoff.numba.NumbaTersoffCalculator",
}

_SW_ENGINES = {
    "numpy": "forgeff.potentials.sw.numpy.NumpySWEngine",
    "numba": "forgeff.potentials.sw.numba.NumbaSWEngine",
}

_ASE_ANALYTICAL_CALCULATORS = {
    "lj": "LennardJones",
    "morse": "MorsePotential",
}


def make_eam_engine(
    engine: str = "numpy",
) -> type:
    """Return the engine class for the given engine name."""
    import importlib

    if engine not in _EAM_ENGINES:
        raise ValueError(f"Unknown EAM engine {engine!r}. Supported: {sorted(_EAM_ENGINES)}")
    module_path, _, class_name = _EAM_ENGINES[engine].rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def make_adp_engine(
    engine: str = "numba",
) -> type:
    """Return the engine class for the given engine name."""
    import importlib

    if engine not in _ADP_ENGINES:
        raise ValueError(f"Unknown ADP engine {engine!r}. Supported: {sorted(_ADP_ENGINES)}")
    module_path, _, class_name = _ADP_ENGINES[engine].rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def make_tersoff_engine(
    engine: str = "numba",
) -> type:
    """Return the engine class for the given Tersoff engine name."""
    import importlib

    if engine not in _TERSOFF_ENGINES:
        raise ValueError(f"Unknown Tersoff engine {engine!r}. Supported: {sorted(_TERSOFF_ENGINES)}")
    module_path, _, class_name = _TERSOFF_ENGINES[engine].rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def make_sw_engine(
    engine: str = "numba",
) -> type:
    """Return the Stillinger-Weber engine class for the given engine name."""
    import importlib

    if engine not in _SW_ENGINES:
        raise ValueError(f"Unknown SW engine {engine!r}. Supported: {sorted(_SW_ENGINES)}")
    module_path, _, class_name = _SW_ENGINES[engine].rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def make_calculator(
    pot_data: Any,
    engine: str = "numpy",
    form: str | None = None,
    mode: str = "run",
    **kwargs: dict,
) -> "EAM | ADP | SW | ASECalculatorWrapper":
    """Create the appropriate calculator based on data type.

    Parameters
    ----------
    pot_data : Any
        Potential data.
    engine : str
        Engine name (``"numpy"``, ``"numba"``, etc.).
    form : str | None
        Potential form. Pass this explicitly when the file format or calculator
        family needs it.
    mode : str
        Operation mode.
    **kwargs
        Forwarded to the calculator constructor.

    Returns
    -------
    Calculator for the matching potential family.

    """
    engine_name = str(engine).lower()
    form_name = None if form is None else str(form).lower()
    if isinstance(pot_data, ASEData):
        calculator_kwargs = dict(getattr(pot_data, "calculator_kwargs", {}))
        if form_name is not None and "form" not in calculator_kwargs:
            calculator_kwargs["form"] = form_name
        if engine_name == "ase":
            ase_form = str(calculator_kwargs.get("form", "")).lower()
            if "calculator" not in calculator_kwargs:
                if ase_form not in _ASE_ANALYTICAL_CALCULATORS:
                    warnings.warn(
                        f"ASE does not support analytical form {ase_form!r}; falling back to engine='numpy'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    engine_name = "numpy"
                else:
                    calculator_kwargs["calculator"] = _ASE_ANALYTICAL_CALCULATORS[ase_form]
            pot_data.engine = "ASE"
            if engine_name == "ase":
                return ASECalculatorWrapper(pot_data, mode=mode, **kwargs)
        if engine_name == "numba" and "form" not in calculator_kwargs and "expression" in calculator_kwargs:
            warnings.warn(
                "Analytical custom expressions do not support engine='numba'; falling back to engine='numpy'.",
                RuntimeWarning,
                stacklevel=2,
            )
            engine_name = "numpy"
        if form_name is not None and hasattr(pot_data, "form"):
            pot_data.form = form_name
        pot_data.engine = engine_name
        pot_data.calculator_kwargs = calculator_kwargs
        return ASECalculatorWrapper(pot_data, mode=mode, **kwargs)
    if isinstance(pot_data, ADPData):
        if form_name is not None and hasattr(pot_data, "form"):
            pot_data.form = form_name
        if engine_name not in _ADP_ENGINES:
            if engine_name == "ase":
                warnings.warn(
                    "ASE does not support ADP fitting; falling back to engine='numba'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            engine_name = "numba"
        return ADP(pot_data, engine=engine_name, mode=mode, **kwargs)
    if isinstance(pot_data, TersoffData):
        if form_name is not None and hasattr(pot_data, "form"):
            pot_data.form = form_name
        if engine_name not in _TERSOFF_ENGINES:
            if engine_name == "ase":
                warnings.warn(
                    "ASE does not support Tersoff fitting; falling back to engine='numba'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            engine_name = "numba"
        engine_cls = make_tersoff_engine(engine_name)
        return engine_cls(pot_data, **kwargs)
    if isinstance(pot_data, SWData):
        if form_name is not None and hasattr(pot_data, "form"):
            pot_data.form = form_name
        if engine_name not in _SW_ENGINES:
            if engine_name == "ase":
                warnings.warn(
                    "ASE does not support Stillinger-Weber; falling back to engine='numba'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                engine_name = "numba"
            else:
                raise ValueError(f"Unknown SW engine {engine!r}. Supported: {sorted(_SW_ENGINES)}")
        return SW(pot_data, engine=engine_name, mode=mode, **kwargs)
    if isinstance(pot_data, EAMData):
        if form_name is not None and hasattr(pot_data, "form"):
            pot_data.form = form_name
        if engine_name not in _EAM_ENGINES:
            engine_name = "numpy"
        return EAM(pot_data, engine=engine_name, mode=mode, **kwargs)
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


class SW(Calculator):
    """Calculator for the Stillinger-Weber potential."""

    implemented_properties = (
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    )

    def __init__(
        self,
        sw_data: SWData,
        *args: tuple,
        engine: str = "numba",
        mode: str = "run",
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        engine_cls = make_sw_engine(engine)
        self.engine = engine_cls(sw_data, mode=mode)

    def update_parameters(self, sw_data: SWData) -> None:
        self.engine.update(sw_data)
        self.results = {}

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
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
        mode: str = "run",
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = GenericASEEngine(ase_data, mode=mode)

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
    "SW",
    "TersoffData",
    "make_calculator",
]
