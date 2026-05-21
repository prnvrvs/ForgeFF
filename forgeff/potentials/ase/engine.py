"""ASE engine adapter for ForgeFF."""

import importlib
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress

from forgeff.potentials.ase.data import ASEData

class GenericASEEngine:
    """Engine that wraps any ASE calculator."""

    _DISALLOWED_CALCULATORS = {"EMT"}

    def __init__(self, ase_data: ASEData, mode: str = "run"):
        self.ase_data = ase_data
        self.mode = mode
        self.calculator = None
        self._init_calculator()

    def _init_calculator(self):
        """Instantiate the ASE calculator."""
        name = self.ase_data.calculator_kwargs.get("calculator", self.ase_data.engine)

        # Try some common mappings.
        mappings = {
            "LennardJones": ("ase.calculators.lj", "LennardJones"),
            "EAM": ("ase.calculators.eam", "EAM"),
            "Morse": ("ase.calculators.morse", "MorsePotential"),
            "MorsePotential": ("ase.calculators.morse", "MorsePotential"),
            "CustomPairPotential": ("forgeff.potentials.ase.custom", "CustomPairPotential"),
            "numpy": ("forgeff.potentials.ase.custom", "CustomPairPotential"),
            "NumbaPairPotential": ("forgeff.potentials.ase.numba_pair", "NumbaPairPotential"),
            "numba": ("forgeff.potentials.ase.numba_pair", "NumbaPairPotential"),
        }

        module_path = mappings.get(name)
        if module_path:
            module, class_name = module_path
            module = importlib.import_module(module)
            calc_class = getattr(module, class_name)
        else:
            # Try full path if provided in name (e.g., 'ase.calculators.lj.LennardJones')
            if "." in name:
                module_path, class_name = name.rsplit(".", 1)
                module = importlib.import_module(module_path)
                calc_class = getattr(module, class_name)
            else:
                # Fallback: try to import from ase.calculators.lowercase_name
                try:
                    module = importlib.import_module("ase.calculators." + name.lower())
                    calc_class = getattr(module, name)
                except (ImportError, AttributeError):
                    raise ImportError(f"Could not locate ASE calculator '{name}'. "
                                    f"Please provide full path or add to mappings.")
        if calc_class.__name__ in self._DISALLOWED_CALCULATORS:
            raise ValueError(
                f"ASE calculator '{calc_class.__name__}' is not supported as a fitting target in ForgeFF."
            )

        # Merge static kwargs with optimized parameters.
        params = self.ase_data.get_parameter_dict()
        if calc_class.__name__ == "MorsePotential":
            if {"De", "a", "re"}.issubset(params):
                params = {
                    "epsilon": params["De"],
                    "rho0": params["a"] * params["re"],
                    "r0": params["re"],
                    **{k: v for k, v in self.ase_data.calculator_kwargs.items() if k in {"rcut1", "rcut2"}},
                }
        elif calc_class.__name__ == "LennardJones":
            # ASE and ForgeFF use the same parameter names for LJ.
            pass
        # Handle 1-element arrays: convert to scalar for ASE calculators that expect floats
        for k, v in params.items():
            if isinstance(v, np.ndarray) and v.size == 1:
                params[k] = float(v.item())

        kwargs = {**self.ase_data.calculator_kwargs, **params}
        kwargs.pop("calculator", None)
        if calc_class.__module__.startswith("ase.calculators."):
            kwargs.pop("form", None)
            kwargs.pop("expression", None)
            kwargs.pop("parameter_names", None)
            kwargs.pop("variable", None)
            kwargs.pop("cutoff", None)
        
        self.calculator = calc_class(**kwargs)

    def update(self, ase_data: ASEData):
        """Update parameters."""
        self.ase_data = ase_data
        self._init_calculator()

    def calculate(self, atoms: Atoms) -> dict:
        """Calculate properties."""
        # Save old calculator to restore later (avoid recursion/leaks)
        old_calc = atoms.calc
        atoms.calc = self.calculator
        try:
            # results might need mapping to forgeff names
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            try:
                stress = atoms.get_stress()
            except Exception:
                stress = None
        finally:
            atoms.calc = old_calc

        if len(atoms) == 0:
            site_energies = np.zeros(0, dtype=float)
        else:
            site_energies = np.full(len(atoms), energy / len(atoms), dtype=float)

        return {
            "energy": energy,
            "energies": site_energies,
            "forces": forces,
            "stress": stress
        }

    def _finite_difference_response(
        self,
        atoms: Atoms,
        delta: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return numerical derivatives for ASE calculators."""
        orig_params = np.asarray(self.ase_data.parameters, dtype=float).copy()
        nprm = orig_params.size
        natoms = len(atoms)
        d_energy = np.zeros(nprm, dtype=float)
        d_energies = np.zeros((nprm, natoms), dtype=float)
        d_forces = np.zeros((nprm, natoms, 3), dtype=float)
        d_stress = np.zeros((nprm, 3, 3), dtype=float)

        try:
            for idx in range(nprm):
                p_plus = orig_params.copy()
                p_plus[idx] += delta
                self.ase_data.parameters = p_plus
                self.update(self.ase_data)
                plus = self.calculate(atoms)

                p_minus = orig_params.copy()
                p_minus[idx] -= delta
                self.ase_data.parameters = p_minus
                self.update(self.ase_data)
                minus = self.calculate(atoms)

                scale = 1.0 / (2.0 * delta)
                d_energy[idx] = (plus["energy"] - minus["energy"]) * scale
                d_energies[idx] = (plus["energies"] - minus["energies"]) * scale
                d_forces[idx] = (plus["forces"] - minus["forces"]) * scale
                if plus.get("stress") is not None and minus.get("stress") is not None:
                    plus_stress = np.asarray(plus["stress"], dtype=float)
                    minus_stress = np.asarray(minus["stress"], dtype=float)
                    if plus_stress.shape == (6,):
                        plus_stress = voigt_6_to_full_3x3_stress(plus_stress)
                    if minus_stress.shape == (6,):
                        minus_stress = voigt_6_to_full_3x3_stress(minus_stress)
                    d_stress[idx] = (plus_stress - minus_stress) * scale
        finally:
            self.ase_data.parameters = orig_params
            self.update(self.ase_data)

        return d_energy, d_energies, d_forces, d_stress

    def jac_energy(self, atoms: Atoms):
        d_energy, _, _, _ = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=d_energy)

    def jac_energies(self, atoms: Atoms):
        """Site-energy Jacobians are not meaningful for generic ASE adapters.

        The adapter only provides a uniform average per-atom energy for API
        compatibility. That is not a physical site-energy decomposition, so
        exposing a Jacobian here would mislead neighborhood grading.
        """
        raise NotImplementedError(
            "GenericASEEngine does not provide site-energy Jacobians because its "
            "'energies' output is only a uniform average, not a physical site-energy decomposition."
        )

    def jac_forces(self, atoms: Atoms):
        _, _, d_forces, _ = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=d_forces)

    def jac_stress(self, atoms: Atoms):
        _, _, _, d_stress = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=d_stress)
