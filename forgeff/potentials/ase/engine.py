"""Generic ASE Engine for forgeff."""

import importlib
import numpy as np
from ase import Atoms
from forgeff.potentials.ase.data import ASEData

class GenericASEEngine:
    """Engine that wraps any ASE calculator."""

    def __init__(self, ase_data: ASEData, mode: str = "run"):
        self.ase_data = ase_data
        self.mode = mode
        self.calculator = None
        self._init_calculator()

    def _init_calculator(self):
        """Instantiate the ASE calculator."""
        name = self.ase_data.engine
        
        # Try some common mappings
        mappings = {
            "LennardJones": ("ase.calculators.lj", "LennardJones"),
            "EAM": ("ase.calculators.eam", "EAM"),
            "EMT": ("ase.calculators.emt", "EMT"),
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

        # Merge static kwargs with optimized parameters
        params = self.ase_data.get_parameter_dict()
        # Handle 1-element arrays: convert to scalar for ASE calculators that expect floats
        for k, v in params.items():
            if isinstance(v, np.ndarray) and v.size == 1:
                params[k] = float(v.item())

        kwargs = {**self.ase_data.calculator_kwargs, **params}
        
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

        return {
            "energy": energy,
            "energies": np.array([energy / len(atoms)] * len(atoms)),
            "forces": forces,
            "stress": stress
        }

    def jac_energy(self, atoms: Atoms):
        """Jacobian (placeholder)."""
        nprm = self.ase_data.number_of_parameters_optimized
        return type('JacobianShim', (), {'parameters': np.zeros(nprm)})

    def jac_forces(self, atoms: Atoms):
        nprm = self.ase_data.number_of_parameters_optimized
        return type('JacobianShim', (), {'parameters': np.zeros((nprm, len(atoms), 3))})

    def jac_stress(self, atoms: Atoms):
        nprm = self.ase_data.number_of_parameters_optimized
        return type('JacobianShim', (), {'parameters': np.zeros((nprm, 3, 3))})
