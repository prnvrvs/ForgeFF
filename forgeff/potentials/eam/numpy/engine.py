"""ASE EAM Calculator Engine for forgeff."""

import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.data import chemical_symbols
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from types import SimpleNamespace

from forgeff.potentials.eam.data import EAMData

class ASEAMEngine:
    """Engine that wraps ASE's EAM calculator for fitting."""

    def __init__(self, eam_data: EAMData, mode: str = "run"):
        self.eam_data = eam_data
        self.mode = mode
        self.calculator = None
        self._init_calculator()

    def _init_calculator(self):
        """Initialize the ASE EAM calculator using splines from eam_data."""
        eam_data = self.eam_data
        spc = eam_data.species_count
        elements = [
            s if isinstance(s, str) else chemical_symbols[int(s)]
            for s in eam_data.species
        ]
        
        embedded_energy = []
        d_embedded_energy = []
        electron_density = []
        d_electron_density = []
        phi = np.empty((spc, spc), dtype=object)
        d_phi = np.empty((spc, spc), dtype=object)
        
        # Alloy form: rho depends only on neighbor species
        # FS form: rho depends on both species
        form = getattr(eam_data, "form", "alloy")

        for i in range(spc):
            # Embedding energy F(rho)
            f_spline = spline(eam_data.rho_grid, eam_data.emb_values[i], k=3)
            embedded_energy.append(f_spline)
            d_embedded_energy.append(f_spline.derivative())
            
            if form == "alloy":
                # In Alloy form, ASE expects a 1D list of electron density functions
                # where rho[j] is the density contribution of atom j.
                # We use the diagonal rho_values[i, i] as the representative rho for species i.
                rho_spline = spline(eam_data.r_grid, eam_data.rho_values[i, i], k=3)
                electron_density.append(rho_spline)
                d_electron_density.append(rho_spline.derivative())
            else:
                # FS form: rho is a 2D matrix of functions
                row = []
                d_row = []
                for j in range(spc):
                    rho_spline = spline(eam_data.r_grid, eam_data.rho_values[i, j], k=3)
                    row.append(rho_spline)
                    d_row.append(rho_spline.derivative())
                electron_density.append(row)
                d_electron_density.append(d_row)
            
            for j in range(i, spc):
                # Pair potential phi(r)
                phi_spline = spline(eam_data.r_grid, eam_data.phi_values[i, j], k=3)
                phi[i, j] = phi[j, i] = phi_spline
                dphi_spline = phi_spline.derivative()
                d_phi[i, j] = d_phi[j, i] = dphi_spline

        self.calculator = EAM(
            elements=elements,
            embedded_energy=embedded_energy,
            d_embedded_energy=d_embedded_energy,
            electron_density=electron_density,
            d_electron_density=d_electron_density,
            phi=phi,
            d_phi=d_phi,
            form=form,
            cutoff=float(eam_data.r_grid[-1]),
        )

    def update(self, eam_data: EAMData):
        """Update the underlying ASE calculator with new parameters."""
        self.eam_data = eam_data
        self._init_calculator()

    def calculate(self, atoms: Atoms) -> dict:
        """Perform the calculation using ASE EAM."""
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        
        return {
            "energy": energy,
            "energies": np.array([energy / len(atoms)] * len(atoms)), # Approximation if local energies not available
            "forces": forces,
            "stress": stress
        }
    
    def jac_energy(self, atoms: Atoms):
        """Numerical Jacobian for energy.

        The NumPy-backed EAM engine does not expose an analytical Jacobian,
        so we use a symmetric finite-difference fallback over the serialized
        parameter vector.
        """
        dx = 1e-6
        orig_params = np.asarray(self.eam_data.parameters, dtype=float).copy()
        jac = np.zeros_like(orig_params)

        try:
            for i in range(orig_params.size):
                p_plus = orig_params.copy()
                p_plus[i] += dx
                self.eam_data.parameters = p_plus
                self.update(self.eam_data)
                e_plus = self.calculate(atoms)["energy"]

                p_minus = orig_params.copy()
                p_minus[i] -= dx
                self.eam_data.parameters = p_minus
                self.update(self.eam_data)
                e_minus = self.calculate(atoms)["energy"]

                jac[i] = (e_plus - e_minus) / (2.0 * dx)
        finally:
            self.eam_data.parameters = orig_params
            self.update(self.eam_data)

        return SimpleNamespace(parameters=jac)
