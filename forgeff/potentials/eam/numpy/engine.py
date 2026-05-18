"""ASE EAM Calculator Engine for forgeff."""

import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.data import chemical_symbols
from scipy.interpolate import InterpolatedUnivariateSpline as spline

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
        """Numerical Jacobian for energy (fallback).
        
        For semi-empirical fitting, analytical Jacobians are preferred.
        """
        # TODO: Implement analytical Jacobian if possible, 
        # or use finite differences here.
        raise NotImplementedError("Analytical Jacobian for ASE EAM engine not yet implemented.")
