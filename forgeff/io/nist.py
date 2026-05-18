"""Load NIST-style EAM and ADP potential files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.calculators.eam import EAM as ASEEAM

from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData


def _sample_grid(n: int, step: float) -> np.ndarray:
    return np.arange(n, dtype=float) * float(step)


def read_nist_potential(filename: str | Path) -> EAMData | ADPData:
    """Read a raw NIST EAM/ADP file into ForgeFF tabulated data."""
    calc = ASEEAM(potential=str(filename))
    r_grid = _sample_grid(calc.nr, calc.dr)
    rho_grid = _sample_grid(calc.nrho, calc.drho)
    species = np.array([int(z) for z in calc.Z], dtype=np.int32)
    spc = len(species)

    phi = np.zeros((spc, spc, calc.nr), dtype=float)
    rho = np.zeros((spc, spc, calc.nr), dtype=float)
    emb = np.zeros((spc, calc.nrho), dtype=float)

    for i in range(spc):
        emb[i] = np.asarray(calc.embedded_energy[i](rho_grid), dtype=float)
        if calc.form == "fs":
            for j in range(spc):
                rho[i, j] = np.asarray(calc.electron_density[i, j](r_grid), dtype=float)
        else:
            rho[i, i] = np.asarray(calc.electron_density[i](r_grid), dtype=float)
        for j in range(i, spc):
            values = np.asarray(calc.phi[i, j](r_grid), dtype=float)
            phi[i, j] = values
            phi[j, i] = values

    if calc.form == "adp":
        dipole = np.asarray(calc.d_data, dtype=float)
        quadrupole = np.asarray(calc.q_data, dtype=float)
        data = ADPData(
            potential_name=Path(filename).name,
            form="alloy",
            r_grid=r_grid,
            rho_grid=rho_grid,
            phi_values=phi,
            rho_values=rho,
            emb_values=emb,
            dipole_values=dipole,
            quadrupole_values=quadrupole,
        )
    else:
        data = EAMData(
            potential_name=Path(filename).name,
            form="alloy" if calc.form == "alloy" else "eam",
            r_grid=r_grid,
            rho_grid=rho_grid,
            phi_values=phi,
            rho_values=rho,
            emb_values=emb,
        )

    data.species = species
    return data
