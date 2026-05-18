"""JIT-backed EAM calculator for forgeff.

Adapted from ase.calculators.eam_jit.
"""

from pathlib import Path
import os
from types import SimpleNamespace

import numba
import numpy as np
from scipy.interpolate import CubicSpline

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.neighborlist import neighbor_list
from ase import Atoms
from forgeff.potentials.eam.data import EAMData

@numba.njit(cache=True, inline="always")
def _spline_eval_1d(coeffs, x, x0, h, idx0):
    nseg = coeffs.shape[1]
    idx = int((x - x0) / h)
    if idx < 0:
        idx = 0
    elif idx >= nseg:
        idx = nseg - 1
    dx = x - (x0 + idx * h)
    return (((coeffs[0, idx, idx0] * dx + coeffs[1, idx, idx0]) * dx
             + coeffs[2, idx, idx0]) * dx + coeffs[3, idx, idx0])


@numba.njit(cache=True, inline="always")
def _spline_eval_2d(coeffs, x, x0, h, idx0, idx1):
    nseg = coeffs.shape[1]
    idx = int((x - x0) / h)
    if idx < 0:
        idx = 0
    elif idx >= nseg:
        idx = nseg - 1
    dx = x - (x0 + idx * h)
    return (((coeffs[0, idx, idx0, idx1] * dx + coeffs[1, idx, idx0, idx1]) * dx
             + coeffs[2, idx, idx0, idx1]) * dx + coeffs[3, idx, idx0, idx1])


@numba.njit(cache=True, inline="always")
def _spline_deriv_1d(coeffs, x, x0, h, idx0):
    nseg = coeffs.shape[1]
    idx = int((x - x0) / h)
    if idx < 0:
        idx = 0
    elif idx >= nseg:
        idx = nseg - 1
    dx = x - (x0 + idx * h)
    return (3.0 * coeffs[0, idx, idx0] * dx * dx +\
            2.0 * coeffs[1, idx, idx0] * dx +\
            coeffs[2, idx, idx0])


@numba.njit(cache=True, inline="always")
def _spline_deriv_2d(coeffs, x, x0, h, idx0, idx1):
    nseg = coeffs.shape[1]
    idx = int((x - x0) / h)
    if idx < 0:
        idx = 0
    elif idx >= nseg:
        idx = nseg - 1
    dx = x - (x0 + idx * h)
    return (3.0 * coeffs[0, idx, idx0, idx1] * dx * dx +\
            2.0 * coeffs[1, idx, idx0, idx1] * dx +\
            coeffs[2, idx, idx0, idx1])


@numba.njit(cache=True)
def _calculate_eam_alloy(types, i_list, j_list, dist, rvec,
                        emb_coeffs, dens_coeffs, phi_coeffs,
                        drho, dr, rho_start, r_start):
    natoms = types.shape[0]
    total_density = np.zeros(natoms)
    pair_energy_sum = 0.0

    for k in range(dist.shape[0]):
        i = i_list[k]
        j = j_list[k]
        r = dist[k]
        if r <= 0.0:
            continue
        ti = types[i]
        tj = types[j]
        pair_energy_sum += _spline_eval_2d(phi_coeffs, r, r_start, dr, ti, tj)
        total_density[i] += _spline_eval_1d(dens_coeffs, r, r_start, dr, tj)

    pair_energy = 0.5 * pair_energy_sum
    embedding_energy = 0.0
    d_emb = np.zeros(natoms)
    for i in range(natoms):
        ti = types[i]
        d_emb[i] = _spline_deriv_1d(emb_coeffs, total_density[i], rho_start, drho, ti)
        embedding_energy += _spline_eval_1d(emb_coeffs, total_density[i], rho_start, drho, ti)

    forces = np.zeros((natoms, 3))
    stresses = np.zeros((natoms, 3, 3))
    for k in range(dist.shape[0]):
        i = i_list[k]
        j = j_list[k]
        r = dist[k]
        if r <= 0.0:
            continue
        ti = types[i]
        tj = types[j]
        
        scale = (_spline_deriv_2d(phi_coeffs, r, r_start, dr, ti, tj) +\
                 d_emb[i] * _spline_deriv_1d(dens_coeffs, r, r_start, dr, tj) +\
                 d_emb[j] * _spline_deriv_1d(dens_coeffs, r, r_start, dr, ti))
            
        fx = scale * rvec[k, 0] / r
        fy = scale * rvec[k, 1] / r
        fz = scale * rvec[k, 2] / r
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[i, 2] += fz
        stresses[i, 0, 0] += fx * rvec[k, 0]
        stresses[i, 0, 1] += fx * rvec[k, 1]
        stresses[i, 0, 2] += fx * rvec[k, 2]
        stresses[i, 1, 0] += fy * rvec[k, 0]
        stresses[i, 1, 1] += fy * rvec[k, 1]
        stresses[i, 1, 2] += fy * rvec[k, 2]
        stresses[i, 2, 0] += fz * rvec[k, 0]
        stresses[i, 2, 1] += fz * rvec[k, 1]
        stresses[i, 2, 2] += fz * rvec[k, 2]

    return pair_energy, embedding_energy, total_density, forces, stresses


@numba.njit(cache=True)
def _calculate_eam_fs(types, i_list, j_list, dist, rvec,
                     emb_coeffs, dens_coeffs, phi_coeffs,
                     drho, dr, rho_start, r_start):
    natoms = types.shape[0]
    total_density = np.zeros(natoms)
    pair_energy_sum = 0.0

    for k in range(dist.shape[0]):
        i = i_list[k]
        j = j_list[k]
        r = dist[k]
        if r <= 0.0:
            continue
        ti = types[i]
        tj = types[j]
        pair_energy_sum += _spline_eval_2d(phi_coeffs, r, r_start, dr, ti, tj)
        total_density[i] += _spline_eval_2d(dens_coeffs, r, r_start, dr, ti, tj)

    pair_energy = 0.5 * pair_energy_sum
    embedding_energy = 0.0
    d_emb = np.zeros(natoms)
    for i in range(natoms):
        ti = types[i]
        d_emb[i] = _spline_deriv_1d(emb_coeffs, total_density[i], rho_start, drho, ti)
        embedding_energy += _spline_eval_1d(emb_coeffs, total_density[i], rho_start, drho, ti)

    forces = np.zeros((natoms, 3))
    stresses = np.zeros((natoms, 3, 3))
    for k in range(dist.shape[0]):
        i = i_list[k]
        j = j_list[k]
        r = dist[k]
        if r <= 0.0:
            continue
        ti = types[i]
        tj = types[j]
        
        scale = (_spline_deriv_2d(phi_coeffs, r, r_start, dr, ti, tj) +\
                 d_emb[i] * _spline_deriv_2d(dens_coeffs, r, r_start, dr, ti, tj) +\
                 d_emb[j] * _spline_deriv_2d(dens_coeffs, r, r_start, dr, tj, ti))
            
        fx = scale * rvec[k, 0] / r
        fy = scale * rvec[k, 1] / r
        fz = scale * rvec[k, 2] / r
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[i, 2] += fz
        stresses[i, 0, 0] += fx * rvec[k, 0]
        stresses[i, 0, 1] += fx * rvec[k, 1]
        stresses[i, 0, 2] += fx * rvec[k, 2]
        stresses[i, 1, 0] += fy * rvec[k, 0]
        stresses[i, 1, 1] += fy * rvec[k, 1]
        stresses[i, 1, 2] += fy * rvec[k, 2]
        stresses[i, 2, 0] += fz * rvec[k, 0]
        stresses[i, 2, 1] += fz * rvec[k, 1]
        stresses[i, 2, 2] += fz * rvec[k, 2]

    return pair_energy, embedding_energy, total_density, forces, stresses


class NumbaEAMEngine:
    """Engine that uses Numba-accelerated EAM calculator."""

    def __init__(self, eam_data: EAMData, mode: str = "run"):
        self.eam_data = eam_data
        self.mode = mode
        self._build_splines()

    def _build_splines(self):
        eam_data = self.eam_data
        self.r = eam_data.r_grid
        self.rho = eam_data.rho_grid
        self.dr = self.r[1] - self.r[0]
        self.drho = self.rho[1] - self.rho[0]
        self.form = getattr(eam_data, "form", "alloy")

        # Spline coefficients for Numba
        # CubicSpline.c has shape (4, nseg, ...)
        self._emb_coeffs = np.ascontiguousarray(
            CubicSpline(self.rho, eam_data.emb_values, axis=-1).c
        )
        if self.form == "fs":
            self._dens_coeffs = np.ascontiguousarray(
                CubicSpline(self.r, eam_data.rho_values, axis=-1).c
            )
        else:
            self._dens_coeffs = np.ascontiguousarray(
                CubicSpline(self.r, eam_data.rho_values.diagonal(axis1=0, axis2=1).T, axis=-1).c
            )
            
        self._phi_coeffs = np.ascontiguousarray(
            CubicSpline(self.r, eam_data.phi_values, axis=-1).c
        )

    def update(self, eam_data: EAMData):
        self.eam_data = eam_data
        self._build_splines()

    def jac_energy(self, atoms: Atoms) -> SimpleNamespace:
        """Numerical Jacobian for energy."""
        dx = 1e-6
        params = self.eam_data.parameters
        jac = np.zeros(len(params))
        
        orig_params = params.copy()
        
        for i in range(len(params)):
            # Forward
            p_plus = orig_params.copy()
            p_plus[i] += dx
            self.eam_data.parameters = p_plus
            self.update(self.eam_data)
            e_plus = self.calculate(atoms)["energy"]
            
            # Backward
            p_minus = orig_params.copy()
            p_minus[i] -= dx
            self.eam_data.parameters = p_minus
            self.update(self.eam_data)
            e_minus = self.calculate(atoms)["energy"]
            
            jac[i] = (e_plus - e_minus) / (2.0 * dx)
            
        # Restore original parameters
        self.eam_data.parameters = orig_params
        self.update(self.eam_data)
        
        return SimpleNamespace(parameters=jac)

    def calculate(self, atoms: Atoms) -> dict:
        symbols = atoms.get_chemical_symbols()
        species = self.eam_data.species.tolist()
        types = np.array([species.index(atoms.numbers[i]) for i in range(len(atoms))], dtype=np.int64)

        cutoff = self.r[-1]
        i_list, j_list, shifts, dist = neighbor_list('ijSd', atoms, cutoff)
        rvec = atoms.positions[j_list] + shifts @ atoms.cell.array - atoms.positions[i_list]

        if self.form == "fs":
            pair_energy, embedding_energy, total_density, forces, stresses = _calculate_eam_fs(
                types, i_list.astype(np.int64), j_list.astype(np.int64),
                dist.astype(np.float64), rvec.astype(np.float64),
                self._emb_coeffs, self._dens_coeffs, self._phi_coeffs,
                float(self.drho), float(self.dr), float(self.rho[0]), float(self.r[0]),
            )
        else:
            pair_energy, embedding_energy, total_density, forces, stresses = _calculate_eam_alloy(
                types, i_list.astype(np.int64), j_list.astype(np.int64),
                dist.astype(np.float64), rvec.astype(np.float64),
                self._emb_coeffs, self._dens_coeffs, self._phi_coeffs,
                float(self.drho), float(self.dr), float(self.rho[0]), float(self.r[0]),
            )
        
        energy = pair_energy + embedding_energy
        
        results = {
            "energy": energy,
            "energies": np.array([energy / len(atoms)] * len(atoms)), # Approximation
            "forces": forces,
        }
        
        if atoms.cell.rank == 3:
            stress_tensor = 0.5 * np.sum(stresses, axis=0) / atoms.get_volume()
            results["stress"] = full_3x3_to_voigt_6_stress(stress_tensor)
            
        return results
