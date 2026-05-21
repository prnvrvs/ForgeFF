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
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from ase.neighborlist import neighbor_list
from ase import Atoms
from forgeff.potentials.eam.data import EAMData

@numba.njit(cache=False, inline="always")
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


@numba.njit(cache=False, inline="always")
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


@numba.njit(cache=False, inline="always")
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


@numba.njit(cache=False, inline="always")
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


@numba.njit(cache=False)
def _calculate_eam_alloy(types, i_list, j_list, dist, rvec,
                        emb_coeffs, dens_coeffs, phi_coeffs,
                        drho, dr, rho_start, r_start):
    natoms = types.shape[0]
    total_density = np.zeros(natoms)
    site_energies = np.zeros(natoms)
    pair_energy_sum = 0.0

    for k in range(dist.shape[0]):
        i = i_list[k]
        j = j_list[k]
        r = dist[k]
        if r <= 0.0:
            continue
        ti = types[i]
        tj = types[j]
        pair_energy = _spline_eval_2d(phi_coeffs, r, r_start, dr, ti, tj)
        pair_energy_sum += pair_energy
        if i < j:
            site_energies[i] += 0.5 * pair_energy
            site_energies[j] += 0.5 * pair_energy
        total_density[i] += _spline_eval_1d(dens_coeffs, r, r_start, dr, tj)

    pair_energy = 0.5 * pair_energy_sum
    embedding_energy = 0.0
    d_emb = np.zeros(natoms)
    for i in range(natoms):
        ti = types[i]
        emb_i = _spline_eval_1d(emb_coeffs, total_density[i], rho_start, drho, ti)
        d_emb[i] = _spline_deriv_1d(emb_coeffs, total_density[i], rho_start, drho, ti)
        embedding_energy += emb_i
        site_energies[i] += emb_i

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

    return pair_energy, embedding_energy, total_density, site_energies, forces, stresses


@numba.njit(cache=False)
def _calculate_eam_fs(types, i_list, j_list, dist, rvec,
                     emb_coeffs, dens_coeffs, phi_coeffs,
                     drho, dr, rho_start, r_start):
    natoms = types.shape[0]
    total_density = np.zeros(natoms)
    site_energies = np.zeros(natoms)
    pair_energy_sum = 0.0

    for k in range(dist.shape[0]):
        i = i_list[k]
        j = j_list[k]
        r = dist[k]
        if r <= 0.0:
            continue
        ti = types[i]
        tj = types[j]
        pair_energy = _spline_eval_2d(phi_coeffs, r, r_start, dr, ti, tj)
        pair_energy_sum += pair_energy
        if i < j:
            site_energies[i] += 0.5 * pair_energy
            site_energies[j] += 0.5 * pair_energy
        total_density[i] += _spline_eval_2d(dens_coeffs, r, r_start, dr, tj, ti)

    pair_energy = 0.5 * pair_energy_sum
    embedding_energy = 0.0
    d_emb = np.zeros(natoms)
    for i in range(natoms):
        ti = types[i]
        emb_i = _spline_eval_1d(emb_coeffs, total_density[i], rho_start, drho, ti)
        d_emb[i] = _spline_deriv_1d(emb_coeffs, total_density[i], rho_start, drho, ti)
        embedding_energy += emb_i
        site_energies[i] += emb_i

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
                 d_emb[i] * _spline_deriv_2d(dens_coeffs, r, r_start, dr, tj, ti) +\
                 d_emb[j] * _spline_deriv_2d(dens_coeffs, r, r_start, dr, ti, tj))
            
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

    return pair_energy, embedding_energy, total_density, site_energies, forces, stresses


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

    def _finite_difference_response(self, atoms: Atoms, delta: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return numerical derivatives of energy, forces, and stress.

        The EAM Numba engine still does not expose closed-form Jacobians for
        forces or stress, so we keep the finite-difference fallback local to
        the engine.
        """
        orig_params = np.asarray(self.eam_data.parameters, dtype=float).copy()
        nprm = orig_params.size
        natoms = len(atoms)
        d_energy = np.zeros(nprm, dtype=float)
        d_forces = np.zeros((nprm, natoms, 3), dtype=float)
        d_stress = np.zeros((nprm, 3, 3), dtype=float)

        try:
            for i in range(nprm):
                p_plus = orig_params.copy()
                p_plus[i] += delta
                self.eam_data.parameters = p_plus
                self.update(self.eam_data)
                plus = self.calculate(atoms)

                p_minus = orig_params.copy()
                p_minus[i] -= delta
                self.eam_data.parameters = p_minus
                self.update(self.eam_data)
                minus = self.calculate(atoms)

                scale = 1.0 / (2.0 * delta)
                d_energy[i] = (plus["energy"] - minus["energy"]) * scale
                d_forces[i] = (plus["forces"] - minus["forces"]) * scale
                if "stress" in plus and "stress" in minus:
                    plus_stress = np.asarray(plus["stress"], dtype=float)
                    minus_stress = np.asarray(minus["stress"], dtype=float)
                    if plus_stress.shape == (6,):
                        plus_stress = voigt_6_to_full_3x3_stress(plus_stress)
                    if minus_stress.shape == (6,):
                        minus_stress = voigt_6_to_full_3x3_stress(minus_stress)
                    d_stress[i] = (plus_stress - minus_stress) * scale
        finally:
            self.eam_data.parameters = orig_params
            self.update(self.eam_data)

        return d_energy, d_forces, d_stress

    def jac_energy(self, atoms: Atoms) -> SimpleNamespace:
        """Numerical Jacobian for energy."""
        jac, _, _ = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=jac)

    def jac_forces(self, atoms: Atoms) -> SimpleNamespace:
        """Numerical Jacobian for forces."""
        _, jac, _ = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=jac)

    def jac_stress(self, atoms: Atoms) -> SimpleNamespace:
        """Numerical Jacobian for stress."""
        _, _, jac = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=jac)

    def calculate(self, atoms: Atoms) -> dict:
        symbols = atoms.get_chemical_symbols()
        species = self.eam_data.species.tolist()
        types = np.array([species.index(atoms.numbers[i]) for i in range(len(atoms))], dtype=np.int64)

        cutoff = self.r[-1]
        i_list, j_list, shifts, dist = neighbor_list('ijSd', atoms, cutoff)
        rvec = atoms.positions[j_list] + shifts @ atoms.cell.array - atoms.positions[i_list]

        if self.form == "fs":
            pair_energy, embedding_energy, total_density, site_energies, forces, stresses = _calculate_eam_fs(
                types, i_list.astype(np.int64), j_list.astype(np.int64),
                dist.astype(np.float64), rvec.astype(np.float64),
                self._emb_coeffs, self._dens_coeffs, self._phi_coeffs,
                float(self.drho), float(self.dr), float(self.rho[0]), float(self.r[0]),
            )
        else:
            pair_energy, embedding_energy, total_density, site_energies, forces, stresses = _calculate_eam_alloy(
                types, i_list.astype(np.int64), j_list.astype(np.int64),
                dist.astype(np.float64), rvec.astype(np.float64),
                self._emb_coeffs, self._dens_coeffs, self._phi_coeffs,
                float(self.drho), float(self.dr), float(self.rho[0]), float(self.r[0]),
            )
        
        energy = float(np.sum(site_energies))
        
        results = {
            "energy": energy,
            "energies": site_energies,
            "forces": forces,
        }
        
        if atoms.cell.rank == 3:
            stress_tensor = 0.5 * np.sum(stresses, axis=0) / atoms.get_volume()
            results["stress"] = full_3x3_to_voigt_6_stress(stress_tensor)
            
        return results
