"""JIT-backed ADP calculator for forgeff."""

import numba
import numpy as np
from scipy.interpolate import CubicSpline

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.neighborlist import neighbor_list
from ase import Atoms
from forgeff.potentials.eam.adp_data import ADPData
from .engine import _spline_eval_1d, _spline_eval_2d, _spline_deriv_1d, _spline_deriv_2d

@numba.njit(cache=True)
def _calculate_adp(types, i_list, j_list, dist, rvec,
                  emb_coeffs, dens_coeffs, phi_coeffs,
                  dipole_coeffs, quadrupole_coeffs,
                  drho, dr, rho_start, r_start):
    natoms = types.shape[0]
    total_density = np.zeros(natoms)
    pair_energy_sum = 0.0
    
    # Dipole and Quadrupole accumulators
    mu = np.zeros((natoms, 3))
    nu = np.zeros((natoms, 3, 3))

    for k in range(dist.shape[0]):
        i = i_list[k]
        j = j_list[k]
        r = dist[k]
        if r <= 0.0:
            continue
        ti = types[i]
        tj = types[j]
        
        # EAM Pair
        pair_energy_sum += _spline_eval_2d(phi_coeffs, r, r_start, dr, ti, tj)
        # EAM Density
        total_density[i] += _spline_eval_1d(dens_coeffs, r, r_start, dr, tj)
        
        # ADP Dipole
        u = _spline_eval_2d(dipole_coeffs, r, r_start, dr, ti, tj)
        mu[i, 0] += u * rvec[k, 0]
        mu[i, 1] += u * rvec[k, 1]
        mu[i, 2] += u * rvec[k, 2]

        # ADP Quadrupole
        w = _spline_eval_2d(quadrupole_coeffs, r, r_start, dr, ti, tj)
        for alpha in range(3):
            for beta in range(3):
                nu[i, alpha, beta] += w * rvec[k, alpha] * rvec[k, beta]

    pair_energy = 0.5 * pair_energy_sum
    embedding_energy = 0.0
    d_emb = np.zeros(natoms)
    for i in range(natoms):
        ti = types[i]
        d_emb[i] = _spline_deriv_1d(emb_coeffs, total_density[i], rho_start, drho, ti)
        embedding_energy += _spline_eval_1d(emb_coeffs, total_density[i], rho_start, drho, ti)

    # ADP Energy
    dipole_energy = 0.0
    quad_energy = 0.0
    for i in range(natoms):
        # Dipole: 0.5 * sum(mu_alpha^2)
        for alpha in range(3):
            dipole_energy += 0.5 * mu[i, alpha]**2
        
        # Quadrupole: 0.5 * sum(nu_alpha_beta^2) - 1/6 * (sum nu_alpha_alpha)^2
        t_nu = nu[i, 0, 0] + nu[i, 1, 1] + nu[i, 2, 2]
        for alpha in range(3):
            for beta in range(3):
                quad_energy += 0.5 * nu[i, alpha, beta]**2
        quad_energy -= (1.0/6.0) * t_nu**2

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
        
        # --- EAM Part ---
        # Fi_eam = (phi' + Fi' rho' + Fj' rho') * r_ij / r
        scale_eam = (_spline_deriv_2d(phi_coeffs, r, r_start, dr, ti, tj) +\
                     d_emb[i] * _spline_deriv_1d(dens_coeffs, r, r_start, dr, tj) +\
                     d_emb[j] * _spline_deriv_1d(dens_coeffs, r, r_start, dr, ti))
        
        # --- Dipole and Quadrupole Part ---
        # Mirror the ASE angular force expression for a single neighbor pair.
        dip = _spline_eval_2d(dipole_coeffs, r, r_start, dr, ti, tj)
        ddip = _spline_deriv_2d(dipole_coeffs, r, r_start, dr, ti, tj)
        quad = _spline_eval_2d(quadrupole_coeffs, r, r_start, dr, ti, tj)
        dquad = _spline_deriv_2d(quadrupole_coeffs, r, r_start, dr, ti, tj)

        mu_diff_x = mu[i, 0] - mu[j, 0]
        mu_diff_y = mu[i, 1] - mu[j, 1]
        mu_diff_z = mu[i, 2] - mu[j, 2]
        mu_diff_dot_r = mu_diff_x * rvec[k, 0] + mu_diff_y * rvec[k, 1] + mu_diff_z * rvec[k, 2]
        trace_i = nu[i, 0, 0] + nu[i, 1, 1] + nu[i, 2, 2]
        trace_j = nu[j, 0, 0] + nu[j, 1, 1] + nu[j, 2, 2]

        f_adp_x = (mu_diff_x * dip) + (mu_diff_dot_r * ddip * rvec[k, 0] / r)
        f_adp_y = (mu_diff_y * dip) + (mu_diff_dot_r * ddip * rvec[k, 1] / r)
        f_adp_z = (mu_diff_z * dip) + (mu_diff_dot_r * ddip * rvec[k, 2] / r)

        term3_x = 0.0
        term3_y = 0.0
        term3_z = 0.0
        term4_x = 0.0
        term4_y = 0.0
        term4_z = 0.0
        for alpha in range(3):
            term3_x += (nu[i, alpha, 0] + nu[j, alpha, 0]) * rvec[k, alpha]
            term3_y += (nu[i, alpha, 1] + nu[j, alpha, 1]) * rvec[k, alpha]
            term3_z += (nu[i, alpha, 2] + nu[j, alpha, 2]) * rvec[k, alpha]
            for beta in range(3):
                rs = rvec[k, alpha] * rvec[k, beta]
                term4_x += (nu[i, alpha, beta] + nu[j, alpha, beta]) * dquad * rs * rvec[k, 0] / r
                term4_y += (nu[i, alpha, beta] + nu[j, alpha, beta]) * dquad * rs * rvec[k, 1] / r
                term4_z += (nu[i, alpha, beta] + nu[j, alpha, beta]) * dquad * rs * rvec[k, 2] / r

        term3_x *= 2.0 * quad
        term3_y *= 2.0 * quad
        term3_z *= 2.0 * quad

        term5_x = (trace_i + trace_j) * (dquad * r + 2.0 * quad) * rvec[k, 0] / 3.0
        term5_y = (trace_i + trace_j) * (dquad * r + 2.0 * quad) * rvec[k, 1] / 3.0
        term5_z = (trace_i + trace_j) * (dquad * r + 2.0 * quad) * rvec[k, 2] / 3.0

        f_adp_x = f_adp_x + term3_x + term4_x - term5_x
        f_adp_y = f_adp_y + term3_y + term4_y - term5_y
        f_adp_z = f_adp_z + term3_z + term4_z - term5_z

        w = _spline_eval_2d(quadrupole_coeffs, r, r_start, dr, ti, tj)
        # Total force on atom i from this pair interaction
        fx = scale_eam * rvec[k, 0] / r + f_adp_x
        fy = scale_eam * rvec[k, 1] / r + f_adp_y
        fz = scale_eam * rvec[k, 2] / r + f_adp_z
        
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[i, 2] += fz
        
        # Stress tensor: 0.5 * sum_pairs F_ij \otimes r_ij
        stresses[i, 0, 0] += fx * rvec[k, 0]
        stresses[i, 0, 1] += fx * rvec[k, 1]
        stresses[i, 0, 2] += fx * rvec[k, 2]
        stresses[i, 1, 0] += fy * rvec[k, 0]
        stresses[i, 1, 1] += fy * rvec[k, 1]
        stresses[i, 1, 2] += fy * rvec[k, 2]
        stresses[i, 2, 0] += fz * rvec[k, 0]
        stresses[i, 2, 1] += fz * rvec[k, 1]
        stresses[i, 2, 2] += fz * rvec[k, 2]

    total_energy = pair_energy + embedding_energy + dipole_energy + quad_energy
    return total_energy, forces, stresses


class NumbaADPEngine:
    """Engine that uses Numba-accelerated ADP calculator."""

    def __init__(self, pot_data: ADPData, mode: str = "run"):
        self.pot_data = pot_data
        self.mode = mode
        self._build_splines()

    def _build_splines(self):
        pot_data = self.pot_data
        self.r = pot_data.r_grid
        self.rho = pot_data.rho_grid
        self.dr = self.r[1] - self.r[0]
        self.drho = self.rho[1] - self.rho[0]
        self.form = getattr(pot_data, "form", "alloy")
        if self.form == "fs":
            raise NotImplementedError(
                "ADP Finnis-Sinclair mode is not supported in the Numba engine yet."
            )

        self._emb_coeffs = np.ascontiguousarray(
            CubicSpline(self.rho, pot_data.emb_values, axis=-1).c
        )
        self._dens_coeffs = np.ascontiguousarray(
            CubicSpline(self.r, pot_data.rho_values.diagonal(axis1=0, axis2=1).T, axis=-1).c
        )
        self._phi_coeffs = np.ascontiguousarray(
            CubicSpline(self.r, pot_data.phi_values, axis=-1).c
        )
        self._dipole_coeffs = np.ascontiguousarray(
            CubicSpline(self.r, pot_data.dipole_values, axis=-1).c
        )
        self._quad_coeffs = np.ascontiguousarray(
            CubicSpline(self.r, pot_data.quadrupole_values, axis=-1).c
        )

    def update(self, pot_data: ADPData):
        self.pot_data = pot_data
        self._build_splines()

    def calculate(self, atoms: Atoms) -> dict:
        species = self.pot_data.species.tolist()
        types = np.array([species.index(atoms.numbers[i]) for i in range(len(atoms))], dtype=np.int64)

        cutoff = self.r[-1]
        i_list, j_list, shifts, dist = neighbor_list('ijSd', atoms, cutoff)
        rvec = atoms.positions[j_list] + shifts @ atoms.cell.array - atoms.positions[i_list]

        energy, forces, stresses = _calculate_adp(
            types, i_list.astype(np.int64), j_list.astype(np.int64),
            dist.astype(np.float64), rvec.astype(np.float64),
            self._emb_coeffs, self._dens_coeffs, self._phi_coeffs,
            self._dipole_coeffs, self._quad_coeffs,
            float(self.drho), float(self.dr), float(self.rho[0]), float(self.r[0])
        )
        
        results = {
            "energy": energy,
            "energies": np.array([energy / len(atoms)] * len(atoms)),
            "forces": forces,
        }
        
        if atoms.cell.rank == 3:
            stress_tensor = 0.5 * np.sum(stresses, axis=0) / atoms.get_volume()
            results["stress"] = full_3x3_to_voigt_6_stress(stress_tensor)
            
        return results
