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
        mu[i, 0] += u * rvec[k, 0] / r
        mu[i, 1] += u * rvec[k, 1] / r
        mu[i, 2] += u * rvec[k, 2] / r
            
        # ADP Quadrupole
        w = _spline_eval_2d(quadrupole_coeffs, r, r_start, dr, ti, tj)
        w_r2 = w / (r * r)
        for alpha in range(3):
            for beta in range(3):
                nu[i, alpha, beta] += w_r2 * rvec[k, alpha] * rvec[k, beta]

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
        
        # --- Dipole Part ---
        # Fi_dip = sum_alpha (mu_i_alpha * D_alpha - mu_j_alpha * D_rev_alpha)
        u = _spline_eval_2d(dipole_coeffs, r, r_start, dr, ti, tj)
        du = _spline_deriv_2d(dipole_coeffs, r, r_start, dr, ti, tj)
        u_rev = _spline_eval_2d(dipole_coeffs, r, r_start, dr, tj, ti)
        du_rev = _spline_deriv_2d(dipole_coeffs, r, r_start, dr, tj, ti)
        
        mu_i_dot_r = mu[i, 0]*rvec[k, 0] + mu[i, 1]*rvec[k, 1] + mu[i, 2]*rvec[k, 2]
        mu_j_dot_r = mu[j, 0]*rvec[k, 0] + mu[j, 1]*rvec[k, 1] + mu[j, 2]*rvec[k, 2]
        
        f_dip_x = (mu_i_dot_r * (du - u/r) * rvec[k, 0]/(r*r) + (u/r)*mu[i, 0]) - \
                  (mu_j_dot_r * (du_rev - u_rev/r) * rvec[k, 0]/(r*r) + (u_rev/r)*mu[j, 0])
        f_dip_y = (mu_i_dot_r * (du - u/r) * rvec[k, 1]/(r*r) + (u/r)*mu[i, 1]) - \
                  (mu_j_dot_r * (du_rev - u_rev/r) * rvec[k, 1]/(r*r) + (u_rev/r)*mu[j, 1])
        f_dip_z = (mu_i_dot_r * (du - u/r) * rvec[k, 2]/(r*r) + (u/r)*mu[i, 2]) - \
                  (mu_j_dot_r * (du_rev - u_rev/r) * rvec[k, 2]/(r*r) + (u_rev/r)*mu[j, 2])

        # --- Quadrupole Part ---
        # Fi_quad = sum_alpha,beta (val_i_alpha_beta * H_alpha_beta + val_j_alpha_beta * H_rev_alpha_beta)
        w = _spline_eval_2d(quadrupole_coeffs, r, r_start, dr, ti, tj)
        dw = _spline_deriv_2d(quadrupole_coeffs, r, r_start, dr, ti, tj)
        w_rev = _spline_eval_2d(quadrupole_coeffs, r, r_start, dr, tj, ti)
        dw_rev = _spline_deriv_2d(quadrupole_coeffs, r, r_start, dr, tj, ti)
        
        t_nu_i = nu[i, 0, 0] + nu[i, 1, 1] + nu[i, 2, 2]
        t_nu_j = nu[j, 0, 0] + nu[j, 1, 1] + nu[j, 2, 2]
        
        dw_r2_i = (dw - 2.0*w/r) / (r*r)
        dw_r2_j = (dw_rev - 2.0*w_rev/r) / (r*r)
        
        sum_i = np.zeros(3)
        sum_j = np.zeros(3)
        for alpha in range(3):
            for beta in range(3):
                # val = nu - 1/3 Tr(nu) delta
                val_i = nu[i, alpha, beta]
                if alpha == beta: val_i -= (1.0/3.0) * t_nu_i
                
                # H_alpha_beta_gamma = (w' - 2w/r) * r_alpha*r_beta*r_gamma/r^3 + w/r^2 * (delta_alpha_gamma*r_beta + delta_beta_gamma*r_alpha)
                common_i = dw_r2_i * rvec[k, alpha] * rvec[k, beta] / r
                sum_i[0] += val_i * (common_i * rvec[k, 0] + (w/(r*r)) * ((1.0 if alpha==0 else 0.0)*rvec[k, beta] + (1.0 if beta==0 else 0.0)*rvec[k, alpha]))
                sum_i[1] += val_i * (common_i * rvec[k, 1] + (w/(r*r)) * ((1.0 if alpha==1 else 0.0)*rvec[k, beta] + (1.0 if beta==1 else 0.0)*rvec[k, alpha]))
                sum_i[2] += val_i * (common_i * rvec[k, 2] + (w/(r*r)) * ((1.0 if alpha==2 else 0.0)*rvec[k, beta] + (1.0 if beta==2 else 0.0)*rvec[k, alpha]))
                
                val_j = nu[j, alpha, beta]
                if alpha == beta: val_j -= (1.0/3.0) * t_nu_j
                
                common_j = dw_r2_j * rvec[k, alpha] * rvec[k, beta] / r
                sum_j[0] += val_j * (common_j * rvec[k, 0] + (w_rev/(r*r)) * ((1.0 if alpha==0 else 0.0)*rvec[k, beta] + (1.0 if beta==0 else 0.0)*rvec[k, alpha]))
                sum_j[1] += val_j * (common_j * rvec[k, 1] + (w_rev/(r*r)) * ((1.0 if alpha==1 else 0.0)*rvec[k, beta] + (1.0 if beta==1 else 0.0)*rvec[k, alpha]))
                sum_j[2] += val_j * (common_j * rvec[k, 2] + (w_rev/(r*r)) * ((1.0 if alpha==2 else 0.0)*rvec[k, beta] + (1.0 if beta==2 else 0.0)*rvec[k, alpha]))

        # Total force on atom i from this pair interaction
        fx = (scale_eam * rvec[k, 0] / r + f_dip_x + sum_i[0] + sum_j[0])
        fy = (scale_eam * rvec[k, 1] / r + f_dip_y + sum_i[1] + sum_j[1])
        fz = (scale_eam * rvec[k, 2] / r + f_dip_z + sum_i[2] + sum_j[2])
        
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
