"""Numba-backed Stillinger-Weber calculator."""

from __future__ import annotations

import numba
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress

from .data import SWData


@numba.njit(cache=True)
def _pair_energy_and_dedr_numba(r, params):
    A = params[0]
    B = params[1]
    p = params[2]
    q = params[3]
    delta = params[4]
    a1 = params[5]
    if r <= 0.0 or r >= a1:
        return 0.0, 0.0
    exp_term = np.exp(delta / (r - a1))
    power = A * r ** (-p) - B * r ** (-q)
    pair_energy = power * exp_term
    d_power = -A * p * r ** (-p - 1.0) + B * q * r ** (-q - 1.0)
    d_pair = d_power * exp_term + power * exp_term * (-delta / (r - a1) ** 2)
    return pair_energy, d_pair


@numba.njit(cache=True)
def _triplet_energy_and_forces_numba(rij, rik, pair_ij, pair_ik, lambda_value, costheta0):
    gamma_ij = pair_ij[6]
    a2_ij = pair_ij[7]
    gamma_ik = pair_ik[6]
    a2_ik = pair_ik[7]

    u = np.sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2])
    v = np.sqrt(rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2])
    if u <= 0.0 or v <= 0.0 or u >= a2_ij or v >= a2_ik:
        return 0.0, np.zeros(3), np.zeros(3), np.zeros(3)

    u_hat = rij / u
    v_hat = rik / v
    c = u_hat[0] * v_hat[0] + u_hat[1] * v_hat[1] + u_hat[2] * v_hat[2]
    expo_u = np.exp(gamma_ij / (u - a2_ij))
    expo_v = np.exp(gamma_ik / (v - a2_ik))
    g = (c + costheta0) ** 2
    pref = lambda_value * expo_u * expo_v
    energy = pref * g

    d_pref_du = pref * (-gamma_ij / (u - a2_ij) ** 2)
    d_pref_dv = pref * (-gamma_ik / (v - a2_ik) ** 2)
    dE_du = d_pref_du * g
    dE_dv = d_pref_dv * g
    dE_dc = pref * 2.0 * (c + costheta0)

    dc_drij0 = (v_hat[0] - c * u_hat[0]) / u
    dc_drij1 = (v_hat[1] - c * u_hat[1]) / u
    dc_drij2 = (v_hat[2] - c * u_hat[2]) / u
    dc_drik0 = (u_hat[0] - c * v_hat[0]) / v
    dc_drik1 = (u_hat[1] - c * v_hat[1]) / v
    dc_drik2 = (u_hat[2] - c * v_hat[2]) / v

    grad_j = np.empty(3)
    grad_k = np.empty(3)
    force_j = np.empty(3)
    force_k = np.empty(3)
    force_i = np.empty(3)

    grad_j[0] = dE_du * u_hat[0] + dE_dc * dc_drij0
    grad_j[1] = dE_du * u_hat[1] + dE_dc * dc_drij1
    grad_j[2] = dE_du * u_hat[2] + dE_dc * dc_drij2

    grad_k[0] = dE_dv * v_hat[0] + dE_dc * dc_drik0
    grad_k[1] = dE_dv * v_hat[1] + dE_dc * dc_drik1
    grad_k[2] = dE_dv * v_hat[2] + dE_dc * dc_drik2

    force_j[0] = -grad_j[0]
    force_j[1] = -grad_j[1]
    force_j[2] = -grad_j[2]
    force_k[0] = -grad_k[0]
    force_k[1] = -grad_k[1]
    force_k[2] = -grad_k[2]
    force_i[0] = -(force_j[0] + force_k[0])
    force_i[1] = -(force_j[1] + force_k[1])
    force_i[2] = -(force_j[2] + force_k[2])
    return energy, force_i, force_j, force_k


@numba.njit(cache=True)
def _calculate_sw(
    i_p,
    j_p,
    r_p,
    r_pc,
    natoms,
    species_index,
    pair_parameters,
    lambda_values,
    costheta0,
):
    forces = np.zeros((natoms, 3))
    virial = np.zeros((3, 3))
    energy = 0.0
    n_neigh = i_p.shape[0]

    # Pair term
    for idx in range(n_neigh):
        i = int(i_p[idx])
        j = int(j_p[idx])
        if i >= j:
            continue
        ti = int(species_index[i])
        tj = int(species_index[j])
        pair_params = pair_parameters[ti, tj]
        pair_energy, d_pair = _pair_energy_and_dedr_numba(r_p[idx], pair_params)
        if pair_energy == 0.0:
            continue
        inv_r = 1.0 / r_p[idx]
        unit0 = r_pc[idx, 0] * inv_r
        unit1 = r_pc[idx, 1] * inv_r
        unit2 = r_pc[idx, 2] * inv_r
        f_j0 = -d_pair * unit0
        f_j1 = -d_pair * unit1
        f_j2 = -d_pair * unit2
        f_i0 = -f_j0
        f_i1 = -f_j1
        f_i2 = -f_j2
        energy += pair_energy
        forces[i, 0] += f_i0
        forces[i, 1] += f_i1
        forces[i, 2] += f_i2
        forces[j, 0] += f_j0
        forces[j, 1] += f_j1
        forces[j, 2] += f_j2
        virial[0, 0] += r_pc[idx, 0] * f_j0
        virial[0, 1] += r_pc[idx, 0] * f_j1
        virial[0, 2] += r_pc[idx, 0] * f_j2
        virial[1, 0] += r_pc[idx, 1] * f_j0
        virial[1, 1] += r_pc[idx, 1] * f_j1
        virial[1, 2] += r_pc[idx, 1] * f_j2
        virial[2, 0] += r_pc[idx, 2] * f_j0
        virial[2, 1] += r_pc[idx, 2] * f_j1
        virial[2, 2] += r_pc[idx, 2] * f_j2

    # Triplet term
    for i in range(natoms):
        count = 0
        for idx in range(n_neigh):
            if int(i_p[idx]) == i:
                count += 1
        if count < 2:
            continue

        neigh_j = np.empty(count, dtype=np.int64)
        neigh_r = np.empty(count, dtype=np.float64)
        neigh_vec = np.empty((count, 3), dtype=np.float64)
        neigh_species = np.empty(count, dtype=np.int64)
        pos = 0
        for idx in range(n_neigh):
            if int(i_p[idx]) == i:
                neigh_j[pos] = int(j_p[idx])
                neigh_r[pos] = r_p[idx]
                neigh_vec[pos, 0] = r_pc[idx, 0]
                neigh_vec[pos, 1] = r_pc[idx, 1]
                neigh_vec[pos, 2] = r_pc[idx, 2]
                neigh_species[pos] = int(species_index[int(j_p[idx])])
                pos += 1

        ti = int(species_index[i])
        for aidx in range(count - 1):
            j = neigh_j[aidx]
            tj = neigh_species[aidx]
            rij = neigh_vec[aidx]
            pair_ij = pair_parameters[ti, tj]
            for bidx in range(aidx + 1, count):
                k = neigh_j[bidx]
                tk = neigh_species[bidx]
                rik = neigh_vec[bidx]
                pair_ik = pair_parameters[ti, tk]
                lambda_jk = lambda_values[ti, tj, tk]
                if tj > tk:
                    lambda_jk = lambda_values[ti, tk, tj]
                if lambda_jk == 0.0:
                    continue
                trip_energy, f_i, f_j, f_k = _triplet_energy_and_forces_numba(
                    rij, rik, pair_ij, pair_ik, lambda_jk, costheta0
                )
                if trip_energy == 0.0:
                    continue
                energy += trip_energy
                forces[i, 0] += f_i[0]
                forces[i, 1] += f_i[1]
                forces[i, 2] += f_i[2]
                forces[j, 0] += f_j[0]
                forces[j, 1] += f_j[1]
                forces[j, 2] += f_j[2]
                forces[k, 0] += f_k[0]
                forces[k, 1] += f_k[1]
                forces[k, 2] += f_k[2]
                virial[0, 0] += rij[0] * f_j[0] + rik[0] * f_k[0]
                virial[0, 1] += rij[0] * f_j[1] + rik[0] * f_k[1]
                virial[0, 2] += rij[0] * f_j[2] + rik[0] * f_k[2]
                virial[1, 0] += rij[1] * f_j[0] + rik[1] * f_k[0]
                virial[1, 1] += rij[1] * f_j[1] + rik[1] * f_k[1]
                virial[1, 2] += rij[1] * f_j[2] + rik[1] * f_k[2]
                virial[2, 0] += rij[2] * f_j[0] + rik[2] * f_k[0]
                virial[2, 1] += rij[2] * f_j[1] + rik[2] * f_k[1]
                virial[2, 2] += rij[2] * f_j[2] + rik[2] * f_k[2]

    return energy, forces, virial


class NumbaSWEngine:
    """Numba-accelerated Stillinger-Weber calculator."""

    def __init__(self, sw_data: SWData, mode: str = "run"):
        self.sw_data = sw_data
        self.mode = mode

    def update(self, sw_data: SWData) -> None:
        self.sw_data = sw_data

    def _species_index(self, atoms: Atoms) -> np.ndarray:
        mapping = {int(number): idx for idx, number in enumerate(self.sw_data.species_numbers.tolist())}
        indices = np.empty(len(atoms), dtype=np.int64)
        for idx, number in enumerate(atoms.numbers.tolist()):
            number = int(number)
            if number not in mapping:
                raise ValueError(
                    f"Atom species {number} is not present in SW species list {self.sw_data.species!r}."
                )
            indices[idx] = mapping[number]
        return indices

    def calculate(self, atoms: Atoms) -> dict:
        sw = self.sw_data
        cutoff = sw.max_cutoff
        i_p, j_p, r_p, r_pc = neighbor_list("ijdD", atoms, cutoff)
        species_index = self._species_index(atoms)
        energy, forces, virial = _calculate_sw(
            np.asarray(i_p, dtype=np.int64),
            np.asarray(j_p, dtype=np.int64),
            np.asarray(r_p, dtype=np.float64),
            np.asarray(r_pc, dtype=np.float64),
            len(atoms),
            species_index,
            np.asarray(sw.pair_parameters, dtype=np.float64),
            np.asarray(sw.lambda_values, dtype=np.float64),
            float(sw.costheta0),
        )
        stress = full_3x3_to_voigt_6_stress(-virial / atoms.get_volume())
        return {
            "energy": float(energy),
            "energies": np.full(len(atoms), float(energy) / len(atoms) if len(atoms) else 0.0, dtype=float),
            "forces": forces,
            "stress": stress,
        }

    def _finite_difference_response(self, atoms: Atoms, delta: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_params = np.asarray(self.sw_data.parameters, dtype=float).copy()
        nprm = orig_params.size
        natoms = len(atoms)
        d_energy = np.zeros(nprm, dtype=float)
        d_forces = np.zeros((nprm, natoms, 3), dtype=float)
        d_stress = np.zeros((nprm, 3, 3), dtype=float)

        try:
            for i in range(nprm):
                p_plus = orig_params.copy()
                p_plus[i] += delta
                self.sw_data.parameters = p_plus
                plus = self.calculate(atoms)

                p_minus = orig_params.copy()
                p_minus[i] -= delta
                self.sw_data.parameters = p_minus
                minus = self.calculate(atoms)

                d_energy[i] = (plus["energy"] - minus["energy"]) / (2.0 * delta)
                d_forces[i] = (plus["forces"] - minus["forces"]) / (2.0 * delta)
                d_stress[i] = (plus["stress"] - minus["stress"]) / (2.0 * delta)
        finally:
            self.sw_data.parameters = orig_params
        return d_energy, d_forces, d_stress

    def jac_energy(self, atoms: Atoms):
        d_energy, _, _ = self._finite_difference_response(atoms)
        return type("JacobianShim", (), {"parameters": d_energy})

    def jac_forces(self, atoms: Atoms):
        _, d_forces, _ = self._finite_difference_response(atoms)
        return type("JacobianShim", (), {"parameters": d_forces})

    def jac_stress(self, atoms: Atoms):
        _, _, d_stress = self._finite_difference_response(atoms)
        return type("JacobianShim", (), {"parameters": d_stress})
