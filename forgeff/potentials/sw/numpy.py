"""NumPy reference Stillinger-Weber calculator."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

from .data import SWData


def _pair_energy_and_dedr(r: float, params: np.ndarray) -> tuple[float, float]:
    A, B, p, q, delta, a1, _, _ = params
    if r <= 0.0 or r >= a1:
        return 0.0, 0.0
    exp_term = np.exp(delta / (r - a1))
    power = A * r**(-p) - B * r**(-q)
    pair_energy = power * exp_term
    d_power = -A * p * r**(-p - 1.0) + B * q * r**(-q - 1.0)
    d_pair = d_power * exp_term + power * exp_term * (-delta / (r - a1) ** 2)
    return float(pair_energy), float(d_pair)


def _pair_energy_and_dedr_vectorized(r: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(r, dtype=float).reshape(-1)
    prm = np.asarray(params, dtype=float)
    if prm.shape != (arr.size, 8):
        raise ValueError("Pair parameter array must have shape (n, 8) for vectorized SW evaluation.")

    energy = np.zeros(arr.size, dtype=float)
    dedr = np.zeros(arr.size, dtype=float)
    mask = (arr > 0.0) & (arr < prm[:, 5])
    if not np.any(mask):
        return energy, dedr

    r_m = arr[mask]
    prm_m = prm[mask]
    a1 = prm_m[:, 5]
    delta = prm_m[:, 4]
    exp_term = np.exp(delta / (r_m - a1))
    power = prm_m[:, 0] * np.power(r_m, -prm_m[:, 2]) - prm_m[:, 1] * np.power(r_m, -prm_m[:, 3])
    energy_m = power * exp_term
    d_power = (
        -prm_m[:, 0] * prm_m[:, 2] * np.power(r_m, -prm_m[:, 2] - 1.0)
        + prm_m[:, 1] * prm_m[:, 3] * np.power(r_m, -prm_m[:, 3] - 1.0)
    )
    dedr_m = d_power * exp_term + power * exp_term * (-delta / (r_m - a1) ** 2)
    energy[mask] = energy_m
    dedr[mask] = dedr_m
    return energy, dedr


def _triplet_energy_and_forces(
    rij: np.ndarray,
    rik: np.ndarray,
    pair_ij: np.ndarray,
    pair_ik: np.ndarray,
    lambda_value: float,
    costheta0: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, _, _, _, gamma_ij, a2_ij = pair_ij
    _, _, _, _, _, _, gamma_ik, a2_ik = pair_ik
    u = float(np.linalg.norm(rij))
    v = float(np.linalg.norm(rik))
    if u <= 0.0 or v <= 0.0 or u >= a2_ij or v >= a2_ik:
        return 0.0, np.zeros(3), np.zeros(3), np.zeros(3)

    u_hat = rij / u
    v_hat = rik / v
    c = float(np.dot(u_hat, v_hat))
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

    dc_drij = (v_hat - c * u_hat) / u
    dc_drik = (u_hat - c * v_hat) / v

    grad_j = dE_du * u_hat + dE_dc * dc_drij
    grad_k = dE_dv * v_hat + dE_dc * dc_drik
    force_j = -grad_j
    force_k = -grad_k
    force_i = -(force_j + force_k)
    return float(energy), force_i, force_j, force_k


def _triplet_energy_and_forces_vectorized(
    rij: np.ndarray,
    rik: np.ndarray,
    pair_ij: np.ndarray,
    pair_ik: np.ndarray,
    lambda_value: np.ndarray,
    costheta0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rij = np.asarray(rij, dtype=float)
    rik = np.asarray(rik, dtype=float)
    pair_ij = np.asarray(pair_ij, dtype=float)
    pair_ik = np.asarray(pair_ik, dtype=float)
    lambda_value = np.asarray(lambda_value, dtype=float).reshape(-1)

    if rij.size == 0:
        zeros = np.zeros((0, 3), dtype=float)
        return np.zeros(0, dtype=float), zeros, zeros, zeros
    if pair_ij.shape != pair_ik.shape or pair_ij.shape[0] != rij.shape[0]:
        raise ValueError("Triplet parameter arrays must match the number of neighbor pairs.")

    u = np.linalg.norm(rij, axis=1)
    v = np.linalg.norm(rik, axis=1)
    mask = (
        (u > 0.0)
        & (v > 0.0)
        & (u < pair_ij[:, 7])
        & (v < pair_ik[:, 7])
        & (lambda_value != 0.0)
    )
    energies = np.zeros(rij.shape[0], dtype=float)
    force_i = np.zeros_like(rij)
    force_j = np.zeros_like(rij)
    force_k = np.zeros_like(rij)
    if not np.any(mask):
        return energies, force_i, force_j, force_k

    rij_m = rij[mask]
    rik_m = rik[mask]
    pair_ij_m = pair_ij[mask]
    pair_ik_m = pair_ik[mask]
    lambda_m = lambda_value[mask]
    u_m = u[mask]
    v_m = v[mask]

    u_hat = rij_m / u_m[:, None]
    v_hat = rik_m / v_m[:, None]
    c = np.einsum("ij,ij->i", u_hat, v_hat)
    gamma_ij = pair_ij_m[:, 6]
    gamma_ik = pair_ik_m[:, 6]
    a2_ij = pair_ij_m[:, 7]
    a2_ik = pair_ik_m[:, 7]
    expo_u = np.exp(gamma_ij / (u_m - a2_ij))
    expo_v = np.exp(gamma_ik / (v_m - a2_ik))
    g = (c + costheta0) ** 2
    pref = lambda_m * expo_u * expo_v
    energy_m = pref * g

    d_pref_du = pref * (-gamma_ij / (u_m - a2_ij) ** 2)
    d_pref_dv = pref * (-gamma_ik / (v_m - a2_ik) ** 2)
    dE_du = d_pref_du * g
    dE_dv = d_pref_dv * g
    dE_dc = pref * 2.0 * (c + costheta0)

    dc_drij = (v_hat - c[:, None] * u_hat) / u_m[:, None]
    dc_drik = (u_hat - c[:, None] * v_hat) / v_m[:, None]
    grad_j = dE_du[:, None] * u_hat + dE_dc[:, None] * dc_drij
    grad_k = dE_dv[:, None] * v_hat + dE_dc[:, None] * dc_drik
    force_j_m = -grad_j
    force_k_m = -grad_k
    force_i_m = -(force_j_m + force_k_m)

    energies[mask] = energy_m
    force_i[mask] = force_i_m
    force_j[mask] = force_j_m
    force_k[mask] = force_k_m
    return energies, force_i, force_j, force_k


def _collect_triplets(
    i_sorted: np.ndarray,
    j_sorted: np.ndarray,
    vec_sorted: np.ndarray,
    species_index: np.ndarray,
    pair_parameters: np.ndarray,
    lambda_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers, first = np.unique(i_sorted, return_index=True)
    if centers.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        empty_vec = np.empty((0, 3), dtype=float)
        empty_pair = np.empty((0, 8), dtype=float)
        return empty, empty, empty, empty_vec, empty_vec, empty_pair, empty_pair, empty

    last = np.r_[first[1:], len(i_sorted)]
    center_chunks: list[np.ndarray] = []
    j_chunks: list[np.ndarray] = []
    k_chunks: list[np.ndarray] = []
    rij_chunks: list[np.ndarray] = []
    rik_chunks: list[np.ndarray] = []
    pair_ij_chunks: list[np.ndarray] = []
    pair_ik_chunks: list[np.ndarray] = []
    lambda_chunks: list[np.ndarray] = []

    for idx, center in enumerate(centers.tolist()):
        start = int(first[idx])
        stop = int(last[idx])
        n_neigh = stop - start
        if n_neigh < 2:
            continue
        local = np.arange(n_neigh, dtype=np.int64)
        a_idx, b_idx = np.triu_indices(n_neigh, k=1)
        neigh_j = j_sorted[start:stop]
        neigh_vec = vec_sorted[start:stop]
        j = neigh_j[a_idx]
        k = neigh_j[b_idx]
        ti = int(species_index[center])
        tj = species_index[j]
        tk = species_index[k]
        center_chunks.append(np.full(a_idx.size, center, dtype=np.int64))
        j_chunks.append(np.asarray(j, dtype=np.int64))
        k_chunks.append(np.asarray(k, dtype=np.int64))
        rij_chunks.append(np.asarray(neigh_vec[a_idx], dtype=float))
        rik_chunks.append(np.asarray(neigh_vec[b_idx], dtype=float))
        pair_ij_chunks.append(np.asarray(pair_parameters[ti, tj], dtype=float))
        pair_ik_chunks.append(np.asarray(pair_parameters[ti, tk], dtype=float))
        lambda_chunks.append(np.asarray(lambda_values[ti, tj, tk], dtype=float).reshape(-1))

    if not center_chunks:
        empty = np.empty((0,), dtype=np.int64)
        empty_vec = np.empty((0, 3), dtype=float)
        empty_pair = np.empty((0, 8), dtype=float)
        return empty, empty, empty, empty_vec, empty_vec, empty_pair, empty_pair, empty

    return (
        np.concatenate(center_chunks),
        np.concatenate(j_chunks),
        np.concatenate(k_chunks),
        np.concatenate(rij_chunks),
        np.concatenate(rik_chunks),
        np.concatenate(pair_ij_chunks),
        np.concatenate(pair_ik_chunks),
        np.concatenate(lambda_chunks),
    )


class NumpySWEngine:
    """Reference SW calculator used as the NumPy path."""

    def __init__(self, sw_data: SWData, mode: str = "run"):
        self.sw_data = sw_data
        self.mode = mode

    def update(self, sw_data: SWData) -> None:
        self.sw_data = sw_data

    def _species_index(self, atoms: Atoms) -> np.ndarray:
        mapping = {int(number): idx for idx, number in enumerate(self.sw_data.species_numbers.tolist())}
        indices = np.empty(len(atoms), dtype=np.int64)
        for idx, number in enumerate(atoms.numbers.tolist()):
            if int(number) not in mapping:
                raise ValueError(
                    f"Atom species {number} is not present in SW species list {self.sw_data.species!r}."
                )
            indices[idx] = mapping[int(number)]
        return indices

    def calculate(self, atoms: Atoms) -> dict:
        sw = self.sw_data
        cutoff = sw.max_cutoff
        i_p, j_p, r_p, r_pc = neighbor_list("ijdD", atoms, cutoff)
        species_index = self._species_index(atoms)

        natoms = len(atoms)
        energy = 0.0
        forces = np.zeros((natoms, 3), dtype=float)
        virial = np.zeros((3, 3), dtype=float)

        # Pair term: count each undirected pair once.
        pair_mask = i_p < j_p
        if np.any(pair_mask):
            pi = np.asarray(i_p[pair_mask], dtype=np.int64)
            pj = np.asarray(j_p[pair_mask], dtype=np.int64)
            pr = np.asarray(r_p[pair_mask], dtype=float)
            prc = np.asarray(r_pc[pair_mask], dtype=float)
            ti = species_index[pi]
            tj = species_index[pj]
            pair_params = sw.pair_parameters[ti, tj]
            pair_energy, d_pair = _pair_energy_and_dedr_vectorized(pr, pair_params)
            unit = prc / pr[:, None]
            f_j = -d_pair[:, None] * unit
            f_i = -f_j
            energy += float(np.sum(pair_energy))
            np.add.at(forces, pi, f_i)
            np.add.at(forces, pj, f_j)
            virial += np.einsum("pi,pj->ij", prc, f_j)

        self_mask = i_p == j_p
        if np.any(self_mask):
            pi = np.asarray(i_p[self_mask], dtype=np.int64)
            pr = np.asarray(r_p[self_mask], dtype=float)
            prc = np.asarray(r_pc[self_mask], dtype=float)
            ti = species_index[pi]
            pair_params = sw.pair_parameters[ti, ti]
            pair_energy, d_pair = _pair_energy_and_dedr_vectorized(pr, pair_params)
            unit = prc / pr[:, None]
            f_j = -d_pair[:, None] * unit
            energy += 0.5 * float(np.sum(pair_energy))
            virial += 0.5 * np.einsum("pi,pj->ij", prc, f_j)

        # Triplet term: center i with unordered neighbor pairs (j, k), j < k.
        if len(i_p) > 0:
            order = np.argsort(i_p, kind="mergesort")
            i_sorted = np.asarray(i_p[order], dtype=np.int64)
            j_sorted = np.asarray(j_p[order], dtype=np.int64)
            vec_sorted = np.asarray(r_pc[order], dtype=float)
            (
                center_idx,
                j_idx,
                k_idx,
                rij,
                rik,
                pair_ij,
                pair_ik,
                lambda_value,
            ) = _collect_triplets(i_sorted, j_sorted, vec_sorted, species_index, sw.pair_parameters, sw.lambda_values)
            if center_idx.size:
                trip_energy, f_i, f_j, f_k = _triplet_energy_and_forces_vectorized(
                    rij,
                    rik,
                    pair_ij,
                    pair_ik,
                    lambda_value,
                    sw.costheta0,
                )
                if np.any(trip_energy):
                    energy += float(np.sum(trip_energy))
                    np.add.at(forces, center_idx, f_i)
                    np.add.at(forces, j_idx, f_j)
                    np.add.at(forces, k_idx, f_k)
                    virial += np.einsum("pi,pj->ij", rij, f_j)
                    virial += np.einsum("pi,pj->ij", rik, f_k)

        results = {
            "energy": energy,
            "energies": np.full(natoms, energy / natoms if natoms else 0.0, dtype=float),
            "forces": forces,
        }
        if atoms.cell.rank == 3 and atoms.get_volume() != 0.0:
            results["stress"] = full_3x3_to_voigt_6_stress(-virial / atoms.get_volume())
        return results

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
                if "stress" in plus and "stress" in minus:
                    d_stress[i] = (
                        voigt_6_to_full_3x3_stress(plus["stress"])
                        - voigt_6_to_full_3x3_stress(minus["stress"])
                    ) / (2.0 * delta)
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
