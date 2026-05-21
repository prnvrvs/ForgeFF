"""NumPy reference Stillinger-Weber calculator."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress

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

        neighbors: dict[int, list[tuple[int, float, np.ndarray]]] = defaultdict(list)

        # Pair term: count each undirected pair once.
        for i, j, r, rij in zip(i_p, j_p, r_p, r_pc, strict=True):
            i = int(i)
            j = int(j)
            neighbors[i].append((j, float(r), np.asarray(rij, dtype=float)))
            if i >= j:
                continue
            pair_params = sw.pair_parameter_block(species_index[i], species_index[j])
            pair_energy, d_pair = _pair_energy_and_dedr(float(r), pair_params)
            if pair_energy == 0.0:
                continue
            unit = rij / float(r)
            f_j = -d_pair * unit
            f_i = -f_j
            energy += pair_energy
            forces[i] += f_i
            forces[j] += f_j
            virial += np.outer(rij, f_j)

        # Triplet term: center i with unordered neighbor pairs (j, k), j < k.
        for i, neigh in neighbors.items():
            if len(neigh) < 2:
                continue
            ti = species_index[i]
            for a in range(len(neigh) - 1):
                j, _, rij = neigh[a]
                tj = species_index[j]
                pair_ij = sw.pair_parameter_block(ti, tj)
                for b in range(a + 1, len(neigh)):
                    k, _, rik = neigh[b]
                    tk = species_index[k]
                    pair_ik = sw.pair_parameter_block(ti, tk)
                    lambda_value = sw.lambda_parameter(ti, tj, tk)
                    if lambda_value == 0.0:
                        continue
                    trip_energy, f_i, f_j, f_k = _triplet_energy_and_forces(
                        rij, rik, pair_ij, pair_ik, lambda_value, sw.costheta0
                    )
                    if trip_energy == 0.0:
                        continue
                    energy += trip_energy
                    forces[i] += f_i
                    forces[j] += f_j
                    forces[k] += f_k
                    virial += np.outer(rij, f_j)
                    virial += np.outer(rik, f_k)

        stress = full_3x3_to_voigt_6_stress(-virial / atoms.get_volume())
        return {
            "energy": energy,
            "energies": np.full(natoms, energy / natoms if natoms else 0.0, dtype=float),
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
