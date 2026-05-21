"""NumPy reference EAM calculator.

This engine evaluates the tabulated EAM splines directly with NumPy and
SciPy, without delegating to ASE's EAM calculator.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from scipy.interpolate import CubicSpline

from forgeff.potentials.eam.data import EAMData


def _spline_eval_1d(coeffs, x, x0, h, idx0):
    nseg = coeffs.shape[1]
    idx = int((x - x0) / h)
    if idx < 0:
        idx = 0
    elif idx >= nseg:
        idx = nseg - 1
    dx = x - (x0 + idx * h)
    return (((coeffs[0, idx, idx0] * dx + coeffs[1, idx, idx0]) * dx + coeffs[2, idx, idx0]) * dx + coeffs[3, idx, idx0])


def _spline_eval_2d(coeffs, x, x0, h, idx0, idx1):
    nseg = coeffs.shape[1]
    idx = int((x - x0) / h)
    if idx < 0:
        idx = 0
    elif idx >= nseg:
        idx = nseg - 1
    dx = x - (x0 + idx * h)
    return (
        (((coeffs[0, idx, idx0, idx1] * dx + coeffs[1, idx, idx0, idx1]) * dx + coeffs[2, idx, idx0, idx1]) * dx)
        + coeffs[3, idx, idx0, idx1]
    )


def _spline_deriv_1d(coeffs, x, x0, h, idx0):
    nseg = coeffs.shape[1]
    idx = int((x - x0) / h)
    if idx < 0:
        idx = 0
    elif idx >= nseg:
        idx = nseg - 1
    dx = x - (x0 + idx * h)
    return 3.0 * coeffs[0, idx, idx0] * dx * dx + 2.0 * coeffs[1, idx, idx0] * dx + coeffs[2, idx, idx0]


def _spline_deriv_2d(coeffs, x, x0, h, idx0, idx1):
    nseg = coeffs.shape[1]
    idx = int((x - x0) / h)
    if idx < 0:
        idx = 0
    elif idx >= nseg:
        idx = nseg - 1
    dx = x - (x0 + idx * h)
    return 3.0 * coeffs[0, idx, idx0, idx1] * dx * dx + 2.0 * coeffs[1, idx, idx0, idx1] * dx + coeffs[2, idx, idx0, idx1]


def _spline_eval_many(coeffs, x, x0, h):
    arr = np.asarray(x, dtype=float).reshape(-1)
    nseg = coeffs.shape[1]
    idx = np.clip(((arr - x0) / h).astype(np.int64), 0, nseg - 1)
    dx = (arr - (x0 + idx * h)).reshape((arr.size,) + (1,) * (coeffs.ndim - 2))
    c0 = coeffs[0, idx]
    c1 = coeffs[1, idx]
    c2 = coeffs[2, idx]
    c3 = coeffs[3, idx]
    return (((c0 * dx + c1) * dx + c2) * dx + c3)


def _spline_deriv_many(coeffs, x, x0, h):
    arr = np.asarray(x, dtype=float).reshape(-1)
    nseg = coeffs.shape[1]
    idx = np.clip(((arr - x0) / h).astype(np.int64), 0, nseg - 1)
    dx = (arr - (x0 + idx * h)).reshape((arr.size,) + (1,) * (coeffs.ndim - 2))
    c0 = coeffs[0, idx]
    c1 = coeffs[1, idx]
    c2 = coeffs[2, idx]
    return 3.0 * c0 * dx * dx + 2.0 * c1 * dx + c2


def _calculate_eam_alloy(types, i_list, j_list, dist, rvec, emb_coeffs, dens_coeffs, phi_coeffs, drho, dr, rho_start, r_start):
    natoms = types.shape[0]
    total_density = np.zeros(natoms)
    site_energies = np.zeros(natoms)
    if dist.size == 0:
        return 0.0, 0.0, total_density, site_energies, np.zeros((natoms, 3)), np.zeros((natoms, 3, 3))

    ti = types[i_list]
    tj = types[j_list]
    pair_idx = np.arange(dist.shape[0], dtype=np.int64)
    phi_all = _spline_eval_many(phi_coeffs, dist, r_start, dr)
    dens_all = _spline_eval_many(dens_coeffs, dist, r_start, dr)
    dphi_all = _spline_deriv_many(phi_coeffs, dist, r_start, dr)
    ddens_j_all = _spline_deriv_many(dens_coeffs, dist, r_start, dr)
    ddens_i_all = _spline_deriv_many(dens_coeffs, dist, r_start, dr)

    phi_vals = phi_all[pair_idx, ti, tj]
    dens_vals = dens_all[pair_idx, tj]
    dphi_vals = dphi_all[pair_idx, ti, tj]
    ddens_j = ddens_j_all[pair_idx, tj]
    ddens_i = ddens_i_all[pair_idx, ti]

    pair_energy_sum = float(np.sum(phi_vals))
    pair_mask = i_list < j_list
    if np.any(pair_mask):
        half_pair = 0.5 * phi_vals[pair_mask]
        np.add.at(site_energies, i_list[pair_mask], half_pair)
        np.add.at(site_energies, j_list[pair_mask], half_pair)

    np.add.at(total_density, i_list, dens_vals)

    emb_all = _spline_eval_many(emb_coeffs, total_density, rho_start, drho)
    d_emb_all = _spline_deriv_many(emb_coeffs, total_density, rho_start, drho)
    atom_idx = np.arange(natoms, dtype=np.int64)
    emb_i = emb_all[atom_idx, types]
    d_emb = d_emb_all[atom_idx, types]
    site_energies += emb_i

    forces = np.zeros((natoms, 3))
    stresses = np.zeros((natoms, 3, 3))
    scale = dphi_vals + d_emb[i_list] * ddens_j + d_emb[j_list] * ddens_i
    unit = rvec / dist[:, None]
    pair_force = scale[:, None] * unit
    np.add.at(forces, i_list, pair_force)
    np.add.at(stresses, i_list, pair_force[:, :, None] * rvec[:, None, :])

    pair_energy = 0.5 * pair_energy_sum
    embedding_energy = float(np.sum(emb_i))
    return pair_energy, embedding_energy, total_density, site_energies, forces, stresses


def _calculate_eam_fs(types, i_list, j_list, dist, rvec, emb_coeffs, dens_coeffs, phi_coeffs, drho, dr, rho_start, r_start):
    natoms = types.shape[0]
    total_density = np.zeros(natoms)
    site_energies = np.zeros(natoms)
    if dist.size == 0:
        return 0.0, 0.0, total_density, site_energies, np.zeros((natoms, 3)), np.zeros((natoms, 3, 3))

    ti = types[i_list]
    tj = types[j_list]
    pair_idx = np.arange(dist.shape[0], dtype=np.int64)
    pair_energy_all = _spline_eval_many(phi_coeffs, dist, r_start, dr)
    dens_all = _spline_eval_many(dens_coeffs, dist, r_start, dr)
    dphi_all = _spline_deriv_many(phi_coeffs, dist, r_start, dr)
    ddens_ij_all = _spline_deriv_many(dens_coeffs, dist, r_start, dr)
    ddens_ji_all = _spline_deriv_many(dens_coeffs, dist, r_start, dr)

    pair_energy_vals = pair_energy_all[pair_idx, ti, tj]
    dens_vals = dens_all[pair_idx, tj, ti]
    dphi_vals = dphi_all[pair_idx, ti, tj]
    ddens_ij = ddens_ij_all[pair_idx, tj, ti]
    ddens_ji = ddens_ji_all[pair_idx, ti, tj]

    pair_energy_sum = float(np.sum(pair_energy_vals))
    pair_mask = i_list < j_list
    if np.any(pair_mask):
        half_pair = 0.5 * pair_energy_vals[pair_mask]
        np.add.at(site_energies, i_list[pair_mask], half_pair)
        np.add.at(site_energies, j_list[pair_mask], half_pair)
    np.add.at(total_density, i_list, dens_vals)

    emb_all = _spline_eval_many(emb_coeffs, total_density, rho_start, drho)
    d_emb_all = _spline_deriv_many(emb_coeffs, total_density, rho_start, drho)
    atom_idx = np.arange(natoms, dtype=np.int64)
    emb_i = emb_all[atom_idx, types]
    d_emb = d_emb_all[atom_idx, types]
    site_energies += emb_i

    forces = np.zeros((natoms, 3))
    stresses = np.zeros((natoms, 3, 3))
    scale = dphi_vals + d_emb[i_list] * ddens_ij + d_emb[j_list] * ddens_ji
    unit = rvec / dist[:, None]
    pair_force = scale[:, None] * unit
    np.add.at(forces, i_list, pair_force)
    np.add.at(stresses, i_list, pair_force[:, :, None] * rvec[:, None, :])

    pair_energy = 0.5 * pair_energy_sum
    embedding_energy = float(np.sum(emb_i))
    return pair_energy, embedding_energy, total_density, site_energies, forces, stresses


class NumpyEAMEngine:
    """NumPy/SciPy EAM calculator used as the native reference path."""

    def __init__(self, eam_data: EAMData, mode: str = "run"):
        self.eam_data = eam_data
        self.mode = mode
        self._build_splines()

    def _build_splines(self):
        eam_data = self.eam_data
        self.r = eam_data.r_grid
        self.rho = eam_data.rho_grid
        self.dr = float(self.r[1] - self.r[0])
        self.drho = float(self.rho[1] - self.rho[0])
        self.form = getattr(eam_data, "form", "alloy")

        self._emb_coeffs = np.ascontiguousarray(CubicSpline(self.rho, eam_data.emb_values, axis=-1).c)
        if self.form == "fs":
            self._dens_coeffs = np.ascontiguousarray(CubicSpline(self.r, eam_data.rho_values, axis=-1).c)
        else:
            self._dens_coeffs = np.ascontiguousarray(
                CubicSpline(self.r, eam_data.rho_values.diagonal(axis1=0, axis2=1).T, axis=-1).c
            )
        self._phi_coeffs = np.ascontiguousarray(CubicSpline(self.r, eam_data.phi_values, axis=-1).c)

    def update(self, eam_data: EAMData):
        self.eam_data = eam_data
        self._build_splines()

    def _species_index(self, atoms: Atoms) -> np.ndarray:
        species = np.asarray(self.eam_data.species, dtype=np.int64)
        mapping = {int(number): idx for idx, number in enumerate(species.tolist())}
        indices = np.empty(len(atoms), dtype=np.int64)
        for idx, number in enumerate(atoms.numbers.tolist()):
            number = int(number)
            if number not in mapping:
                raise ValueError(
                    f"Atom species {number} is not present in EAM species list {self.eam_data.species!r}."
                )
            indices[idx] = mapping[number]
        return indices

    def _finite_difference_response(self, atoms: Atoms, delta: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        jac, _, _ = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=jac)

    def jac_forces(self, atoms: Atoms) -> SimpleNamespace:
        _, jac, _ = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=jac)

    def jac_stress(self, atoms: Atoms) -> SimpleNamespace:
        _, _, jac = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=jac)

    def calculate(self, atoms: Atoms) -> dict:
        types = self._species_index(atoms)
        cutoff = float(self.r[-1])
        i_list, j_list, shifts, dist = neighbor_list("ijSd", atoms, cutoff)
        rvec = atoms.positions[j_list] + shifts @ atoms.cell.array - atoms.positions[i_list]

        if self.form == "fs":
            pair_energy, embedding_energy, total_density, site_energies, forces, stresses = _calculate_eam_fs(
                types,
                i_list.astype(np.int64),
                j_list.astype(np.int64),
                dist.astype(np.float64),
                rvec.astype(np.float64),
                self._emb_coeffs,
                self._dens_coeffs,
                self._phi_coeffs,
                float(self.drho),
                float(self.dr),
                float(self.rho[0]),
                float(self.r[0]),
            )
        else:
            pair_energy, embedding_energy, total_density, site_energies, forces, stresses = _calculate_eam_alloy(
                types,
                i_list.astype(np.int64),
                j_list.astype(np.int64),
                dist.astype(np.float64),
                rvec.astype(np.float64),
                self._emb_coeffs,
                self._dens_coeffs,
                self._phi_coeffs,
                float(self.drho),
                float(self.dr),
                float(self.rho[0]),
                float(self.r[0]),
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


# Backward-compatible alias. Keep the old name alive while the native NumPy
# engine becomes the public path.
ASEAMEngine = NumpyEAMEngine
