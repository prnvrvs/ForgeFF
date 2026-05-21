"""NumPy reference ADP calculator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress
from scipy.interpolate import CubicSpline

from forgeff.potentials.eam.adp_data import ADPData


class NumpyADPEngine:
    """Reference ADP calculator used as the NumPy path."""

    def __init__(self, pot_data: ADPData, mode: str = "run"):
        self.pot_data = pot_data
        self.mode = mode
        self._build_splines()

    def _build_splines(self) -> None:
        pot_data = self.pot_data
        self.r = pot_data.r_grid
        self.rho = pot_data.rho_grid
        self.dr = float(self.r[1] - self.r[0])
        self.drho = float(self.rho[1] - self.rho[0])
        self.form = getattr(pot_data, "form", "alloy")
        if self.form == "fs":
            raise NotImplementedError("ADP Finnis-Sinclair mode is not supported in the NumPy engine yet.")

        self._emb_spline = CubicSpline(self.rho, pot_data.emb_values, axis=-1)
        self._dens_spline = CubicSpline(self.r, pot_data.rho_values.diagonal(axis1=0, axis2=1).T, axis=-1)
        self._phi_spline = CubicSpline(self.r, pot_data.phi_values, axis=-1)
        self._dipole_spline = CubicSpline(self.r, pot_data.dipole_values, axis=-1)
        self._quadrupole_spline = CubicSpline(self.r, pot_data.quadrupole_values, axis=-1)

    def update(self, pot_data: ADPData) -> None:
        self.pot_data = pot_data
        self._build_splines()

    def _species_index(self, atoms: Atoms) -> np.ndarray:
        mapping = {int(number): idx for idx, number in enumerate(self.pot_data.species.tolist())}
        indices = np.empty(len(atoms), dtype=np.int64)
        for idx, number in enumerate(atoms.numbers.tolist()):
            if int(number) not in mapping:
                raise ValueError(
                    f"Atom species {number} is not present in ADP species list {self.pot_data.species!r}."
                )
            indices[idx] = mapping[int(number)]
        return indices

    def _pair_values(self, r: float, ti: int, tj: int) -> tuple[float, float, float]:
        phi = float(self._phi_spline(r)[ti, tj])
        dphi = float(self._phi_spline(r, 1)[ti, tj])
        dens = float(self._dens_spline(r)[tj])
        ddens = float(self._dens_spline(r, 1)[tj])
        return phi, dphi, dens, ddens

    def _angular_forces(
        self,
        mu_i: np.ndarray,
        mu_j: np.ndarray,
        lam_i: np.ndarray,
        lam_j: np.ndarray,
        r: float,
        rvec: np.ndarray,
        ti: int,
        tj: int,
    ) -> np.ndarray:
        dip = float(self._dipole_spline(r)[ti, tj])
        ddip = float(self._dipole_spline(r, 1)[ti, tj])
        quad = float(self._quadrupole_spline(r)[ti, tj])
        dquad = float(self._quadrupole_spline(r, 1)[ti, tj])

        force = np.zeros(3, dtype=float)
        mu_diff = mu_i - mu_j
        trace_i = float(np.trace(lam_i))
        trace_j = float(np.trace(lam_j))

        for gamma in range(3):
            term1 = mu_diff[gamma] * dip
            term2 = float(np.dot(mu_diff, rvec)) * ddip * rvec[gamma] / r

            term3 = 0.0
            for alpha in range(3):
                term3 += (lam_i[alpha, gamma] + lam_j[alpha, gamma]) * rvec[alpha]
            term3 *= 2.0 * quad

            term4 = 0.0
            for alpha in range(3):
                for beta in range(3):
                    rs = rvec[alpha] * rvec[beta] * rvec[gamma]
                    term4 += (lam_i[alpha, beta] + lam_j[alpha, beta]) * dquad * rs / r

            term5 = (trace_i + trace_j) * (dquad * r + 2.0 * quad) * rvec[gamma] / 3.0
            force[gamma] = term1 + term2 + term3 + term4 - term5

        return force

    def calculate(self, atoms: Atoms) -> dict:
        species_index = self._species_index(atoms)
        cutoff = float(self.r[-1])
        i_list, j_list, shifts, dist = neighbor_list("ijSd", atoms, cutoff)
        i_list = np.asarray(i_list, dtype=np.int64)
        j_list = np.asarray(j_list, dtype=np.int64)
        dist = np.asarray(dist, dtype=float)
        rvec = atoms.positions[j_list] + shifts @ atoms.cell.array - atoms.positions[i_list]

        natoms = len(atoms)
        total_density = np.zeros(natoms, dtype=float)
        mu = np.zeros((natoms, 3), dtype=float)
        nu = np.zeros((natoms, 3, 3), dtype=float)
        site_energies = np.zeros(natoms, dtype=float)
        pair_count = dist.shape[0]
        if pair_count == 0:
            results = {
                "energy": 0.0,
                "energies": site_energies,
                "forces": np.zeros((natoms, 3), dtype=float),
            }
            if atoms.cell.rank == 3 and atoms.get_volume() != 0.0:
                results["stress"] = np.zeros(6, dtype=float)
            return results

        ti = species_index[i_list]
        tj = species_index[j_list]
        pair_idx = np.arange(pair_count, dtype=np.int64)

        phi_all = np.moveaxis(self._phi_spline(dist), -1, 0)
        dens_all = np.moveaxis(self._dens_spline(dist), -1, 0)
        dip_all = np.moveaxis(self._dipole_spline(dist), -1, 0)
        quad_all = np.moveaxis(self._quadrupole_spline(dist), -1, 0)
        dphi_all = np.moveaxis(self._phi_spline(dist, 1), -1, 0)
        ddens_all = np.moveaxis(self._dens_spline(dist, 1), -1, 0)
        ddip_all = np.moveaxis(self._dipole_spline(dist, 1), -1, 0)
        dquad_all = np.moveaxis(self._quadrupole_spline(dist, 1), -1, 0)

        pair_energy_pair = phi_all[pair_idx, ti, tj]
        pair_energy_sum = float(np.sum(pair_energy_pair))
        pair_mask = i_list < j_list
        if np.any(pair_mask):
            half_pair = 0.5 * pair_energy_pair[pair_mask]
            np.add.at(site_energies, i_list[pair_mask], half_pair)
            np.add.at(site_energies, j_list[pair_mask], half_pair)

        dens_contrib = dens_all[pair_idx, tj]
        np.add.at(total_density, i_list, dens_contrib)
        np.add.at(mu, i_list, dip_all[pair_idx, ti, tj][:, None] * rvec)
        np.add.at(
            nu,
            i_list,
            quad_all[pair_idx, ti, tj][:, None, None] * (rvec[:, :, None] * rvec[:, None, :]),
        )

        emb_all = np.moveaxis(self._emb_spline(total_density), -1, 0)
        emb_deriv_all = np.moveaxis(self._emb_spline(total_density, 1), -1, 0)
        atom_idx = np.arange(natoms, dtype=np.int64)
        emb_i = emb_all[atom_idx, species_index]
        d_emb = emb_deriv_all[atom_idx, species_index]
        site_energies += emb_i

        dipole_energy = 0.5 * float(np.sum(mu * mu))
        nu_trace = np.trace(nu, axis1=1, axis2=2)
        quad_energy = 0.5 * float(np.sum(nu * nu)) - (1.0 / 6.0) * float(np.sum(nu_trace * nu_trace))
        site_energies += 0.5 * np.sum(mu * mu, axis=1)
        site_energies += 0.5 * np.sum(nu * nu, axis=(1, 2)) - (1.0 / 6.0) * nu_trace**2

        dphi = dphi_all[pair_idx, ti, tj]
        ddens_ij = ddens_all[pair_idx, tj]
        ddens_ji = ddens_all[pair_idx, ti]
        scale_eam = dphi + d_emb[i_list] * ddens_ij + d_emb[j_list] * ddens_ji

        pair_norm = dist[:, None]
        unit = rvec / pair_norm
        forces = np.zeros((natoms, 3), dtype=float)
        stresses = np.zeros((natoms, 3, 3), dtype=float)

        mu_diff = mu[i_list] - mu[j_list]
        mu_dot = np.einsum("ij,ij->i", mu_diff, rvec)
        trace_sum = nu_trace[i_list] + nu_trace[j_list]
        nu_sum = nu[i_list] + nu[j_list]
        term1 = dip_all[pair_idx, ti, tj][:, None] * mu_diff
        term2 = (ddip_all[pair_idx, ti, tj] * mu_dot / dist)[:, None] * rvec
        term3 = 2.0 * quad_all[pair_idx, ti, tj][:, None] * np.einsum("kij,kj->ki", nu_sum, rvec)
        term4_scalar = dquad_all[pair_idx, ti, tj] * np.einsum("kij,ki,kj->k", nu_sum, rvec, rvec) / dist
        term4 = term4_scalar[:, None] * rvec
        term5 = (
            trace_sum * (dquad_all[pair_idx, ti, tj] * dist + 2.0 * quad_all[pair_idx, ti, tj]) / 3.0
        )[:, None] * rvec
        adp_force = term1 + term2 + term3 + term4 - term5

        pair_force = scale_eam[:, None] * unit + adp_force
        np.add.at(forces, i_list, pair_force)
        np.add.at(stresses, i_list, pair_force[:, :, None] * rvec[:, None, :])

        total_energy = 0.5 * pair_energy_sum + float(np.sum(emb_i)) + dipole_energy + quad_energy
        results = {
            "energy": total_energy,
            "energies": site_energies,
            "forces": forces,
        }
        if atoms.cell.rank == 3 and atoms.get_volume() != 0.0:
            stress_tensor = 0.5 * np.sum(stresses, axis=0) / atoms.get_volume()
            results["stress"] = full_3x3_to_voigt_6_stress(stress_tensor)
        return results

    def _finite_difference_response(self, atoms: Atoms, delta: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_params = np.asarray(self.pot_data.parameters, dtype=float).copy()
        nprm = orig_params.size
        natoms = len(atoms)
        d_energy = np.zeros(nprm, dtype=float)
        d_forces = np.zeros((nprm, natoms, 3), dtype=float)
        d_stress = np.zeros((nprm, 3, 3), dtype=float)

        try:
            for i in range(nprm):
                p_plus = orig_params.copy()
                p_plus[i] += delta
                self.pot_data.parameters = p_plus
                self.update(self.pot_data)
                plus = self.calculate(atoms)

                p_minus = orig_params.copy()
                p_minus[i] -= delta
                self.pot_data.parameters = p_minus
                self.update(self.pot_data)
                minus = self.calculate(atoms)

                d_energy[i] = (plus["energy"] - minus["energy"]) / (2.0 * delta)
                d_forces[i] = (plus["forces"] - minus["forces"]) / (2.0 * delta)
                d_stress[i] = (plus["stress"] - minus["stress"]) / (2.0 * delta)
        finally:
            self.pot_data.parameters = orig_params
            self.update(self.pot_data)
        return d_energy, d_forces, d_stress

    def jac_energy(self, atoms: Atoms):
        d_energy, _, _ = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=d_energy)

    def jac_forces(self, atoms: Atoms):
        _, d_forces, _ = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=d_forces)

    def jac_stress(self, atoms: Atoms):
        _, _, d_stress = self._finite_difference_response(atoms)
        return SimpleNamespace(parameters=d_stress)
