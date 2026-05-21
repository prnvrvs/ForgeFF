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
        rvec = atoms.positions[j_list] + shifts @ atoms.cell.array - atoms.positions[i_list]

        natoms = len(atoms)
        total_density = np.zeros(natoms, dtype=float)
        pair_energy_sum = 0.0
        mu = np.zeros((natoms, 3), dtype=float)
        nu = np.zeros((natoms, 3, 3), dtype=float)
        site_energies = np.zeros(natoms, dtype=float)

        for k in range(dist.shape[0]):
            i = int(i_list[k])
            j = int(j_list[k])
            r = float(dist[k])
            if r <= 0.0:
                continue
            ti = int(species_index[i])
            tj = int(species_index[j])

            pair_energy = float(self._phi_spline(r)[ti, tj])
            pair_energy_sum += pair_energy
            if i < j:
                site_energies[i] += 0.5 * pair_energy
                site_energies[j] += 0.5 * pair_energy
            total_density[i] += float(self._dens_spline(r)[tj])

            u = float(self._dipole_spline(r)[ti, tj])
            mu[i] += u * rvec[k]

            w = float(self._quadrupole_spline(r)[ti, tj])
            nu[i] += w * np.outer(rvec[k], rvec[k])

        pair_energy = 0.5 * pair_energy_sum
        d_emb = np.zeros(natoms, dtype=float)
        embedding_energy = 0.0
        for i in range(natoms):
            ti = int(species_index[i])
            emb_i = float(self._emb_spline(total_density[i])[ti])
            d_emb[i] = float(self._emb_spline(total_density[i], 1)[ti])
            embedding_energy += emb_i
            site_energies[i] += emb_i

        dipole_energy = 0.0
        quad_energy = 0.0
        for i in range(natoms):
            dip_i = 0.5 * float(np.sum(mu[i] ** 2))
            t_nu = float(np.trace(nu[i]))
            quad_i = 0.5 * float(np.sum(nu[i] ** 2)) - (1.0 / 6.0) * t_nu**2
            dipole_energy += dip_i
            quad_energy += quad_i
            site_energies[i] += dip_i + quad_i

        forces = np.zeros((natoms, 3), dtype=float)
        stresses = np.zeros((natoms, 3, 3), dtype=float)

        for k in range(dist.shape[0]):
            i = int(i_list[k])
            j = int(j_list[k])
            r = float(dist[k])
            if r <= 0.0:
                continue
            ti = int(species_index[i])
            tj = int(species_index[j])
            rhat = rvec[k] / r

            dphi = float(self._phi_spline(r, 1)[ti, tj])
            ddens_ij = float(self._dens_spline(r, 1)[tj])
            ddens_ji = float(self._dens_spline(r, 1)[ti])
            scale_eam = dphi + d_emb[i] * ddens_ij + d_emb[j] * ddens_ji

            adp_force = self._angular_forces(mu[i], mu[j], nu[i], nu[j], r, rvec[k], ti, tj)

            fx = scale_eam * rhat[0] + adp_force[0]
            fy = scale_eam * rhat[1] + adp_force[1]
            fz = scale_eam * rhat[2] + adp_force[2]

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

        total_energy = float(np.sum(site_energies))
        results = {
            "energy": total_energy,
            "energies": site_energies,
            "forces": forces,
        }
        if atoms.cell.rank == 3:
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
