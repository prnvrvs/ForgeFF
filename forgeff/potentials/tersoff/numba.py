"""Numba-backed Tersoff calculator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numba
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress

from .data import TersoffData, TersoffParameters


_MAX_EXP_ARG = 69.0776e0
_MIN_EXP_ARG = -69.0776e0


@numba.njit(cache=True, inline="always")
def _calc_fc(r: float, R: float, D: float) -> float:
    if r > R + D:
        return 0.0
    if r < R - D:
        return 1.0
    return 0.5 * (1.0 - np.sin(np.pi * (r - R) / (2.0 * D)))


@numba.njit(cache=True, inline="always")
def _calc_fc_d(r: float, R: float, D: float) -> float:
    if r > R + D or r < R - D:
        return 0.0
    return -0.25 * np.pi / D * np.cos(np.pi * (r - R) / (2.0 * D))


@numba.njit(cache=True, inline="always")
def _calc_gijk(costheta: float, gamma: float, c: float, d: float, h: float) -> float:
    c2 = c * c
    d2 = d * d
    hcth = h - costheta
    return gamma * (1.0 + c2 / d2 - c2 / (d2 + hcth * hcth))


@numba.njit(cache=True, inline="always")
def _calc_gijk_d(costheta: float, gamma: float, c: float, d: float, h: float) -> float:
    c2 = c * c
    d2 = d * d
    hcth = h - costheta
    return (-2.0 * gamma * c2 * hcth) / (d2 + hcth * hcth) ** 2


@numba.njit(cache=True, inline="always")
def _calc_costheta_d(rij: np.ndarray, rik: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    abs_rij = np.sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2])
    abs_rik = np.sqrt(rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2])
    costheta = (rij[0] * rik[0] + rij[1] * rik[1] + rij[2] * rik[2]) / (abs_rij * abs_rik)
    drj = (rik / abs_rik - costheta * rij / abs_rij) / abs_rij
    drk = (rij / abs_rij - costheta * rik / abs_rik) / abs_rik
    dri = -(drj + drk)
    return dri, drj, drk


@numba.njit(cache=True, inline="always")
def _calc_zeta_d(
    rij: np.ndarray,
    rik: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam3 = params[2]
    m = params[0]
    gamma = params[1]
    c = params[3]
    d = params[4]
    h = params[5]
    R = params[10]
    D = params[11]

    abs_rij = np.sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2])
    abs_rik = np.sqrt(rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2])
    rij_hat = rij / abs_rij
    rik_hat = rik / abs_rik

    fcik = _calc_fc(abs_rik, R, D)
    dfcik = _calc_fc_d(abs_rik, R, D)

    tmp = (lam3 * (abs_rij - abs_rik)) ** m
    if tmp > _MAX_EXP_ARG:
        ex_delr = 1.0e30
    elif tmp < _MIN_EXP_ARG:
        ex_delr = 0.0
    else:
        ex_delr = np.exp(tmp)

    ex_delr_d = m * lam3**m * (abs_rij - abs_rik) ** (m - 1.0) * ex_delr
    costheta = rij_hat[0] * rik_hat[0] + rij_hat[1] * rik_hat[1] + rij_hat[2] * rik_hat[2]
    gijk = _calc_gijk(costheta, gamma, c, d, h)
    gijk_d = _calc_gijk_d(costheta, gamma, c, d, h)
    dcosdri, dcosdrj, dcosdrk = _calc_costheta_d(rij, rik)

    dri = -dfcik * gijk * ex_delr * rik_hat
    dri += fcik * gijk_d * ex_delr * dcosdri
    dri += fcik * gijk * ex_delr_d * rik_hat
    dri -= fcik * gijk * ex_delr_d * rij_hat

    drj = fcik * gijk_d * ex_delr * dcosdrj
    drj += fcik * gijk * ex_delr_d * rij_hat

    drk = dfcik * gijk * ex_delr * rik_hat
    drk += fcik * gijk_d * ex_delr * dcosdrk
    drk -= fcik * gijk * ex_delr_d * rik_hat

    return dri, drj, drk


@numba.njit(cache=True, inline="always")
def _calc_bij(zeta: float, beta: float, n: float) -> float:
    tmp = beta * zeta
    return (1.0 + tmp**n) ** (-1.0 / (2.0 * n))


@numba.njit(cache=True, inline="always")
def _calc_bij_d(zeta: float, beta: float, n: float) -> float:
    tmp = beta * zeta
    return -0.5 * (1.0 + tmp**n) ** (-1.0 - (1.0 / (2.0 * n))) * (beta * tmp ** (n - 1.0))


@numba.njit(cache=True)
def _calculate_tersoff(
    atom_species: np.ndarray,
    species_cutoffs: np.ndarray,
    param_table: np.ndarray,
    neighbor_starts: np.ndarray,
    neighbor_indices: np.ndarray,
    vectors: np.ndarray,
    distances: np.ndarray,
    natoms: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    energies = np.zeros(natoms)
    forces = np.zeros((natoms, 3))
    virial = np.zeros((3, 3))

    for i in range(natoms):
        type_i = atom_species[i]
        start = neighbor_starts[i]
        end = neighbor_starts[i + 1]
        for p in range(start, end):
            j = neighbor_indices[p]
            type_j = atom_species[j]
            params = param_table[type_i, type_j, type_j]
            rij = vectors[p]
            abs_rij = distances[p]
            if abs_rij <= 0.0:
                continue

            fc = _calc_fc(abs_rij, params[10], params[11])
            if fc == 0.0:
                continue

            rij_hat = rij / abs_rij
            zeta = 0.0

            for q in range(start, end):
                if q == p:
                    continue
                k = neighbor_indices[q]
                type_k = atom_species[k]
                trip = param_table[type_i, type_j, type_k]
                abs_rik = distances[q]
                if abs_rik > trip[10] + trip[11]:
                    continue

                rik = vectors[q]
                costheta = (
                    rij_hat[0] * rik[0] / abs_rik
                    + rij_hat[1] * rik[1] / abs_rik
                    + rij_hat[2] * rik[2] / abs_rik
                )
                fc_ik = _calc_fc(abs_rik, trip[10], trip[11])
                g_theta = _calc_gijk(costheta, trip[1], trip[3], trip[4], trip[5])

                arg = (trip[2] * (abs_rij - abs_rik)) ** trip[0]
                if arg > _MAX_EXP_ARG:
                    ex_delr = 1.0e30
                elif arg < _MIN_EXP_ARG:
                    ex_delr = 0.0
                else:
                    ex_delr = np.exp(arg)

                zeta += fc_ik * g_theta * ex_delr

            bij = _calc_bij(zeta, params[7], params[6])
            bij_d = _calc_bij_d(zeta, params[7], params[6])
            repulsive = params[13] * np.exp(-params[12] * abs_rij)
            attractive = -params[9] * np.exp(-params[8] * abs_rij)
            pair_energy = 0.25 * fc * (repulsive + bij * attractive)

            energies[i] += pair_energy
            energies[j] += pair_energy

            dfc = _calc_fc_d(abs_rij, params[10], params[11])
            rep_deriv = -params[12] * repulsive
            att_deriv = -params[8] * attractive
            tmp = dfc * (repulsive + bij * attractive)
            tmp += fc * (rep_deriv + bij * att_deriv)

            grad = 0.5 * tmp * rij_hat
            forces[i] += grad
            forces[j] -= grad
            virial += np.outer(grad, rij)

            for q in range(start, end):
                if q == p:
                    continue
                k = neighbor_indices[q]
                type_k = atom_species[k]
                trip = param_table[type_i, type_j, type_k]
                abs_rik = distances[q]
                if abs_rik > trip[10] + trip[11]:
                    continue
                rik = vectors[q]
                dztdri, dztdrj, dztdrk = _calc_zeta_d(rij, rik, trip)
                attractive_trip = -trip[9] * np.exp(-trip[8] * abs_rij)
                gradi = 0.5 * fc * bij_d * dztdri * attractive_trip
                gradj = 0.5 * fc * bij_d * dztdrj * attractive_trip
                gradk = 0.5 * fc * bij_d * dztdrk * attractive_trip
                forces[i] -= gradi
                forces[j] -= gradj
                forces[k] -= gradk
                virial += np.outer(gradj, rij)
                virial += np.outer(gradk, rik)

    return energies.sum(), energies, forces, virial


class NumbaTersoffCalculator(Calculator):
    """JIT-accelerated Tersoff potential matching ASE's reference behavior."""

    implemented_properties = [
        "free_energy",
        "energy",
        "energies",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        parameters: dict[tuple[str, str, str], TersoffParameters | list[float] | tuple[float, ...]] | TersoffData,
        skin: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cutoff_skin = float(skin)
        if isinstance(parameters, TersoffData):
            self.data = parameters
        else:
            self.data = TersoffData.from_parameter_dict(parameters, cutoff_skin=self.cutoff_skin)
        self._init_cache()

    @classmethod
    def from_lammps(
        cls,
        potential_file: str | Path,
        skin: float = 0.3,
        **kwargs: Any,
    ) -> "NumbaTersoffCalculator":
        parameters = cls.read_lammps_format(potential_file)
        return cls(parameters=parameters, skin=skin, **kwargs)

    @staticmethod
    def read_lammps_format(
        potential_file: str | Path,
    ) -> dict[tuple[str, str, str], TersoffParameters]:
        block_size = 17
        with Path(potential_file).open("r", encoding="utf-8") as fd:
            content = (
                "".join([line for line in fd if not line.strip().startswith("#")])
                .replace("\n", " ")
                .split()
            )

        if len(content) % block_size != 0:
            raise ValueError("The potential file does not have the correct LAMMPS format.")

        parameters: dict[tuple[str, str, str], TersoffParameters] = {}
        for i in range(0, len(content), block_size):
            block = content[i : i + block_size]
            key = (block[0], block[1], block[2])
            params = TersoffParameters.from_list([float(value) for value in block[3:]])
            parameters[key] = params
        return parameters

    def _init_cache(self) -> None:
        if self.data.species_count == 0:
            raise ValueError("TersoffData requires a non-empty species list.")
        self._species_index = {symbol: idx for idx, symbol in enumerate(self.data.species)}
        self._species_cutoffs = np.zeros(self.data.species_count, dtype=float)
        table = np.asarray(self.data.parameter_table, dtype=float)
        for i in range(self.data.species_count):
            max_cutoff = 0.0
            for j in range(self.data.species_count):
                for k in range(self.data.species_count):
                    params = table[i, j, k]
                    max_cutoff = max(max_cutoff, float(params[10] + params[11]))
            self._species_cutoffs[i] = max_cutoff
        self._max_cutoff = float(np.max(self._species_cutoffs))
        if self._max_cutoff <= 0.0:
            raise ValueError("Tersoff cutoff must be positive.")
        self.nl = None

    def update(self, parameters: dict[tuple[str, str, str], TersoffParameters] | TersoffData) -> None:
        if isinstance(parameters, TersoffData):
            self.data = parameters
        else:
            self.data = TersoffData.from_parameter_dict(parameters, cutoff_skin=self.cutoff_skin)
        self._init_cache()

    def _update_nl(self, atoms) -> None:
        cutoffs = np.array([self._species_cutoffs[self._species_index[str(sym)]] for sym in atoms.symbols], dtype=float)
        self.nl = NeighborList(
            cutoffs,
            skin=self.cutoff_skin,
            self_interaction=False,
            bothways=True,
        )
        self.nl.update(atoms)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):  # noqa: B006
        Calculator.calculate(self, atoms, properties, system_changes)
        checks = {"positions", "numbers", "cell", "pbc"}
        if any(change in checks for change in system_changes) or not hasattr(self, "nl") or self.nl is None:
            self._update_nl(self.atoms)

        natoms = len(self.atoms)
        atom_species = np.array([self._species_index[str(symbol)] for symbol in self.atoms.symbols], dtype=np.int64)

        neighbor_indices: list[int] = []
        neighbor_starts = np.zeros(natoms + 1, dtype=np.int64)
        vectors_list: list[np.ndarray] = []
        distances_list: list[float] = []

        for i in range(natoms):
            indices, offsets = self.nl.get_neighbors(i)
            pos = self.atoms.positions[indices]
            vecs = pos + offsets @ self.atoms.cell.array - self.atoms.positions[i]
            dists = np.sqrt(np.sum(vecs * vecs, axis=1))
            neighbor_starts[i + 1] = neighbor_starts[i] + len(indices)
            neighbor_indices.extend(int(idx) for idx in indices)
            vectors_list.extend(np.asarray(vec, dtype=float) for vec in vecs)
            distances_list.extend(float(dist) for dist in dists)

        if neighbor_indices:
            neighbor_indices_arr = np.asarray(neighbor_indices, dtype=np.int64)
            vectors = np.asarray(vectors_list, dtype=np.float64)
            distances = np.asarray(distances_list, dtype=np.float64)
            energy, local, forces, virial = _calculate_tersoff(
                atom_species,
                self._species_cutoffs,
                np.asarray(self.data.parameter_table, dtype=np.float64),
                neighbor_starts,
                neighbor_indices_arr,
                vectors,
                distances,
                natoms,
            )
        else:
            energy = 0.0
            local = np.zeros(natoms, dtype=float)
            forces = np.zeros((natoms, 3), dtype=float)
            virial = np.zeros((3, 3), dtype=float)

        self.results["energy"] = float(energy)
        self.results["free_energy"] = float(energy)
        self.results["energies"] = np.asarray(local, dtype=float)
        self.results["forces"] = np.asarray(forces, dtype=float)

        if self.atoms.cell.rank == 3 and self.atoms.get_volume() != 0.0:
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / self.atoms.get_volume())
