"""Numba-backed analytical pair potential for ASE fitting."""

from __future__ import annotations

from typing import Any

try:
    import numba
except ModuleNotFoundError as exc:
    raise RuntimeError("no numba") from exc
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress

from forgeff.potentials.ase.forms import get_form_spec


PAIR_FORM_IDS = {
    "lj": 0,
    "bornmayer": 1,
    "morse": 2,
    "double_morse": 3,
    "power_decay": 4,
    "exp_decay": 5,
    "constant": 6,
    "coul": 7,
    "exponential": 8,
    "hbnd": 9,
    "buck": 10,
    "eopp": 11,
    "csw": 12,
    "csw2": 13,
    "ms": 14,
    "born": 15,
    "softshell": 16,
    "exp_plus": 17,
    "mexp_decay": 18,
    "strmm": 19,
    "poly_5": 20,
    "zero": 21,
}


@numba.njit(cache=False, inline="always")
def _pair_eval(form_id, r, p):
    inv_r = 1.0 / r

    if form_id == 0:  # lj
        sigma = p[1]
        eps = p[0]
        sr = sigma * inv_r
        sr2 = sr * sr
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        energy = 4.0 * eps * (sr12 - sr6)
        ddr = 4.0 * eps * (-12.0 * sr12 + 6.0 * sr6) * inv_r
        return energy, ddr

    if form_id == 1:  # bornmayer
        a = p[0]
        rho = p[1]
        e = np.exp(-r / rho)
        energy = a * e
        ddr = -a * e / rho
        return energy, ddr

    if form_id == 2:  # morse
        de = p[0]
        a = p[1]
        re = p[2]
        x = r - re
        e1 = np.exp(-a * x)
        e2 = np.exp(-2.0 * a * x)
        energy = de * (e2 - 2.0 * e1)
        ddr = 2.0 * a * de * (e1 - e2)
        return energy, ddr

    if form_id == 3:  # double_morse
        e1 = p[0]
        a1 = p[1]
        r1 = p[2]
        e2 = p[3]
        a2 = p[4]
        r2 = p[5]
        delta = p[6]
        x1 = r - r1
        x2 = r - r2
        t11 = np.exp(-a1 * x1)
        t12 = np.exp(-2.0 * a1 * x1)
        t21 = np.exp(-a2 * x2)
        t22 = np.exp(-2.0 * a2 * x2)
        energy = e1 * (t12 - 2.0 * t11) + e2 * (t22 - 2.0 * t21) + delta
        ddr = 2.0 * a1 * e1 * (t11 - t12) + 2.0 * a2 * e2 * (t21 - t22)
        return energy, ddr

    if form_id == 4:  # power_decay
        alpha = p[0]
        beta = p[1]
        energy = alpha * r ** (-beta)
        ddr = -beta * energy * inv_r
        return energy, ddr

    if form_id == 5:  # exp_decay
        alpha = p[0]
        beta = p[1]
        e = np.exp(-beta * r)
        energy = alpha * e
        ddr = -alpha * beta * e
        return energy, ddr

    if form_id == 6:  # constant
        return p[0], 0.0

    if form_id == 7:  # coul
        q1 = p[0]
        q2 = p[1]
        energy = 14.3996454784255 * q1 * q2 * inv_r
        ddr = -energy * inv_r
        return energy, ddr

    if form_id == 8:  # exponential
        a = p[0]
        n = p[1]
        energy = a * r**n
        ddr = a * n * r ** (n - 1.0)
        return energy, ddr

    if form_id == 9:  # hbnd
        a = p[0]
        b = p[1]
        energy = a * r ** (-12.0) - b * r ** (-10.0)
        ddr = -12.0 * a * r ** (-13.0) + 10.0 * b * r ** (-11.0)
        return energy, ddr

    if form_id == 10:  # buck
        a = p[0]
        rho = p[1]
        c = p[2]
        e = np.exp(-r / rho)
        energy = a * e - c * r ** (-6.0)
        ddr = -a * e / rho + 6.0 * c * r ** (-7.0)
        return energy, ddr

    if form_id == 11:  # eopp
        c1 = p[0]
        eta1 = p[1]
        c2 = p[2]
        eta2 = p[3]
        k = p[4]
        phi = p[5]
        ang = k * r + phi
        r1 = r ** (-eta1)
        r2 = r ** (-eta2)
        energy = c1 * r1 + c2 * r2 * np.cos(ang)
        ddr = (
            -c1 * eta1 * r ** (-eta1 - 1.0)
            + c2 * (-eta2 * r ** (-eta2 - 1.0) * np.cos(ang) - k * r2 * np.sin(ang))
        )
        return energy, ddr

    if form_id == 12:  # csw
        c1 = p[0]
        c2 = p[1]
        k = p[2]
        power = p[3]
        kr = k * r
        num = 1.0 + c1 * np.cos(kr) + c2 * np.sin(kr)
        inv_r_pow = r ** (-power)
        energy = num * inv_r_pow
        num_prime = -c1 * k * np.sin(kr) + c2 * k * np.cos(kr)
        ddr = num_prime * inv_r_pow - power * energy * inv_r
        return energy, ddr

    if form_id == 13:  # csw2
        c1 = p[0]
        k = p[1]
        phi = p[2]
        power = p[3]
        ang = k * r + phi
        num = 1.0 + c1 * np.cos(ang)
        inv_r_pow = r ** (-power)
        energy = num * inv_r_pow
        num_prime = -c1 * k * np.sin(ang)
        ddr = num_prime * inv_r_pow - power * energy * inv_r
        return energy, ddr

    if form_id == 14:  # ms
        de = p[0]
        a = p[1]
        r0 = p[2]
        x = 1.0 - r / r0
        e1 = np.exp(a * x)
        e2 = np.exp(0.5 * a * x)
        energy = de * (e1 - 2.0 * e2)
        ddr = de * a / r0 * (e2 - e1)
        return energy, ddr

    if form_id == 15:  # born
        a = p[0]
        sigma = p[1]
        r0 = p[2]
        c = p[3]
        d = p[4]
        e = np.exp((r0 - r) / sigma)
        energy = a * e - c * r ** (-6.0) + d * r ** (-8.0)
        ddr = -a * e / sigma + 6.0 * c * r ** (-7.0) - 8.0 * d * r ** (-9.0)
        return energy, ddr

    if form_id == 16:  # softshell
        alpha = p[0]
        beta = p[1]
        energy = (alpha * inv_r) ** beta
        ddr = -beta * energy * inv_r
        return energy, ddr

    if form_id == 17:  # exp_plus
        alpha = p[0]
        beta = p[1]
        c = p[2]
        e = np.exp(-beta * r)
        energy = alpha * e + c
        ddr = -alpha * beta * e
        return energy, ddr

    if form_id == 18:  # mexp_decay
        alpha = p[0]
        beta = p[1]
        r0 = p[2]
        e = np.exp(-beta * (r - r0))
        energy = alpha * e
        ddr = -alpha * beta * e
        return energy, ddr

    if form_id == 19:  # strmm
        alpha = p[0]
        beta = p[1]
        gamma = p[2]
        delta = p[3]
        r0 = p[4]
        x = r - r0
        e1 = np.exp(-0.5 * beta * x)
        e2 = np.exp(-delta * x)
        energy = 2.0 * alpha * e1 - gamma * (1.0 + delta * x) * e2
        ddr = -alpha * beta * e1 + gamma * delta * delta * x * e2
        return energy, ddr

    if form_id == 20:  # poly_5
        p0 = p[0]
        p1 = p[1]
        p2 = p[2]
        p3 = p[3]
        p4 = p[4]
        x = r - 1.0
        x2 = x * x
        x3 = x2 * x
        x4 = x3 * x
        x5 = x4 * x
        energy = p0 + 0.5 * p1 * x2 + p2 * x3 + p3 * x4 + p4 * x5
        ddr = p1 * x + 3.0 * p2 * x2 + 4.0 * p3 * x3 + 5.0 * p4 * x4
        return energy, ddr

    if form_id == 21:  # zero
        return 0.0, 0.0

    return 0.0, 0.0


@numba.njit(cache=False)
def _calculate_pair(form_id, params, i_list, j_list, dist, rvec, natoms):
    energy = 0.0
    local = np.zeros(natoms)
    forces = np.zeros((natoms, 3))
    virial = np.zeros((3, 3))

    for k in range(dist.shape[0]):
        r = dist[k]
        if r <= 0.0:
            continue
        pair_energy, ddr = _pair_eval(form_id, r, params)
        i = i_list[k]
        j = j_list[k]
        rx = rvec[k, 0]
        ry = rvec[k, 1]
        rz = rvec[k, 2]
        inv_r = 1.0 / r
        fx = ddr * rx * inv_r
        fy = ddr * ry * inv_r
        fz = ddr * rz * inv_r

        energy += pair_energy
        local[i] += 0.5 * pair_energy
        local[j] += 0.5 * pair_energy
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[i, 2] += fz
        forces[j, 0] -= fx
        forces[j, 1] -= fy
        forces[j, 2] -= fz
        virial[0, 0] += rx * fx
        virial[0, 1] += rx * fy
        virial[0, 2] += rx * fz
        virial[1, 0] += ry * fx
        virial[1, 1] += ry * fy
        virial[1, 2] += ry * fz
        virial[2, 0] += rz * fx
        virial[2, 1] += rz * fy
        virial[2, 2] += rz * fz

    return energy, local, forces, virial


@numba.njit(cache=False)
def _calculate_pair_multispecies(form_id, pair_index_map, pair_params, types, i_list, j_list, dist, rvec, natoms):
    energy = 0.0
    local = np.zeros(natoms)
    forces = np.zeros((natoms, 3))
    virial = np.zeros((3, 3))

    for k in range(dist.shape[0]):
        r = dist[k]
        if r <= 0.0:
            continue
        i = i_list[k]
        j = j_list[k]
        pair_idx = pair_index_map[types[i], types[j]]
        if pair_idx < 0:
            continue
        pair_energy, ddr = _pair_eval(form_id, r, pair_params[pair_idx])
        rx = rvec[k, 0]
        ry = rvec[k, 1]
        rz = rvec[k, 2]
        inv_r = 1.0 / r
        fx = ddr * rx * inv_r
        fy = ddr * ry * inv_r
        fz = ddr * rz * inv_r

        energy += pair_energy
        local[i] += 0.5 * pair_energy
        local[j] += 0.5 * pair_energy
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[i, 2] += fz
        forces[j, 0] -= fx
        forces[j, 1] -= fy
        forces[j, 2] -= fz
        virial[0, 0] += rx * fx
        virial[0, 1] += rx * fy
        virial[0, 2] += rx * fz
        virial[1, 0] += ry * fx
        virial[1, 1] += ry * fy
        virial[1, 2] += ry * fz
        virial[2, 0] += rz * fx
        virial[2, 1] += rz * fy
        virial[2, 2] += rz * fz

    return energy, local, forces, virial


class NumbaPairPotential(Calculator):
    """JIT-accelerated built-in analytical pair potential."""

    implemented_properties = (
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    )

    def __init__(
        self,
        *,
        form: str,
        cutoff: float | None = None,
        rc: float | None = None,
        parameter_names: list[str] | tuple[str, ...] | None = None,
        pair_terms: list[dict[str, Any]] | None = None,
        species: list[int] | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(form=form, cutoff=cutoff, rc=rc, parameter_names=parameter_names, **kwargs)
        self.form = str(form).lower()
        if self.form not in PAIR_FORM_IDS:
            raise ValueError(f"Unknown built-in pair form {form!r}.")

        self.form_id = PAIR_FORM_IDS[self.form]
        self.cutoff = cutoff if cutoff is not None else rc
        if self.cutoff is None:
            raise ValueError("NumbaPairPotential requires a finite 'cutoff' (or 'rc').")

        spec = get_form_spec(self.form)
        if parameter_names is None:
            parameter_names = spec["params"]
        self.parameter_names = list(parameter_names)
        self._parameter_index = {name: idx for idx, name in enumerate(self.parameter_names)}

        self.multispecies = bool(pair_terms)
        if self.multispecies:
            if any(term.get("expression") is not None for term in pair_terms or []):
                raise ValueError("Multispecies Numba pair potentials currently support built-in forms only.")
            if species is None:
                raise ValueError("Multispecies pair potential requires an explicit species order.")
            self.species_numbers = np.array(
                [atomic_numbers.get(str(item), int(item)) if not isinstance(item, int) else int(item) for item in species],
                dtype=np.int32,
            )
            self._pair_index_map = np.full((119, 119), -1, dtype=np.int32)
            self._pair_params = np.zeros((len(pair_terms), len(self.parameter_names)), dtype=np.float64)
            for idx, term in enumerate(pair_terms):
                pair_species = term["species"]
                i, j = int(pair_species[0]), int(pair_species[1])
                self._pair_index_map[i, j] = idx
                self._pair_index_map[j, i] = idx
                prefix = term["prefix"]
                for pidx, name in enumerate(self.parameter_names):
                    self._pair_params[idx, pidx] = float(kwargs[f"{prefix}_{name}"])
            return

        missing = [name for name in self.parameter_names if name not in kwargs]
        if missing:
            raise ValueError(f"Missing values for pair parameters: {missing}")

        self._parameter_values = np.array([float(kwargs[name]) for name in self.parameter_names], dtype=float)

    def update_parameters(self, **kwargs: Any) -> None:
        for name, value in kwargs.items():
            idx = self._parameter_index.get(name)
            if idx is not None:
                self._parameter_values[idx] = float(value)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):  # noqa: B006
        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)
        i_list, j_list, shifts, dist = neighbor_list("ijSd", self.atoms, float(self.cutoff))
        if len(i_list):
            unique = i_list < j_list
            i_list = i_list[unique]
            j_list = j_list[unique]
            shifts = shifts[unique]
            dist = dist[unique]
            rvec = self.atoms.positions[j_list] + shifts @ self.atoms.cell.array - self.atoms.positions[i_list]
            if self.multispecies:
                types = self.atoms.get_atomic_numbers().astype(np.int64)
                energy, local, forces, virial = _calculate_pair_multispecies(
                    self.form_id,
                    self._pair_index_map,
                    np.asarray(self._pair_params, dtype=np.float64),
                    types,
                    i_list.astype(np.int64),
                    j_list.astype(np.int64),
                    np.asarray(dist, dtype=np.float64),
                    np.asarray(rvec, dtype=np.float64),
                    natoms,
                )
            else:
                energy, local, forces, virial = _calculate_pair(
                    self.form_id,
                    np.asarray(self._parameter_values, dtype=np.float64),
                    i_list.astype(np.int64),
                    j_list.astype(np.int64),
                    np.asarray(dist, dtype=np.float64),
                    np.asarray(rvec, dtype=np.float64),
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
