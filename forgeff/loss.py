"""Loss function."""

import logging
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress
from scipy.constants import eV

from forgeff.calculator import make_calculator
from forgeff.parallel import DummyMPIComm, master, world
from forgeff.setting import DataclassFromAny

logger = logging.getLogger(__name__)


@dataclass
class LossSetting(DataclassFromAny):
    """Setting of the loss function."""

    energy_weight: float = 1.0
    forces_weight: float = 0.01
    stress_weight: float = 0.001
    energy_per_atom: bool = True
    forces_per_atom: bool = True
    stress_times_volume: bool = True
    energy_per_conf: bool = True
    forces_per_conf: bool = True
    stress_per_conf: bool = True


def _calc_errors_from_diff(diff: np.ndarray) -> dict[str, float]:
    if diff.size == 0:
        return {"N": diff.size, "MAX": np.nan, "ABS": np.nan, "RMS": np.nan}
    return {
        "N": diff.size,
        "MAX": np.max(np.abs(diff)),
        "ABS": np.mean(np.abs(diff)),
        "RMS": np.sqrt(np.mean(diff**2)),
    }


def _format_error_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.6g}"


def format_error_statistics(errors: dict[str, dict[str, float]]) -> str:
    """Format error statistics as a compact ASCII table."""
    rows: list[tuple[str, int, float, float, float]] = []
    for label, key, scale in [
        ("Energy (eV)", "energy", 1.0),
        ("Energy per atom (eV/atom)", "energy_per_atom", 1.0),
        ("Forces per component (eV/angstrom)", "forces", 1.0),
        ("Stress per component (GPa)", "stress", eV * 1e21),
    ]:
        stats = errors.get(key, {})
        if not stats:
            continue
        count = int(stats.get("N", 0))
        rows.append(
            (
                label,
                count,
                float(stats["MAX"]) * scale,
                float(stats["ABS"]) * scale,
                float(stats["RMS"]) * scale,
            )
        )

    headers = ["Metric", "Count", "MAX", "ABS", "RMS"]
    table_rows = [
        [label, str(count), _format_error_value(maxv), _format_error_value(absv), _format_error_value(rmsv)]
        for label, count, maxv, absv, rmsv in rows
    ]

    widths = [len(header) for header in headers]
    for row in table_rows:
        widths[0] = max(widths[0], len(row[0]))
        for i in range(1, len(headers)):
            widths[i] = max(widths[i], len(row[i]))

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                [
                    f"{row[0]:<{widths[0]}}",
                    f"{row[1]:>{widths[1]}}",
                    f"{row[2]:>{widths[2]}}",
                    f"{row[3]:>{widths[3]}}",
                    f"{row[4]:>{widths[4]}}",
                ]
            )
            + " |"
        )

    lines = ["Error statistics:", border(), render(headers), border()]
    lines.extend(render(row) for row in table_rows)
    lines.append(border())
    return "\n".join(lines)


class LossFunctionEnergy:
    """Energy contribution to the loss function."""

    def __init__(
        self,
        images: list[Atoms],
        *,
        pot_data: Any,
        energy_per_atom: bool = False,
        energy_per_conf: bool = True,
        comm: DummyMPIComm = world,
    ) -> None:
        """Initialize."""
        self.images = images
        self.pot_data = pot_data
        self.energy_per_atom = energy_per_atom
        self.energy_per_conf = energy_per_conf
        self.comm = comm

        numbers_of_atoms = np.fromiter(
            (len(atoms) for atoms in images),
            dtype=float,
            count=len(images),
        )
        self.inverse_numbers_of_atoms = np.divide(
            1.0,
            numbers_of_atoms,
            out=np.zeros_like(numbers_of_atoms),
            where=numbers_of_atoms > 0,
        )

        self.configuration_weight = np.ones(len(self.images))

    def calculate(self) -> np.float64:
        """Calculate the contribution to the loss function.

        Returns
        -------
        loss : float
            Energy contribution to the loss function.

        """
        ncnf = len(self.images)
        loss_cnf = 0.0
        for i in range(self.comm.rank, ncnf, self.comm.size):
            atoms = self.images[i]
            target = atoms.calc.targets["energy"]
            result = atoms.calc.results["energy"]
            c = self.configuration_weight[i]
            if self.energy_per_atom:
                c *= self.inverse_numbers_of_atoms[i] ** 2
            loss_cnf += c * (result - target) ** 2
        loss_all = self.comm.allreduce(loss_cnf)
        return loss_all / ncnf if self.energy_per_conf else loss_all

    def jac(self) -> npt.NDArray[np.float64]:
        """Calculate the contribution to the loss function Jacobian.

        Returns
        -------
        jac : npt.NDArray[np.float64]
            Energy contribution to the loss function Jacobian.

        """
        ncnf = len(self.images)
        nprm = self.pot_data.number_of_parameters_optimized
        jac_cnf = np.zeros(nprm)
        jac_all = np.zeros(nprm)
        for i in range(self.comm.rank, ncnf, self.comm.size):
            atoms = self.images[i]
            target = atoms.calc.targets["energy"]
            result = atoms.calc.results["energy"]
            c = self.configuration_weight[i]
            if self.energy_per_atom:
                c *= self.inverse_numbers_of_atoms[i] ** 2
            dedp = atoms.calc.engine.jac_energy(atoms).parameters
            jac_cnf += c * 2.0 * (result - target) * dedp
        self.comm.Allreduce(jac_cnf, jac_all)
        return jac_all / ncnf if self.energy_per_conf else jac_all


class LossFunctionForces:
    """Forces contribution to the loss function.

    Attributes
    ----------
    idcs_frc : npt.NDArray[np.int32]
        Indices of images that have forces.

    """

    def __init__(
        self,
        images: list[Atoms],
        *,
        pot_data: Any,
        forces_per_atom: bool = False,
        forces_per_conf: bool = True,
        comm: DummyMPIComm = world,
    ) -> None:
        """Initialize."""
        self.images = images
        self.pot_data = pot_data
        self.forces_per_atom = forces_per_atom
        self.forces_per_conf = forces_per_conf
        self.comm = comm

        numbers_of_atoms = np.fromiter(
            (len(atoms) for atoms in images),
            dtype=float,
            count=len(images),
        )
        self.inverse_numbers_of_atoms = np.divide(
            1.0,
            numbers_of_atoms,
            out=np.zeros_like(numbers_of_atoms),
            where=numbers_of_atoms > 0,
        )

        self.idcs_frc = np.fromiter(
            (i for i, atoms in enumerate(images) if "forces" in atoms.calc.results),
            dtype=int,
        )

        self.configuration_weight = np.ones(len(self.images))

    def calculate(self) -> np.float64:
        """Calculate the contribution to the loss function.

        Returns
        -------
        loss : float
            Force contribution to the loss function.

        """
        ncnf = len(self.images)
        loss_cnf = 0.0
        for i in range(self.comm.rank, ncnf, self.comm.size):
            if i not in self.idcs_frc:
                continue
            atoms = self.images[i]
            target = atoms.calc.targets["forces"]
            result = atoms.calc.results["forces"]
            c = self.configuration_weight[i]
            if self.forces_per_atom:
                c *= self.inverse_numbers_of_atoms[i]
            loss_cnf += c * np.sum((result - target) ** 2)
        loss_all = self.comm.allreduce(loss_cnf)
        return loss_all / ncnf if self.forces_per_conf else loss_all

    def jac(self) -> npt.NDArray[np.float64]:
        """Calculate the contribution to the loss function Jacobian.

        Returns
        -------
        jac : npt.NDArray[np.float64]
            Force contribution to the loss function Jacobian.

        """
        ncnf = len(self.images)
        jac_cnf = np.zeros(self.pot_data.number_of_parameters_optimized)
        jac_all = np.zeros(self.pot_data.number_of_parameters_optimized)
        for i in range(self.comm.rank, ncnf, self.comm.size):
            if i not in self.idcs_frc:
                continue
            atoms = self.images[i]
            target = atoms.calc.targets["forces"]
            result = atoms.calc.results["forces"]
            c = self.configuration_weight[i]
            if self.forces_per_atom:
                c *= self.inverse_numbers_of_atoms[i]
            dfdp = atoms.calc.engine.jac_forces(atoms).parameters
            jac_cnf += c * 2.0 * np.sum((result - target) * dfdp, axis=(-2, -1))
        self.comm.Allreduce(jac_cnf, jac_all)
        return jac_all / ncnf if self.forces_per_conf else jac_all


class LossFunctionStress:
    """Stress contribution to the loss function.

    Attributes
    ----------
    idcs_str : npt.NDArray[np.int32]
        Indices of images that have 3D cells.

    """

    def __init__(
        self,
        images: list[Atoms],
        pot_data: Any,
        *,
        stress_times_volume: bool = False,
        stress_per_conf: bool = True,
        energy_per_atom: bool = False,
        comm: DummyMPIComm = world,
    ) -> None:
        """Initialize."""
        self.images = images
        self.pot_data = pot_data
        self.stress_times_volume = stress_times_volume
        self.stress_per_conf = stress_per_conf
        self.energy_per_atom = energy_per_atom
        self.comm = comm

        self.idcs_str = np.fromiter(
            (
                i
                for i, atoms in enumerate(images)
                if atoms.cell.rank == 3 and "stress" in atoms.calc.results
            ),
            dtype=int,
        )

        self.volumes = np.fromiter(
            (images[i].cell.volume for i in self.idcs_str),
            dtype=float,
            count=self.idcs_str.size,
        )

        numbers_of_atoms = np.fromiter(
            (len(atoms) for atoms in images),
            dtype=float,
            count=len(images),
        )
        self.inverse_numbers_of_atoms = np.divide(
            1.0,
            numbers_of_atoms,
            out=np.zeros_like(numbers_of_atoms),
            where=numbers_of_atoms > 0,
        )

        self.configuration_weight = np.ones(len(self.images))

    def calculate(self) -> np.float64:
        """Calculate the contribution to the loss function.

        Returns
        -------
        loss : float
            Stress contribution to the loss function.

        """
        ncnf = len(self.images)
        f = voigt_6_to_full_3x3_stress
        loss_cnf = 0.0
        for i in range(self.comm.rank, ncnf, self.comm.size):
            if i not in self.idcs_str:
                continue
            atoms = self.images[i]
            target = atoms.calc.targets["stress"]
            result = atoms.calc.results["stress"]
            c = self.configuration_weight[i]
            if self.stress_times_volume:
                c *= self.volumes[i] ** 2
                if self.energy_per_atom:
                    c *= self.inverse_numbers_of_atoms[i] ** 2
            loss_cnf += c * np.sum((f(target - result)) ** 2)
        loss_all = self.comm.allreduce(loss_cnf)
        return loss_all / ncnf if self.stress_per_conf else loss_all

    def jac(self) -> npt.NDArray[np.float64]:
        """Calculate the contribution to the loss function Jacobian.

        Returns
        -------
        jac : npt.NDArray[np.float64]
            Stress contribution to the loss function Jacobian.

        """
        ncnf = len(self.images)
        f = voigt_6_to_full_3x3_stress
        jac_cnf = np.zeros(self.pot_data.number_of_parameters_optimized)
        jac_all = np.zeros(self.pot_data.number_of_parameters_optimized)
        for i in range(self.comm.rank, ncnf, self.comm.size):
            if i not in self.idcs_str:
                continue
            atoms = self.images[i]
            target = atoms.calc.targets["stress"]
            result = atoms.calc.results["stress"]
            c = self.configuration_weight[i]
            if self.stress_times_volume:
                c *= self.volumes[i] ** 2
                if self.energy_per_atom:
                    c *= self.inverse_numbers_of_atoms[i] ** 2
            dsdp = atoms.calc.engine.jac_stress(atoms).parameters
            jac_cnf += c * 2.0 * np.sum(f(result - target) * dsdp, axis=(-2, -1))
        self.comm.Allreduce(jac_cnf, jac_all)
        return jac_all / len(self.images) if self.stress_per_conf else jac_all
class LossFunctionBase(ABC):
    """Loss function."""

    def __init__(
        self,
        images: list[Atoms],
        pot_data: Any,
        setting: LossSetting,
        *,
        comm: DummyMPIComm = world,
    ) -> None:
        """Loss function.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects for the training dataset.
        pot_data : Any
            Potential data object.
        setting : :class:`forgeff.setting.LossSetting`
            Setting of the loss function.
        comm : MPI.Comm
            MPI.Comm object.

        Notes
        -----
        This class creates a lightweight shallow copy of the provided Atoms
        objects. Atomic positions and arrays are treated as immutable and are
        shared with the input. Only the calculator is replaced internally.

        """
        self.images = [copy(_) for _ in images]
        self.pot_data = pot_data
        self.setting = setting
        self.comm = comm

        self.loss_energy = LossFunctionEnergy(
            self.images,
            pot_data=self.pot_data,
            energy_per_atom=self.setting.energy_per_atom,
            energy_per_conf=self.setting.energy_per_conf,
            comm=self.comm,
        )
        self.loss_forces = LossFunctionForces(
            self.images,
            pot_data=self.pot_data,
            forces_per_atom=self.setting.forces_per_atom,
            forces_per_conf=self.setting.forces_per_conf,
            comm=self.comm,
        )
        self.loss_stress = LossFunctionStress(
            self.images,
            pot_data=self.pot_data,
            stress_times_volume=self.setting.stress_times_volume,
            stress_per_conf=self.setting.stress_per_conf,
            energy_per_atom=self.setting.energy_per_atom,
            comm=self.comm,
        )

    @abstractmethod
    def __call__(self, parameters: npt.NDArray[np.float64]) -> np.float64:
        """Evaluate the loss function."""

    def _run_calculations(self) -> None:
        """Run calculations of the properties."""
        ncnf = len(self.images)
        for i in range(self.comm.rank, ncnf, self.comm.size):
            self.images[i].get_potential_energy()

    def broadcast_results(self) -> None:
        """Broadcast data."""
        size = self.comm.size
        ncnf = len(self.images)
        for i in range(ncnf):
            results = self.images[i].calc.results
            results.update(self.comm.bcast(results, root=i % size))

    def gather_data(self) -> None:
        """Gather data to root process."""
        rank = self.comm.rank
        size = self.comm.size
        ncnf = len(self.images)
        if rank == 0:
            for i in range(ncnf):
                root = i % size
                if root != 0 and hasattr(self.images[i].calc, "engine"):
                    mbd = self.comm.recv(source=root, tag=i + ncnf)
                    self.images[i].calc.engine.mbd = mbd
                    rbd = self.comm.recv(source=root, tag=i + 2 * ncnf)
                    self.images[i].calc.engine.rbd = rbd
        else:
            for i in range(rank, ncnf, size):
                if hasattr(self.images[i].calc, "engine"):
                    self.comm.send(self.images[i].calc.engine.mbd, dest=0, tag=i + ncnf)
                    self.comm.send(
                        self.images[i].calc.engine.rbd, dest=0, tag=i + 2 * ncnf
                    )

    def calc_loss_function(self) -> float:
        """Calculate the value of the loss function.

        Returns
        -------
        float

        """
        self._run_calculations()
        return (
            self.setting.energy_weight * self.loss_energy.calculate()
            + self.setting.forces_weight * self.loss_forces.calculate()
            + self.setting.stress_weight * self.loss_stress.calculate()
        )

    def jac(self, parameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the Jacobian of the loss function.

        Returns
        -------
        npt.NDArray[np.float64]

        """
        jac = self.setting.energy_weight * self.loss_energy.jac()
        if self.loss_forces.idcs_frc.size and self.setting.forces_weight:
            jac += self.setting.forces_weight * self.loss_forces.jac()
        if self.loss_stress.idcs_str.size and self.setting.stress_weight:
            jac += self.setting.stress_weight * self.loss_stress.jac()
        return jac


class ErrorPrinter:
    """Printer of errors for energy, forces, and stress.

    Attributes
    ----------
    idcs_frc : npt.NDArray[np.int32]
        Indices of images that have target forces.
    idcs_str : npt.NDArray[np.int32]
        Indices of images that have target stress.

    """

    def __init__(self, images: list[Atoms]) -> None:
        """Initialize `ErrorPrinter`.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects with calculated results and targets.
            Assumes that targets are stored in `atoms.calc.targets`.

        """
        self.images = images

        # Calculate indices of images with specific properties
        self.idcs_frc = np.fromiter(
            (i for i, atoms in enumerate(images) if "forces" in atoms.calc.targets),
            dtype=int,
        )
        self.idcs_str = np.fromiter(
            (
                i
                for i, atoms in enumerate(images)
                if atoms.cell.rank == 3 and "stress" in atoms.calc.targets and "stress" in atoms.calc.results
            ),
            dtype=int,
        )

    def _calc_errors_energy(self) -> dict[str, float]:
        iterable = (
            atoms.calc.results["energy"] - atoms.calc.targets["energy"]
            for atoms in self.images
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def _calc_errors_energy_per_atom(self) -> dict[str, float]:
        iterable = (
            ((atoms.calc.results["energy"] - atoms.calc.targets["energy"]) / len(atoms))
            for atoms in self.images
            if len(atoms) > 0
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def _calc_errors_forces(self) -> dict[str, float]:
        iterable = (
            self.images[i].calc.results["forces"][j, k]
            - self.images[i].calc.targets["forces"][j, k]
            for i in self.idcs_frc
            for j in range(len(self.images[i]))
            for k in range(3)
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def _calc_errors_stress(self) -> dict[str, float]:
        f = voigt_6_to_full_3x3_stress
        iterable = (
            f(self.images[i].calc.results["stress"])[j, k]
            - f(self.images[i].calc.targets["stress"])[j, k]
            for i in self.idcs_str
            for j in range(3)
            for k in range(3)
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def calculate(self) -> dict[str, dict[str, float]]:
        """Calculate errors.

        The properties should be computed before called.

        Returns
        -------
        dict[str, dict[str, float]]
            Errors for the properties.

        """
        errors = {}
        errors["energy"] = self._calc_errors_energy()
        errors["energy_per_atom"] = self._calc_errors_energy_per_atom()
        errors["forces"] = self._calc_errors_forces()
        errors["stress"] = self._calc_errors_stress()  # eV/Ang^3
        return errors

    def log(self, logger: logging.Logger = logger) -> dict[str, dict[str, float]]:
        """Log errors.

        Returns
        -------
        errors : dict[str, dict[str, float]]
            Errors.

        """
        errors = self.calculate()

        if not master:
            return errors

        logger.info("%s", format_error_statistics(errors))

        return errors


class LossFunction(LossFunctionBase):
    """Loss function."""

    def __init__(
        self,
        *args: tuple,
        engine: str = "numpy",
        relax_magmoms: bool | None = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        for atoms in self.images:
            targets = atoms.calc.results
            atoms.calc = make_calculator(
                self.pot_data,
                engine=self.engine,
                form=getattr(self.pot_data, "form", None),
                mode="train",
                relax_magmoms=relax_magmoms,
            )
            atoms.calc.targets = targets

    def __call__(self, parameters: list[float]) -> float:
        parameters = self.comm.bcast(parameters, root=0)
        self.pot_data.parameters = parameters
        for atoms in self.images:
            atoms.calc.update_parameters(self.pot_data)
        return self.calc_loss_function()
