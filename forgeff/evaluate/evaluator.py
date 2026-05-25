"""`forgeff evaluate` command."""

import logging
from copy import copy
from pathlib import Path
from pprint import pformat

from ase import Atoms

import forgeff.io
from forgeff.calculator import make_calculator
from forgeff.io import read_potential
from forgeff.io.utils import get_dummy_species, read_images, set_potential_species
from forgeff.loss import ErrorPrinter
from forgeff.parallel import DummyMPIComm, is_master, world
from typing import Any

from .setting import load_setting_evaluate

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for potential data on test configurations."""

    def __init__(
        self,
        pot_data: Any,
        engine: str = "numpy",
        *,
        relax_magmoms: bool | None = None,
        comm: DummyMPIComm = world,
    ) -> None:
        """Initialize Evaluator.

        Parameters
        ----------
        pot_data : potential data
            Potential data.
        engine : str
            Engine to use for calculations ("numpy", "numba", etc.).
        relax_magmoms : bool or None
            Whether to relax magnetic moments.  ``None`` uses mode-based default.
        comm : MPI.Comm
            MPI communicator.

        """
        self.pot_data = pot_data
        self.engine = engine
        self.relax_magmoms = relax_magmoms
        self.comm = comm

    def evaluate(self, images: list[Atoms]) -> list[Atoms]:
        """Run potential calculations on images.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects with targets stored in `atoms.calc.targets`.

        Returns
        -------
        list[Atoms]
            Images with computed results from the potential.

        """
        # Create shallow copies to preserve originals
        images_eval = [copy(_) for _ in images]

        for i, atoms in enumerate(images_eval):
            # Save targets before replacing calculator
            targets = atoms.calc.results if atoms.calc else {}
            atoms.calc = make_calculator(
                self.pot_data,
                engine=self.engine,
                form=getattr(self.pot_data, "form", None),
                relax_magmoms=self.relax_magmoms,
            )
            atoms.calc.targets = targets

            # Special handling of magmoms, since they can be both results and input
            if "magmoms" in targets:
                atoms.set_initial_magnetic_moments(targets["magmoms"])

            energy = atoms.get_potential_energy()
            if is_master(self.comm):
                logger.info("configuration %d: %s", i, energy)

        return images_eval


def evaluate_from_setting(filename_setting: str, comm: DummyMPIComm) -> None:
    """Evaluate the potential on data from a setting file and print errors.

    Parameters
    ----------
    filename_setting : str
        Path to the setting file.
    comm : MPI.Comm
        MPI communicator.

    """
    setting = load_setting_evaluate(filename_setting)
    if is_master(comm):
        logger.info(pformat(setting))
        logger.info("")
        for handler in logger.handlers:
            handler.flush()

    potential_file = str(Path(setting.potentials.final).resolve())

    species = setting.common.species or None
    images_initial = read_images(
        setting.configurations.initial,
        species=species,
        comm=comm,
        title="configurations.initial",
    )
    if not setting.common.species:
        species = get_dummy_species(images_initial)

    pot_data = read_potential(potential_file)
    set_potential_species(pot_data, species)
    if hasattr(pot_data, "engine"):
        pot_data.engine = setting.common.engine

    # Run evaluation
    evaluator = Evaluator(
        pot_data,
        engine=setting.common.engine,
        relax_magmoms=setting.common.relax_magmoms,
        comm=comm,
    )
    images_final = evaluator.evaluate(images_initial)

    # Print errors
    if is_master(comm):
        logger.info("%s\n", "=" * 72)
        ErrorPrinter(images_final).log()
        forgeff.io.write(setting.configurations.final[0], images_final)
