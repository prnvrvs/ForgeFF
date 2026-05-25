"""`forgeff train`."""

import logging
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

from forgeff.io import read_potential, write_potential
from forgeff.io.utils import get_dummy_species, read_images, set_potential_species
from forgeff.loss import ErrorPrinter, LossFunction, LossFunctionBase, LossSetting
from forgeff.optimizers import make_optimizer
from forgeff.parallel import DummyMPIComm, is_master, world
from forgeff.utils import measure_time

from .setting import load_setting_train

if TYPE_CHECKING:
    from forgeff.optimizers.base import OptimizerBase

logger = logging.getLogger(__name__)


def _normalize_species_label(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(chemical_symbols[int(value)])
    return str(value)


def _species_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, np.integer)):
        return [_normalize_species_label(value)]
    try:
        items = list(value)
    except TypeError:
        return [_normalize_species_label(value)]
    return [_normalize_species_label(item) for item in items]


def _validate_potential_species_order(training_species: list[Any], potential_species: Any) -> None:
    training_labels = [_normalize_species_label(item) for item in training_species]
    potential_labels = _species_labels(potential_species)
    if not potential_labels:
        return
    if training_labels != potential_labels:
        raise ValueError(
            "Training species order does not match the initial potential species order. "
            f"Training species = {training_labels!r}, initial potential species = {potential_labels!r}. "
            "Update the initial TOML or the training species list so they match exactly."
        )


class Trainer:
    """Trainer."""

    def __init__(
        self,
        pot_data: Any,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        engine: str = "numpy",
        loss: dict | LossSetting | None = None,
        steps: list[dict] | None = None,
        *,
        update_mindist: bool = False,
        relax_magmoms: bool | None = None,
        comm: DummyMPIComm = world,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        pot_data : Any
            Potential data object (e.g. EAMData, ADPData, ASEData).
        seed : int | None (optional)
            Seed for the random number generator. Disregarded if `rng` is given.
        rng : np.random.Generator | None (optional)
            Pseudo-random-number generator (PRNG) with the NumPy API.
        engine : str (optional)
            Engine name.
        loss : dict | LossSetting | None (optional)
            Dict with settings of the loss function.
        steps : list[dict] | None (optional)
            List of optimization steps.
        comm : MPI.Comm
            MPI.Comm object.
        update_mindist : bool (optional)
            Whether to update min_dist before training.
        relax_magmoms : bool or None (optional)
            Whether to relax magnetic moments.  ``None`` uses mode-based default.

        """
        self.pot_data = pot_data

        seed = seed or comm.bcast(np.random.SeedSequence().entropy % (2**32), root=0)
        if seed is not None and is_master(comm):
            logger.info("[random seed] = %d", seed)
        self.rng = rng or np.random.default_rng(seed)

        self.engine = engine
        self.loss = LossSetting.from_any(loss)
        self.steps = steps or [{"method": "minimize"}]
        self.comm = comm
        self.should_update_mindist = update_mindist
        self.relax_magmoms = relax_magmoms

    def update_mindist(self, images: list[Atoms]) -> None:
        """Update min_dist of the potential."""
        if hasattr(self.pot_data, "min_dist"):
            mindist = np.inf
            for atoms in images:
                if len(atoms) < 2:
                    continue
                distances = atoms.get_all_distances(mic=True)
                np.fill_diagonal(distances, np.inf)
                current = float(np.min(distances))
                if np.isfinite(current):
                    mindist = min(mindist, current)
            if np.isfinite(mindist):
                self.pot_data.min_dist = mindist

    def train(self, images: list[Atoms]) -> LossFunctionBase:
        """Train.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects.

        Returns
        -------
        loss : LossFunctionBase
            LossFunction object after training.

        """
        if self.should_update_mindist:
            self.update_mindist(images)

        loss_args = (images, self.pot_data, self.loss)
        loss = LossFunction(
            *loss_args,
            engine=self.engine,
            relax_magmoms=self.relax_magmoms,
            comm=self.comm,
        )

        for i, step in enumerate(self.steps):
            with measure_time(f"step {i}: {step['method']}", comm=self.comm):
                if is_master(self.comm):
                    logger.info("%s\n", "=" * 72)
                    logger.info(pformat(step))
                    logger.info("")
                    for handler in logger.handlers:
                        handler.flush()

                # Print parameters before optimization.
                if hasattr(self.pot_data, "initialize"):
                    self.pot_data.initialize(self.rng)
                if is_master(self.comm) and hasattr(self.pot_data, "log"):
                    self.pot_data.log()

                # Instantiate an `Optimizer` class
                optimizer_class = make_optimizer(step["method"])
                optimizer = optimizer_class(loss, **step)
                optimizer.optimize(**step.get("kwargs", {}))
                loss.broadcast_results()
                if is_master(self.comm):
                    logger.info("")
                    for handler in logger.handlers:
                        handler.flush()

                    # Print parameters after optimization.
                    if hasattr(self.pot_data, "log"):
                        self.pot_data.log()

                    if hasattr(self.pot_data, "write"):
                        write_potential(f"intermediate_{i}.npy", self.pot_data)

                    ErrorPrinter(loss.images).log()
        return loss



def train_from_setting(filename_setting: str, comm: DummyMPIComm) -> LossFunctionBase:
    """Train."""
    setting = load_setting_train(filename_setting)
    if is_master(comm):
        logger.info(pformat(setting))
        logger.info("")
        for handler in logger.handlers:
            handler.flush()

    untrained_potential = str(Path(setting.potentials.initial).resolve())

    species = setting.common.species or None
    images = read_images(
        setting.configurations.training,
        species=species,
        comm=comm,
        title="configurations.training",
    )
    if not setting.common.species:
        species = get_dummy_species(images)

    pot_data = read_potential(untrained_potential)
    potential_species = getattr(pot_data, "species", None)
    if not _species_labels(potential_species) and hasattr(pot_data, "calculator_kwargs"):
        potential_species = getattr(pot_data, "calculator_kwargs", {}).get("species")
    _validate_potential_species_order(species, potential_species)
    set_potential_species(pot_data, species)
    if hasattr(pot_data, "engine"):
        pot_data.engine = setting.common.engine

    trainer = Trainer(
        pot_data,
        seed=setting.common.seed,
        engine=setting.common.engine,
        loss=setting.loss,
        steps=setting.steps,
        update_mindist=setting.update_mindist,
        relax_magmoms=setting.common.relax_magmoms,
        comm=comm,
    )
    loss = trainer.train(images)

    if is_master(comm):
        logger.info("%s\n", "=" * 72)
        write_potential(setting.potentials.final, pot_data)
    return loss
