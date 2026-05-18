"""Module for `Optimizer` classes."""

from forgeff.optimizers.base import ParallelOptimizerBase
from forgeff.optimizers.ga import GeneticAlgorithmOptimizer
from forgeff.optimizers.anneal import SimulatedAnnealingOptimizer
from forgeff.optimizers.ideal import NoInteractionOptimizer
from forgeff.optimizers.randomizer import Randomizer
from forgeff.optimizers.scipy import (
    ScipyDifferentialEvolutionOptimizer,
    ScipyDualAnnealingOptimizer,
    ScipyMinimizeOptimizer,
)


def make_optimizer(optimizer: str) -> type[ParallelOptimizerBase]:
    """Make an `Optimizer` class.

    Returns
    -------
    type[ParallelOptimizerBase]

    """
    return {
        "GA": GeneticAlgorithmOptimizer,
        "SA": SimulatedAnnealingOptimizer,
        "anneal": SimulatedAnnealingOptimizer,
        "NI": NoInteractionOptimizer,
        "minimize": ScipyMinimizeOptimizer,
        "DA": ScipyDualAnnealingOptimizer,
        "DE": ScipyDifferentialEvolutionOptimizer,
        "randomize": Randomizer,
    }[optimizer]

__all__ = [
    "GeneticAlgorithmOptimizer",
    "SimulatedAnnealingOptimizer",
    "NoInteractionOptimizer",
    "ParallelOptimizerBase",
    "Randomizer",
    "ScipyDifferentialEvolutionOptimizer",
    "ScipyDualAnnealingOptimizer",
    "ScipyMinimizeOptimizer",
    "make_optimizer",
]
