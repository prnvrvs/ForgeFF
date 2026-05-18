"""Tests for simulated annealing optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forgeff.optimizers.anneal import SimulatedAnnealingOptimizer
from forgeff.parallel import DummyMPIComm


@dataclass
class _PotData:
    parameters: np.ndarray

    def get_bounds(self) -> list[tuple[float, float]]:
        return [(-20.0, 20.0)] * len(self.parameters)


class _QuadraticLoss:
    def __init__(self, initial: np.ndarray, target: np.ndarray) -> None:
        self.pot_data = _PotData(parameters=np.asarray(initial, dtype=float))
        self.target = np.asarray(target, dtype=float)
        self.comm = DummyMPIComm()

    def __call__(self, parameters: np.ndarray | None) -> float:
        if parameters is not None:
            self.pot_data.parameters = np.asarray(parameters, dtype=float)
        diff = self.pot_data.parameters - self.target
        return float(np.sum(diff * diff))


def test_simulated_annealing_reduces_quadratic_loss() -> None:
    initial = np.array([10.0, -10.0, 5.0], dtype=float)
    target = np.array([1.0, -2.0, 0.5], dtype=float)
    loss = _QuadraticLoss(initial, target)

    optimizer = SimulatedAnnealingOptimizer(loss)

    initial_loss = loss(loss.pot_data.parameters)
    optimizer.optimize(
        seed=7,
        anneal_temp=5.0,
        steps_per_temperature=10,
        maxiter=30,
        cooling=0.9,
        step_scale=0.4,
    )

    final_parameters = loss.pot_data.parameters
    final_loss = loss(final_parameters)

    assert final_loss < initial_loss
    assert np.linalg.norm(final_parameters - target) < np.linalg.norm(initial - target)
