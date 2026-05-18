"""Simulated annealing optimizer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize._optimize import OptimizeResult

from forgeff.optimizers.base import ParallelOptimizerBase
from forgeff.optimizers.scipy import Callback

logger = logging.getLogger(__name__)


def _clip_to_bounds(values: np.ndarray, bounds: list[tuple[float, float]] | None) -> np.ndarray:
    if not bounds:
        return values
    clipped = values.copy()
    for i, (lb, ub) in enumerate(bounds):
        if np.isfinite(lb):
            clipped[i] = max(clipped[i], lb)
        if np.isfinite(ub):
            clipped[i] = min(clipped[i], ub)
    return clipped


def _step_sizes_from_bounds(
    parameters: np.ndarray,
    bounds: list[tuple[float, float]] | None,
    step_scale: float,
) -> np.ndarray:
    if bounds:
        step_sizes = []
        for value, (lb, ub) in zip(parameters, bounds, strict=True):
            if np.isfinite(lb) and np.isfinite(ub) and ub > lb:
                step_sizes.append(max(step_scale * (ub - lb), 1e-12))
            else:
                step_sizes.append(max(step_scale * max(abs(value), 1.0), 1e-12))
        return np.asarray(step_sizes, dtype=float)
    return np.maximum(step_scale * np.maximum(np.abs(parameters), 1.0), 1e-12)


class SimulatedAnnealingOptimizer(ParallelOptimizerBase):
    """Potfit-style simulated annealing optimizer."""

    @property
    def optimized_default(self) -> list[str]:
        return []

    @property
    def optimized_allowed(self) -> list[str]:
        return []

    def _estimate_temperature(
        self,
        parameters: np.ndarray,
        current_loss: float,
        step_sizes: np.ndarray,
        bounds: list[tuple[float, float]] | None,
        rng: np.random.Generator,
        samples: int,
        chi: float,
    ) -> float:
        uphill = 0.0
        downhill = 0
        n = len(parameters)

        for _ in range(samples):
            candidate = parameters.copy()
            idx = int(rng.integers(0, n))
            candidate[idx] += rng.normal(scale=step_sizes[idx])
            candidate = _clip_to_bounds(candidate, bounds)
            new_loss = self.rank0_loss(candidate)
            delta = new_loss - current_loss
            if delta <= 0:
                downhill += 1
            else:
                uphill += delta

        uphill_count = samples - downhill
        if uphill_count <= 0:
            return max(float(np.mean(step_sizes)), 1.0)

        mean_uphill = uphill / uphill_count
        accepted = downhill
        denom = samples * chi + (1.0 - chi) * accepted
        denom = min(max(denom, 1e-12), samples - 1e-12)
        temperature = mean_uphill / np.log(samples / denom)
        if not np.isfinite(temperature) or temperature <= 0:
            temperature = max(mean_uphill, 1.0)
        return float(temperature)

    def _optimize(self, **kwargs: dict[str, Any]) -> npt.NDArray[np.float64]:
        parameters = np.asarray(self.loss.pot_data.parameters, dtype=float).copy()
        if parameters.size == 0:
            return parameters

        bounds = self.loss.pot_data.get_bounds()
        rng = np.random.default_rng(kwargs.get("seed", 40))
        callback = Callback(self.loss)

        step_scale = float(kwargs.get("step_scale", 0.1))
        nsteps = int(kwargs.get("steps_per_temperature", 20))
        maxiter = int(kwargs.get("maxiter", 1000))
        cooling = float(kwargs.get("cooling", 0.85))
        step_growth = float(kwargs.get("step_growth", 2.0))
        accept_low = float(kwargs.get("accept_low", 0.4))
        accept_high = float(kwargs.get("accept_high", 0.6))
        tol = float(kwargs.get("tol", 1e-12))
        min_temperature = float(kwargs.get("min_temperature", 0.0))
        auto_temperature = kwargs.get("anneal_temp", kwargs.get("temperature", "auto"))
        chi = float(kwargs.get("chi", 0.8))
        temperature_trials = int(kwargs.get("temperature_trials", max(10 * parameters.size, 50)))

        step_sizes = _step_sizes_from_bounds(parameters, bounds, step_scale)
        current_loss = self.rank0_loss(parameters)
        best_parameters = parameters.copy()
        best_loss = current_loss
        callback(OptimizeResult(x=parameters, fun=current_loss))

        if auto_temperature in {"auto", None}:
            temperature = self._estimate_temperature(
                parameters,
                current_loss,
                step_sizes,
                bounds,
                rng,
                temperature_trials,
                chi,
            )
        else:
            temperature = float(auto_temperature)
        if temperature <= 0:
            return parameters

        logger.info("Starting simulated annealing with T=%s", temperature)

        loop_counter = 0
        while loop_counter < maxiter and temperature > min_temperature:
            accepted = np.zeros(parameters.size, dtype=int)
            improved = False

            for _ in range(nsteps):
                for idx in range(parameters.size):
                    candidate = parameters.copy()
                    candidate[idx] += rng.uniform(-step_sizes[idx], step_sizes[idx])
                    candidate = _clip_to_bounds(candidate, bounds)

                    new_loss = self.rank0_loss(candidate)
                    delta = new_loss - current_loss
                    if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                        parameters = candidate
                        current_loss = new_loss
                        accepted[idx] += 1
                        improved = improved or (new_loss < best_loss)
                        if new_loss < best_loss:
                            best_loss = new_loss
                            best_parameters = candidate.copy()

            acceptance_rate = accepted / max(nsteps, 1)
            for idx, rate in enumerate(acceptance_rate):
                if rate > accept_high:
                    step_sizes[idx] *= 1.0 + step_growth * (rate - accept_high) / (1.0 - accept_high)
                elif rate < accept_low:
                    step_sizes[idx] /= 1.0 + step_growth * (accept_low - rate) / accept_low

            if bounds:
                for idx, (lb, ub) in enumerate(bounds):
                    if np.isfinite(lb) and np.isfinite(ub):
                        step_sizes[idx] = min(step_sizes[idx], max(ub - lb, tol))

            callback(OptimizeResult(x=parameters, fun=current_loss))
            logger.info("anneal %d: T=%s loss=%s best=%s", loop_counter, temperature, current_loss, best_loss)

            if not improved and np.all(np.abs(current_loss - best_loss) <= tol):
                break

            temperature *= cooling
            loop_counter += 1

        logger.info("")
        logger.info("Optimization result:")
        logger.info("  Message: %s", "Simulated annealing finished")
        logger.info("  Success: %s", True)
        logger.info("  Status code: %s", 0)
        logger.info("  Number of function evaluations: %s", "unknown")
        logger.info("  Number of iterations: %s", loop_counter)

        return best_parameters
