"""Optimizers based on SciPy."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import (
    OptimizeResult,
    differential_evolution,
    dual_annealing,
    minimize,
)

from forgeff.loss import LossFunctionBase
from forgeff.optimizers.base import ParallelOptimizerBase
from forgeff.parallel import is_master

logger = logging.getLogger(__name__)


class Callback:
    """Callback after each iteration."""

    def __init__(self, loss: LossFunctionBase) -> None:
        self.loss = loss
        self.iter = 0

    def __call__(self, intermediate_result: OptimizeResult) -> None:
        fun = intermediate_result.fun
        if is_master(self.loss.comm):
            logger.info("loss %d: %s", self.iter, fun)
            for handler in logger.handlers:
                handler.flush()
        self.iter += 1


class ScipyOptimizerBase(ParallelOptimizerBase):
    """SciPy Optimizer."""

    @property
    def optimized_default(self) -> list[str]:
        return []

    @property
    def optimized_allowed(self) -> list[str]:
        return []

    def print_result(self, result: OptimizeResult) -> None:
        """Print `result`."""
        logger.info("")
        for handler in logger.handlers:
            handler.flush()
        logger.info("Optimization result:")
        logger.info("  Message: %s", result.message)
        logger.info("  Success: %s", result.success)
        logger.info("  Status code: %s", result.status)
        logger.info("  Number of function evaluations: %s", result.nfev)
        logger.info("  Number of iterations: %s", result.nit)


class ScipyDualAnnealingOptimizer(ScipyOptimizerBase):
    def _optimize(self, **kwargs: dict[str, Any]) -> npt.NDArray[np.float64]:
        parameters = self.loss.pot_data.parameters
        bounds = self.loss.pot_data.get_bounds()
        callback = Callback(self.loss)
        callback(OptimizeResult(x=parameters, fun=self.rank0_loss(parameters)))
        result = dual_annealing(
            self.rank0_loss,
            bounds=bounds,
            callback=callback,
            seed=40,
            x0=parameters,
        )
        self.print_result(result)
        return result.x


class ScipyDifferentialEvolutionOptimizer(ScipyOptimizerBase):
    def _optimize(self, **kwargs: dict[str, Any]) -> npt.NDArray[np.float64]:
        parameters = self.loss.pot_data.parameters
        bounds = self.loss.pot_data.get_bounds()
        callback = Callback(self.loss)
        callback(OptimizeResult(x=parameters, fun=self.rank0_loss(parameters)))
        result = differential_evolution(
            self.rank0_loss,
            bounds,
            popsize=30,
            callback=callback,
            x0=parameters,
        )
        self.print_result(result)
        return result.x


class ScipyMinimizeOptimizer(ScipyOptimizerBase):
    """`Optimizer` class using `scipy.minimize`."""

    def _optimize(self, **kwargs: dict[str, Any]) -> npt.NDArray[np.float64]:
        """Optimize using `scipy.optimize.minimize`.

        Returns
        -------
        npt.NDArray[np.float64]

        Raises
        ------
        ValueError
            If `jac` and `scaling` is set at the same time.

        """
        parameters = self.loss.pot_data.parameters
        bounds = self.loss.pot_data.get_bounds()
        callback = Callback(self.loss)
        callback(OptimizeResult(x=parameters, fun=self.rank0_loss(parameters)))

        if kwargs.get("jac") is True:
            if "scaling" in self.optimized:
                msg = "`jac` cannot (so far) be used to optimize `scaling`."
                raise ValueError(msg)
            kwargs["jac"] = self.rank0_jac
        elif kwargs.get("jac") is None:
            # Default to numerical derivatives when the optimizer does not
            # explicitly request an analytical Jacobian.
            kwargs["jac"] = "2-point"
        if kwargs.get("method", "").lower() in {
            "cg",
            "bfgs",
            "newton-cg",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        }:
            bounds = None

        result = minimize(
            self.rank0_loss,
            parameters,
            bounds=bounds,
            callback=callback,
            **kwargs,
        )
        self.print_result(result)
        return result.x
