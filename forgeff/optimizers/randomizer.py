"""Module for `Randomizer`."""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize._optimize import OptimizeResult

from forgeff.loss import LossFunctionBase
from forgeff.optimizers.base import ParallelOptimizerBase
from forgeff.optimizers.scipy import Callback


class Randomizer(ParallelOptimizerBase):
    """Special `Optimizer` class that actually randomizes parameters."""

    def __init__(
        self,
        loss: LossFunctionBase,
        *,
        optimized: list[str] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize `Randomizer`."""
        super().__init__(loss=loss, **kwargs)
        if optimized is None:
            optimized = ["species_coeffs", "radial_coeffs", "moment_coeffs"]
        self.optimized = optimized

    @property
    def optimized_default(self) -> list[str]:
        return ["species_coeffs", "radial_coeffs", "moment_coeffs"]

    @property
    def optimized_allowed(self) -> list[str]:
        return ["species_coeffs", "radial_coeffs", "moment_coeffs"]

    def _optimize(self, **kwargs: dict[str, Any]) -> npt.NDArray[np.float64]:
        parameters = self.loss.pot_data.parameters
        callback = Callback(self.loss)
        seed = kwargs.get("seed", 42)
        rng: np.random.Generator = kwargs.get("rng") or np.random.default_rng(seed)

        # Calculate basis functions of `loss.images`
        loss_value = self.rank0_loss(parameters)
        self.rank0_gather_data()

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

        pot_data = self.loss.pot_data
        for key in self.optimized:
            lb = -5.0
            ub = +5.0
            value = getattr(pot_data, key)
            shape = np.asarray(value).shape
            setattr(pot_data, key, rng.uniform(lb, ub, size=shape))
        # Update `parameters` by calling the property
        parameters = pot_data.parameters

        # Evaluate loss with the new parameters
        loss_value = self.rank0_loss(parameters)

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

        return parameters
