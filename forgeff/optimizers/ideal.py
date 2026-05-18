"""Module for `NoInteractionOptimizer`."""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize._optimize import OptimizeResult

from forgeff.optimizers.base import ParallelOptimizerBase
from forgeff.optimizers.scipy import Callback


class NoInteractionOptimizer(ParallelOptimizerBase):
    """Optimizer assuming no atomic interaction."""

    def _optimize(self, **kwargs: dict[str, Any]) -> npt.NDArray[np.float64]:
        """Optimize the linear parameter groups.

        The values are determined using the least-square method.
        Note that, if there is little diversity in the training set, the
        resulting fit may be less informative.

        Returns
        -------
        npt.NDArray[np.float64]

        """
        parameters = self.loss.pot_data.parameters
        callback = Callback(self.loss)

        # Calculate basis functions of `loss.images`
        loss_value = self.rank0_loss(parameters)
        self.rank0_gather_data()

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

        # Prepare and solve the LLS problem
        matrix = self._calc_matrix()
        vector = self._calc_vector()

        fitted = np.linalg.lstsq(matrix, vector, rcond=None)[0]

        # Update `pot_data`
        self.loss.pot_data.scaling = 1.0
        self.loss.pot_data.moment_coeffs[...] = 0.0
        self.loss.pot_data.radial_coeffs[...] = 0.0
        self.loss.pot_data.species_coeffs = fitted
        # Update `parameters` by calling the property
        parameters = self.loss.pot_data.parameters

        # Evaluate loss with the new parameters
        loss_value = self.rank0_loss(parameters)

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

        return parameters

    @property
    def optimized_default(self) -> list[str]:
        return ["species_coeffs"]

    @property
    def optimized_allowed(self) -> list[str]:
        return ["species_coeffs", "moment_coeffs", "radial_coeffs"]

    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS).

        Returns
        -------
        np.ndarray

        """
        loss = self.loss
        species = loss.pot_data.species
        images = loss.images
        counts = np.full((len(images), len(species)), np.nan)
        for i, atoms in enumerate(images):
            for j, s in enumerate(species):
                counts[i, j] = list(atoms.numbers).count(s)
        return counts

    def _calc_vector(self) -> np.ndarray:
        """Calculate the vector for linear least squares (LLS).

        Returns
        -------
        np.ndarray

        """
        images = self.loss.images
        return np.fromiter(
            (atoms.calc.targets["energy"] for atoms in images),
            dtype=float,
            count=len(images),
        )
