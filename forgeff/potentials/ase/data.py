"""Generic ASE Data for fitting any ASE calculator."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

@dataclass
class ASEData:
    """Data structure for arbitrary ASE calculators.
    
    Attributes
    ----------
    engine : str
        The name of the calculator engine (e.g., 'numpy', 'numba').
    calculator_kwargs : dict
        Arguments to pass to the calculator constructor.
    parameters_map : dict
        Mapping from parameter names to their metadata (shape, slice, etc.).
    """
    engine: str = ""
    calculator_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Internal storage for optimized parameters
    _parameters: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    # List of parameter names in order
    optimized: list[str] = field(default_factory=list)
    # Mapping name -> (shape, initial_value)
    parameter_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    species_energy_offsets: Dict[str, float] = field(default_factory=dict)

    def add_parameter(self, name: str, shape: tuple, initial_value: Any = None):
        """Register a parameter for optimization."""
        size = int(np.prod(shape))
        self.optimized.append(name)
        self.parameter_info[name] = {
            "shape": shape,
            "size": size,
        }
        if initial_value is not None:
            val = np.asanyarray(initial_value).flatten()
            self._parameters = np.concatenate([self._parameters, val])
        else:
            val = np.zeros(size)
            self._parameters = np.concatenate([self._parameters, val])

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: npt.ArrayLike) -> None:
        self._parameters = np.asanyarray(parameters)

    @property
    def number_of_parameters_optimized(self) -> int:
        return self._parameters.size

    def get_parameter_dict(self) -> Dict[str, Any]:
        """Convert flat parameters back to a dictionary of shaped arrays."""
        p_dict = {}
        offset = 0
        for name in self.optimized:
            info = self.parameter_info[name]
            size = info["size"]
            shape = info["shape"]
            p_dict[name] = self._parameters[offset : offset + size].reshape(shape)
            offset += size
        return p_dict

    def get_bounds(self) -> list[tuple[float, float]] | None:
        """Get bounds for the parameters. Returns None if no bounds are set."""
        return [(-10.0, 10.0)] * self.number_of_parameters_optimized

    def initialize(self, rng: np.random.Generator) -> None:
        """Initialize with random values if they are all zero."""
        if self._parameters.size == 0:
            return
        if np.all(self._parameters == 0):
            # Simple uniform initialization for now
            self._parameters = rng.uniform(-0.1, 0.1, self._parameters.size)

    def log(self) -> None:
        logger.debug("ASEData parameters: %s", self.optimized)

    def write(self, filename: str):
        """Save the potential state."""
        np.save(
            filename,
            {
                "params": self._parameters,
                "info": self.parameter_info,
                "kwargs": self.calculator_kwargs,
                "species_energy_offsets": self.species_energy_offsets,
            },
        )
