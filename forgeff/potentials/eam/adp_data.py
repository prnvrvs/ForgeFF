"""ADP data for semi-empirical forcefields."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from .data import EAMData

logger = logging.getLogger(__name__)

@dataclass
class ADPData(EAMData):
    """Data structure for ADP potentials used in fitting.
    
    Extends EAM with dipole and quadrupole terms.
    """
    # Dipole u(r): (species, species, r_grid_size)
    dipole_values: npt.NDArray[np.float64] | None = None
    # Quadrupole w(r): (species, species, r_grid_size)
    quadrupole_values: npt.NDArray[np.float64] | None = None
    
    def __post_init__(self):
        if "dipole_values" not in self.optimized:
            self.optimized.extend(["dipole_values", "quadrupole_values"])

    @property
    def parameters(self) -> np.ndarray:
        """Serialized parameters for the optimizer."""
        tmp = [super().parameters]
        if "dipole_values" in self.optimized:
            tmp.append(self.dipole_values.flat)
        if "quadrupole_values" in self.optimized:
            tmp.append(self.quadrupole_values.flat)
        return np.hstack(tmp)

    @parameters.setter
    def parameters(self, parameters: npt.ArrayLike) -> None:
        """Update values from serialized parameters."""
        params = np.asanyarray(parameters)
        spc = self.species_count
        nr = len(self.r_grid) if self.r_grid is not None else 0
        
        # Determine how many parameters EAM takes
        eam_params_count = super().number_of_parameters_optimized
        super(ADPData, type(self)).parameters.fset(self, params[:eam_params_count])
        
        n = eam_params_count
        if "dipole_values" in self.optimized:
            size = spc * spc * nr
            self.dipole_values = params[n : n + size].reshape(spc, spc, nr)
            self.dipole_values = 0.5 * (self.dipole_values + self.dipole_values.transpose(1, 0, 2))
            n += size
        if "quadrupole_values" in self.optimized:
            size = spc * spc * nr
            self.quadrupole_values = params[n : n + size].reshape(spc, spc, nr)
            self.quadrupole_values = 0.5 * (self.quadrupole_values + self.quadrupole_values.transpose(1, 0, 2))
            n += size

    @property
    def number_of_parameters_optimized(self) -> int:
        n = super().number_of_parameters_optimized
        spc = self.species_count
        nr = len(self.r_grid) if self.r_grid is not None else 0
        if "dipole_values" in self.optimized:
            n += spc * spc * nr
        if "quadrupole_values" in self.optimized:
            n += spc * spc * nr
        return n

    def initialize(self, rng: np.random.Generator) -> None:
        """Random initialization of potential values."""
        super().initialize(rng)
        spc = self.species_count
        nr = len(self.r_grid)
        if self.dipole_values is None:
            self.dipole_values = rng.uniform(-0.01, 0.01, (spc, spc, nr))
            self.dipole_values = 0.5 * (self.dipole_values + self.dipole_values.transpose(1, 0, 2))
        if self.quadrupole_values is None:
            self.quadrupole_values = rng.uniform(-0.01, 0.01, (spc, spc, nr))
            self.quadrupole_values = 0.5 * (self.quadrupole_values + self.quadrupole_values.transpose(1, 0, 2))

    def log(self) -> None:
        super().log()
        logger.debug("ADP Parameters logged")

    def write(self, filename: str | Path) -> None:
        """Write the potential to a NumPy archive."""
        np.save(filename, self.__dict__, allow_pickle=True)

    @classmethod
    def from_file(cls, filename: str | Path) -> "ADPData":
        """Load the potential from a NumPy archive."""
        data = np.load(filename, allow_pickle=True).item()
        return cls(**data)
