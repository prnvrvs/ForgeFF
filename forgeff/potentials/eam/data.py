"""EAM data for semi-empirical forcefields."""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

def _default_factory_int() -> npt.NDArray[np.int32]:
    return np.array([], dtype=np.int32)

@dataclass
class EAMData:
    """Data structure for EAM potentials used in fitting.
    
    This follows the structure of potfit tabulated potentials where 
    functions (phi, rho, F) are represented by splines on a grid.
    """
    potential_name: str = ""
    species_count: int = 0
    form: str = "alloy" # "alloy" or "fs"
    engine: str = ""
    
    # Grids for the different functions
    # phi and rho share a radial grid
    r_grid: npt.NDArray[np.float64] | None = None
    # F uses a density grid
    rho_grid: npt.NDArray[np.float64] | None = None
    
    # Values of the functions at grid points (the parameters to be optimized)
    # phi_values: (species, species, r_grid_size)
    # rphi_values: raw pair table as stored in LAMMPS/ASE export files
    # rho_values: (species, species, r_grid_size)
    # emb_values: (species, rho_grid_size)
    phi_values: npt.NDArray[np.float64] | None = None
    rphi_values: npt.NDArray[np.float64] | None = None
    rho_values: npt.NDArray[np.float64] | None = None
    emb_values: npt.NDArray[np.float64] | None = None
    
    _species: npt.NDArray[np.int32] = field(default_factory=_default_factory_int)
    optimized: list[str] = field(default_factory=lambda: ["phi_values", "rho_values", "emb_values"])

    @property
    def species(self) -> npt.NDArray[np.int32]:
        return self._species

    @species.setter
    def species(self, species: npt.NDArray[np.int32]) -> None:
        self._species = np.array(species, dtype=np.int32)
        self.species_count = self._species.size

    @property
    def parameters(self) -> np.ndarray:
        """Serialized parameters for the optimizer."""
        tmp = []
        if "phi_values" in self.optimized:
            tmp.append(self.phi_values.flat)
        if "rho_values" in self.optimized:
            if self.form == "fs":
                tmp.append(self.rho_values.flat)
            else:
                tmp.append(self.rho_values.diagonal(axis1=0, axis2=1).T.flat)
        if "emb_values" in self.optimized:
            tmp.append(self.emb_values.flat)
        return np.hstack(tmp)

    @parameters.setter
    def parameters(self, parameters: npt.ArrayLike) -> None:
        """Update values from serialized parameters."""
        params = np.asanyarray(parameters)
        spc = self.species_count
        nr = len(self.r_grid) if self.r_grid is not None else 0
        nrho = len(self.rho_grid) if self.rho_grid is not None else 0

        n = 0
        if "phi_values" in self.optimized:
            size = spc * spc * nr
            self.phi_values = params[n : n + size].reshape(spc, spc, nr)
            self.phi_values = 0.5 * (self.phi_values + self.phi_values.transpose(1, 0, 2))
            n += size
        if "rho_values" in self.optimized:
            if self.form == "fs":
                size = spc * spc * nr
                self.rho_values = params[n : n + size].reshape(spc, spc, nr)
            else:
                size = spc * nr
                curves = params[n : n + size].reshape(spc, nr)
                self.rho_values = np.zeros((spc, spc, nr), dtype=float)
                for idx in range(spc):
                    self.rho_values[:, idx, :] = curves[idx]
            n += size
        if "emb_values" in self.optimized:
            size = spc * nrho
            self.emb_values = params[n : n + size].reshape(spc, nrho)
            n += size

    @property
    def number_of_parameters_optimized(self) -> int:
        spc = self.species_count
        nr = len(self.r_grid) if self.r_grid is not None else 0
        nrho = len(self.rho_grid) if self.rho_grid is not None else 0
        n = 0
        if "phi_values" in self.optimized:
            n += spc * spc * nr
        if "rho_values" in self.optimized:
            if self.form == "fs":
                n += spc * spc * nr
            else:
                n += spc * nr
        if "emb_values" in self.optimized:
            n += spc * nrho
        return n

    def get_bounds(self) -> list[tuple[float, float]] | None:
        """Get bounds for EAM parameters."""
        return [(-10.0, 10.0)] * self.number_of_parameters_optimized

    def initialize(self, rng: np.random.Generator) -> None:
        """Random initialization of potential values."""
        spc = self.species_count
        nr = len(self.r_grid)
        nrho = len(self.rho_grid)
        if self.phi_values is None:
            self.phi_values = rng.uniform(-0.1, 0.1, (spc, spc, nr))
            self.phi_values = 0.5 * (self.phi_values + self.phi_values.transpose(1, 0, 2))
        if self.rho_values is None:
            if self.form == "fs":
                self.rho_values = rng.uniform(0.0, 0.1, (spc, spc, nr))
            else:
                curves = rng.uniform(0.0, 0.1, (spc, nr))
                self.rho_values = np.zeros((spc, spc, nr), dtype=float)
                for idx in range(spc):
                    self.rho_values[:, idx, :] = curves[idx]
        if self.emb_values is None:
            self.emb_values = rng.uniform(-0.1, 0.1, (spc, nrho))

    def log(self) -> None:
        logger.debug("EAM Parameters logged")

    def write(self, filename: str | Path) -> None:
        """Write the potential to a NumPy archive."""
        np.save(filename, asdict(self), allow_pickle=True)

    @classmethod
    def from_file(cls, filename: str | Path) -> "EAMData":
        """Load the potential from a NumPy archive."""
        data = np.load(filename, allow_pickle=True).item()
        if "backend" in data and "engine" not in data:
            data = dict(data)
            data["engine"] = data.pop("backend")
        return cls(**data)
