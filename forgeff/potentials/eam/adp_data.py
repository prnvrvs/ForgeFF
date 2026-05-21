"""ADP data for semi-empirical forcefields."""

import logging
from dataclasses import asdict, dataclass
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
        if not self._uses_block_optimization() and "dipole_values" not in self.optimized:
            self.optimized.extend(["dipole_values", "quadrupole_values"])

    def _species_pair_from_name(self, name: str) -> tuple[int, int]:
        labels = self._species_labels()
        suffix = name.split(".", 1)[1] if "." in name else name
        normalized = "".join(ch for ch in suffix if ch.isalnum()).lower()
        matches: list[tuple[int, int]] = []
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                if normalized == ("".join(ch for ch in left if ch.isalnum()).lower() + "".join(ch for ch in right if ch.isalnum()).lower()):
                    matches.append((i, j))
        if not matches:
            raise KeyError(name)
        if len(matches) > 1:
            raise ValueError(f"Ambiguous ADP pair block name {name!r}.")
        return matches[0]

    def _adp_block_values(self, name: str) -> np.ndarray:
        if name.startswith("dipole."):
            i, j = self._species_pair_from_name(name)
            return np.asarray(self.dipole_values[i, j], dtype=float).reshape(-1)
        if name.startswith("quadrupole."):
            i, j = self._species_pair_from_name(name)
            return np.asarray(self.quadrupole_values[i, j], dtype=float).reshape(-1)
        return super()._block_values(name)

    def _adp_set_block_values(self, name: str, values: npt.ArrayLike) -> None:
        if name.startswith("dipole."):
            i, j = self._species_pair_from_name(name)
            arr = np.asarray(values, dtype=float).reshape(-1)
            self.dipole_values[i, j] = arr
            self.dipole_values[j, i] = arr
            return
        if name.startswith("quadrupole."):
            i, j = self._species_pair_from_name(name)
            arr = np.asarray(values, dtype=float).reshape(-1)
            self.quadrupole_values[i, j] = arr
            self.quadrupole_values[j, i] = arr
            return
        super()._set_block_values(name, np.asarray(values, dtype=float))

    @property
    def parameters(self) -> np.ndarray:
        """Serialized parameters for the optimizer."""
        if self._uses_block_optimization():
            return (
                np.hstack([self._adp_block_values(name) for name in self.optimized])
                if self.optimized
                else np.array([], dtype=float)
            )
        if not self.optimized:
            return np.array([], dtype=float)
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
        if self._uses_block_optimization():
            n = 0
            for name in self.optimized:
                size = self._adp_block_values(name).size
                self._adp_set_block_values(name, params[n : n + size])
                n += size
            return
        if not self.optimized:
            if params.size:
                raise ValueError("No ADP parameters are marked for optimization.")
            return
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
        if self._uses_block_optimization():
            return int(sum(self._adp_block_values(name).size for name in self.optimized))
        if not self.optimized:
            return 0
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
        if self._uses_block_optimization():
            if self.number_of_parameters_optimized == 0:
                return
            if np.allclose(self.parameters, 0.0):
                values = rng.uniform(-0.1, 0.1, self.number_of_parameters_optimized)
                self.parameters = values
            return
        if not self.optimized:
            return
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
        np.save(filename, asdict(self), allow_pickle=True)

    @classmethod
    def from_file(cls, filename: str | Path) -> "ADPData":
        """Load the potential from a NumPy archive."""
        data = np.load(filename, allow_pickle=True).item()
        if "backend" in data and "engine" not in data:
            data = dict(data)
            data["engine"] = data.pop("backend")
        return cls(**data)
