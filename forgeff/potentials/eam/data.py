"""EAM data for semi-empirical forcefields."""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ase.data import chemical_symbols

logger = logging.getLogger(__name__)

def _default_factory_int() -> npt.NDArray[np.int32]:
    return np.array([], dtype=np.int32)


def _species_label(number: int) -> str:
    return str(chemical_symbols[int(number)])


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
    species_energy_offsets: dict[str, float] = field(default_factory=dict)
    optimized: list[str] = field(default_factory=lambda: ["phi_values", "rho_values", "emb_values"])

    @property
    def species(self) -> npt.NDArray[np.int32]:
        return self._species

    @species.setter
    def species(self, species: npt.NDArray[np.int32]) -> None:
        self._species = np.array(species, dtype=np.int32)
        self.species_count = self._species.size

    def _species_labels(self) -> list[str]:
        return [_species_label(number) for number in np.asarray(self.species, dtype=int).tolist()]

    def _uses_block_optimization(self) -> bool:
        return any("." in name for name in self.optimized)

    def _block_maps(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        labels = self._species_labels()
        blocks: dict[str, tuple[str, tuple[int, ...]]] = {}
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                if i <= j:
                    blocks[f"pair.{left}{right}"] = ("pair", (i, j))
        if self.form == "fs":
            for i, left in enumerate(labels):
                for j, right in enumerate(labels):
                    blocks[f"density.{left}{right}"] = ("density", (i, j))
        else:
            for i, left in enumerate(labels):
                blocks[f"density.{left}"] = ("density", (i,))
        for i, left in enumerate(labels):
            blocks[f"embedding.{left}"] = ("embedding", (i,))
        return blocks

    def _block_values(self, name: str) -> np.ndarray:
        blocks = self._block_maps()
        if name in blocks:
            kind, index = blocks[name]
            if kind == "pair":
                i, j = index
                return np.asarray(self.phi_values[i, j], dtype=float).reshape(-1)
            if kind == "density":
                if self.form == "fs":
                    i, j = index
                    return np.asarray(self.rho_values[i, j], dtype=float).reshape(-1)
                (j,) = index
                return np.asarray(self.rho_values[0, j], dtype=float).reshape(-1)
            if kind == "embedding":
                (i,) = index
                return np.asarray(self.emb_values[i], dtype=float).reshape(-1)

        legacy = {
            "phi_values": np.asarray(self.phi_values, dtype=float).reshape(-1),
            "rho_values": (
                np.asarray(self.rho_values, dtype=float).reshape(-1)
                if self.form == "fs"
                else np.asarray(self.rho_values.diagonal(axis1=0, axis2=1).T, dtype=float).reshape(-1)
            ),
            "emb_values": np.asarray(self.emb_values, dtype=float).reshape(-1),
        }
        if name not in legacy:
            raise KeyError(name)
        return legacy[name]

    def _set_block_values(self, name: str, values: np.ndarray) -> None:
        blocks = self._block_maps()
        if name in blocks:
            kind, index = blocks[name]
            if kind == "pair":
                i, j = index
                self.phi_values[i, j] = values.reshape(-1)
                self.phi_values[j, i] = self.phi_values[i, j]
                return
            if kind == "density":
                if self.form == "fs":
                    i, j = index
                    self.rho_values[i, j] = values.reshape(-1)
                    return
                (j,) = index
                self.rho_values[:, j, :] = values.reshape(-1)
                return
            if kind == "embedding":
                (i,) = index
                self.emb_values[i] = values.reshape(-1)
                return

        if name == "phi_values":
            arr = values.reshape(self.phi_values.shape)
            self.phi_values = 0.5 * (arr + arr.transpose(1, 0, 2))
            return
        if name == "rho_values":
            if self.form == "fs":
                self.rho_values = values.reshape(self.rho_values.shape)
            else:
                curves = values.reshape(self.species_count, -1)
                self.rho_values = np.zeros_like(self.rho_values)
                for idx in range(self.species_count):
                    self.rho_values[:, idx, :] = curves[idx]
            return
        if name == "emb_values":
            self.emb_values = values.reshape(self.emb_values.shape)
            return
        raise KeyError(name)

    @property
    def parameters(self) -> np.ndarray:
        """Serialized parameters for the optimizer."""
        if self._uses_block_optimization():
            return np.hstack([self._block_values(name) for name in self.optimized]) if self.optimized else np.array([], dtype=float)
        if not self.optimized:
            return np.array([], dtype=float)
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
        if self._uses_block_optimization():
            n = 0
            for name in self.optimized:
                size = self._block_values(name).size
                self._set_block_values(name, params[n : n + size])
                n += size
            return
        if not self.optimized:
            if params.size:
                raise ValueError("No EAM parameters are marked for optimization.")
            return
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
        if self._uses_block_optimization():
            return int(sum(self._block_values(name).size for name in self.optimized))
        if not self.optimized:
            return 0
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
        if self._uses_block_optimization():
            if self.number_of_parameters_optimized == 0:
                return
            if np.allclose(self.parameters, 0.0):
                values = rng.uniform(-0.1, 0.1, self.number_of_parameters_optimized)
                self.parameters = values
            return
        if not self.optimized:
            return
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
