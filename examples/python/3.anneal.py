import importlib.util
from pathlib import Path
import sys
from dataclasses import dataclass

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if "MesonpyMetaFinder" not in type(finder).__name__
    or "ForgeFF" not in repr(finder)
    and "motep" not in repr(finder)
]
spec = importlib.util.spec_from_file_location(
    "forgeff",
    ROOT / "forgeff" / "__init__.py",
    submodule_search_locations=[str(ROOT / "forgeff")],
)
module = importlib.util.module_from_spec(spec)
sys.modules["forgeff"] = module
assert spec.loader is not None
spec.loader.exec_module(module)

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


# %%
# Simulated annealing on a tiny quadratic target.
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

print("Simulated annealing example")
print(f"initial loss: {initial_loss}")
print(f"final loss: {final_loss}")
print(f"final parameters: {np.array2string(final_parameters, precision=3)}")
print(f"target parameters: {np.array2string(target, precision=3)}")
