import importlib.util
from pathlib import Path
import random
import sys

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

from forgeff.optimizers.ga import GeneticAlgorithm


def _rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(20 + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


# %%
# Genetic algorithm on a small multimodal landscape.
random.seed(40)
initial_guess = np.array([4.5, -3.5], dtype=float)
ga = GeneticAlgorithm(
    _rastrigin,
    initial_guess,
    lower_bound=np.array([-5.12, -5.12], dtype=float),
    upper_bound=np.array([5.12, 5.12], dtype=float),
    population_size=40,
    mutation_rate=0.1,
    elitism_rate=0.1,
    crossover_probability=0.9,
)
ga.initialize_population()

initial_population = np.asarray(ga.population, dtype=float)
initial_best = min(_rastrigin(ind) for ind in initial_population)
best_solution = ga.evolve_with_mix(_rastrigin, generations=30)
final_best = _rastrigin(np.asarray(best_solution, dtype=float))

print("Genetic algorithm example")
print(f"initial best fitness: {initial_best}")
print(f"final best fitness: {final_best}")
print(f"best solution: {np.array2string(np.asarray(best_solution), precision=3)}")
