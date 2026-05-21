import importlib.util
from pathlib import Path
import sys
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

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
from forgeff.optimizers.ga import GeneticAlgorithm
from forgeff.parallel import DummyMPIComm


@dataclass
class _PotData:
    parameters: np.ndarray

    def get_bounds(self) -> list[tuple[float, float]]:
        return [(-5.12, 5.12)] * len(self.parameters)


class _RastriginLoss:
    """Multimodal landscape representing non-convex potential energy surfaces."""
    def __init__(self, initial: np.ndarray) -> None:
        self.pot_data = _PotData(parameters=np.asarray(initial, dtype=float))
        self.comm = DummyMPIComm()

    def __call__(self, parameters: np.ndarray | None) -> float:
        if parameters is not None:
            self.pot_data.parameters = np.asarray(parameters, dtype=float)
        x = self.pot_data.parameters
        return float(20 + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def main() -> None:
    print("==============================================================")
    print("Convergence Theory & Optimization Strategies Comparison")
    print("==============================================================")
    print("Theoretical Context:")
    print("1. Deterministic Local (L-BFGS-B): Fast, high precision, but prone to local minima.")
    print("2. Stochastic Global (Simulated Annealing): Can escape local minima, slow convergence.")
    print("3. Hybrid Memetic (Genetic Algorithm + Nelder-Mead): Balance of exploration & exploitation.")
    print("--------------------------------------------------------------")

    # Initial state (stuck in local minimum for deterministic solvers)
    initial = np.array([4.5, -3.5], dtype=float)
    
    # 1. Deterministic Local Optimization (L-BFGS-B)
    loss_lbfgs = _RastriginLoss(initial)
    t0 = time.perf_counter()
    res = minimize(loss_lbfgs, initial, method="L-BFGS-B", bounds=[(-5.12, 5.12)]*2)
    t_lbfgs = time.perf_counter() - t0
    print(f"[L-BFGS-B] Time: {t_lbfgs:.4f}s")
    print(f"  Final Loss: {res.fun:.8f}")
    print(f"  Final Parameters: {res.x}")
    print("  Status: Stuck in local minimum" if res.fun > 1e-4 else "  Status: Global minimum found")
    print("--------------------------------------------------------------")

    # 2. Stochastic Global Optimization (Simulated Annealing)
    loss_sa = _RastriginLoss(initial)
    optimizer_sa = SimulatedAnnealingOptimizer(loss_sa)
    t0 = time.perf_counter()
    optimizer_sa.optimize(
        seed=123,
        anneal_temp=10.0,
        steps_per_temperature=20,
        maxiter=100,
        cooling=0.85,
        step_scale=0.5,
    )
    t_sa = time.perf_counter() - t0
    final_sa_loss = loss_sa(loss_sa.pot_data.parameters)
    print(f"[Simulated Annealing] Time: {t_sa:.4f}s")
    print(f"  Final Loss: {final_sa_loss:.8f}")
    print(f"  Final Parameters: {loss_sa.pot_data.parameters}")
    print("  Status: Stuck in local minimum" if final_sa_loss > 1e-4 else "  Status: Global minimum found")
    print("--------------------------------------------------------------")

    # 3. Hybrid Memetic Optimization (Genetic Algorithm with Nelder-Mead Elites)
    loss_ga = _RastriginLoss(initial)
    t0 = time.perf_counter()
    ga = GeneticAlgorithm(
        loss_ga,
        initial,
        lower_bound=np.array([-5.12, -5.12]),
        upper_bound=np.array([5.12, 5.12]),
        population_size=30,
        mutation_rate=0.1,
        elitism_rate=0.2,
        crossover_probability=0.8,
        seed=123,
        superhuman=True,  # Enables local Nelder-Mead refinement on elites!
    )
    ga.initialize_population()
    best_solution = ga.evolve_with_mix(loss_ga, generations=10)
    t_ga = time.perf_counter() - t0
    final_ga_loss = loss_ga(best_solution)
    print(f"[Hybrid Genetic Algorithm] Time: {t_ga:.4f}s")
    print(f"  Final Loss: {final_ga_loss:.8f}")
    print(f"  Final Parameters: {best_solution}")
    print("  Status: Stuck in local minimum" if final_ga_loss > 1e-4 else "  Status: Global minimum found")
    print("==============================================================")


if __name__ == "__main__":
    main()
