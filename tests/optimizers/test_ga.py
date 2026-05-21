"""Tests for genetic algorithm (GA)."""

from __future__ import annotations

import os
import subprocess
import sys
import random
from pathlib import Path
from textwrap import dedent

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_ga_snippet(snippet: str) -> str:
    root = _repo_root()
    bootstrap = dedent(
        f"""
        from pathlib import Path
        import importlib.util
        import random
        import sys

        import numpy as np

        sys.meta_path = [
            finder
            for finder in sys.meta_path
            if "MesonpyMetaFinder" not in type(finder).__name__
            or "ForgeFF" not in repr(finder)
            and "motep" not in repr(finder)
        ]

        root = Path({str(root)!r})
        spec = importlib.util.spec_from_file_location(
            "forgeff",
            root / "forgeff" / "__init__.py",
            submodule_search_locations=[str(root / "forgeff")],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["forgeff"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        """
    )
    code = bootstrap + "\n" + dedent(snippet)
    env = os.environ.copy()
    env["NUMBA_DISABLE_JIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout


def test_ga_runs_and_keeps_population_size() -> None:
    stdout = _run_ga_snippet(
        """
from forgeff.optimizers.ga import GeneticAlgorithm


def fun(x: np.ndarray) -> float:
    n = len(x)
    sum1 = np.sum(np.square(x))
    sum2 = np.sum(np.cos(2 * np.pi * np.array(x)))
    return float(-20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e)


initial_guess = np.array([0.0, 0.0])
ga = GeneticAlgorithm(
    fun,
    initial_guess,
    lower_bound=np.array([0.0, 0.0]),
    upper_bound=np.array([10.0, 10.0]),
    population_size=100,
    mutation_rate=0.1,
    elitism_rate=0.1,
    crossover_probability=0.9,
    seed=40,
)
ga.initialize_population()
ga.evolve_with_mix(fun, 5)
print(len(ga.population))
print(ga.population_size)
"""
    )
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    assert lines[-2:] == ["100", "100"]


def test_ga_initial_population_is_not_constant_and_seeded() -> None:
    stdout = _run_ga_snippet(
        """
from forgeff.optimizers.ga import GeneticAlgorithm


def fun(x: np.ndarray) -> float:
    return float(np.sum(np.square(x)))


ga = GeneticAlgorithm(
    fun,
    np.array([0.0, 0.0]),
    lower_bound=np.array([0.0, 0.0]),
    upper_bound=np.array([10.0, 10.0]),
    population_size=20,
    mutation_rate=0.1,
    elitism_rate=0.1,
    crossover_probability=0.9,
    seed=40,
)
ga.initialize_population()
population = np.asarray(ga.population, dtype=float)
print(int(np.any(np.any(population[0] != population[1:], axis=-1))))
"""
    )
    assert stdout.strip().endswith("1")


def test_ga_crossover_uses_elementwise_uniform() -> None:
    from forgeff.optimizers.ga import GeneticAlgorithm

    ga = GeneticAlgorithm(
        lambda x: float(np.sum(np.square(x))),
        np.array([0.0, 0.0]),
        lower_bound=np.array([0.0, 0.0]),
        upper_bound=np.array([10.0, 10.0]),
        crossover_probability=1.0,
        seed=0,
    )

    child1, child2 = ga.crossover(np.array([0.0, 1.0]), np.array([2.0, 5.0]))
    lower = np.array([0.0, 1.0])
    upper = np.array([2.0, 5.0])
    ratio1 = (np.asarray(child1) - lower) / (upper - lower)
    ratio2 = (np.asarray(child2) - lower) / (upper - lower)
    assert not np.allclose(np.asarray(child1), lower)
    assert not np.allclose(np.asarray(child2), lower)
    assert not np.allclose(ratio1[0], ratio1[1])
    assert not np.allclose(ratio2[0], ratio2[1])


def test_ga_initialize_population_does_not_touch_global_random_state() -> None:
    from forgeff.optimizers.ga import GeneticAlgorithm

    random.seed(123)
    baseline = random.random()

    random.seed(123)
    ga = GeneticAlgorithm(
        lambda x: float(np.sum(np.square(x))),
        np.array([0.0, 0.0]),
        lower_bound=np.array([0.0, 0.0]),
        upper_bound=np.array([10.0, 10.0]),
        population_size=4,
        mutation_rate=0.1,
        elitism_rate=0.1,
        crossover_probability=0.9,
        seed=40,
    )
    ga.initialize_population()
    after = random.random()

    assert after == baseline
