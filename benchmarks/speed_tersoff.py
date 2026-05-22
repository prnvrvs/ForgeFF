"""Benchmark Tersoff runtime against ASE."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.tersoff import Tersoff as ASETersoff

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgeff.calculator import make_calculator
from forgeff.potentials.tersoff.data import TersoffData, TersoffParameters
from benchmarks.plot_style import PALETTE, apply_publication_style, save_figure, style_axes


@dataclass
class BenchmarkResult:
    atoms: int
    n: int
    ase_mean: float
    ase_std: float
    ase_min: float
    ase_max: float
    numpy_mean: float
    numpy_std: float
    numpy_min: float
    numpy_max: float
    numba_mean: float
    numba_std: float
    numba_min: float
    numba_max: float

    @property
    def speedup(self) -> float:
        return self.ase_mean / self.numba_mean if self.numba_mean > 0 else float("inf")


def _parameters() -> dict[tuple[str, str, str], TersoffParameters]:
    return {
        ("Si", "Si", "Si"): TersoffParameters(
            m=1.0,
            gamma=1.0,
            lambda3=1.2,
            c=1.0,
            d=0.5,
            h=-0.3,
            n=1.0,
            beta=1.0,
            lambda2=1.5,
            B=0.8,
            R=3.0,
            D=0.2,
            lambda1=2.6,
            A=1.4,
        )
    }


def _make_tersoff_data() -> TersoffData:
    return TersoffData.from_parameter_dict(_parameters(), species=["Si"])


def _make_si_atoms(n: int):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (n, n, n)
    rng = np.random.default_rng(5000 + n)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.wrap()
    return atoms


def _time_callable(func, atoms, repeats: int, *, before=None) -> tuple[float, float, float, float]:
    samples = []
    for _ in range(repeats):
        probe = atoms.copy()
        if before is not None:
            before()
        start = perf_counter()
        func(probe)
        samples.append((perf_counter() - start) * 1000.0)
    values = np.array(samples, dtype=float)
    return float(values.mean()), float(values.std(ddof=0)), float(values.min()), float(values.max())


def _evaluate(calc, atoms) -> None:
    calc.calculate(atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)


def _benchmark(sizes: list[int], repeats: int) -> list[BenchmarkResult]:
    ase_calc = ASETersoff(parameters=_parameters(), skin=0.0)
    numpy_calc = make_calculator(_make_tersoff_data(), engine="numpy", skin=0.0)
    numba_calc = make_calculator(_make_tersoff_data(), engine="numba", skin=0.0)

    warm = _make_si_atoms(sizes[0])
    _evaluate(ase_calc, warm.copy())
    _evaluate(numpy_calc, warm.copy())
    _evaluate(numba_calc, warm.copy())

    results: list[BenchmarkResult] = []
    for n in sizes:
        atoms = _make_si_atoms(n)
        ase_mean, ase_std, ase_min, ase_max = _time_callable(
            lambda a: _evaluate(ase_calc, a),
            atoms,
            repeats,
            before=ase_calc.reset,
        )
        numpy_mean, numpy_std, numpy_min, numpy_max = _time_callable(lambda a: _evaluate(numpy_calc, a), atoms, repeats)
        numba_mean, numba_std, numba_min, numba_max = _time_callable(lambda a: _evaluate(numba_calc, a), atoms, repeats)
        results.append(
            BenchmarkResult(
                len(atoms),
                n,
                ase_mean,
                ase_std,
                ase_min,
                ase_max,
                numpy_mean,
                numpy_std,
                numpy_min,
                numpy_max,
                numba_mean,
                numba_std,
                numba_min,
                numba_max,
            )
        )
    return results


def _plot(results: list[BenchmarkResult], output: Path) -> None:
    apply_publication_style()
    atoms = np.array([r.atoms for r in results], dtype=float)
    ase = np.array([r.ase_mean for r in results], dtype=float)
    ase_min = np.array([r.ase_min for r in results], dtype=float)
    ase_max = np.array([r.ase_max for r in results], dtype=float)
    numpy = np.array([r.numpy_mean for r in results], dtype=float)
    numpy_min = np.array([r.numpy_min for r in results], dtype=float)
    numpy_max = np.array([r.numpy_max for r in results], dtype=float)
    numba = np.array([r.numba_mean for r in results], dtype=float)
    numba_min = np.array([r.numba_min for r in results], dtype=float)
    numba_max = np.array([r.numba_max for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    ax.fill_between(atoms, ase_min, ase_max, color=PALETTE["reference"], alpha=0.10)
    ax.fill_between(atoms, numpy_min, numpy_max, color=PALETTE["numpy"], alpha=0.10)
    ax.fill_between(atoms, numba_min, numba_max, color=PALETTE["numba"], alpha=0.10)
    ax.plot(atoms, ase, "o-", color=PALETTE["reference"], label="ASE")
    ax.plot(atoms, numpy, "o-", color=PALETTE["numpy"], label="ForgeFF (NumPy)")
    ax.plot(atoms, numba, "o-", color=PALETTE["numba"], label="ForgeFF (Numba)")
    style_axes(ax, title="Tersoff runtime on distorted Si")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "_static" / "performance",
    )
    args = parser.parse_args()

    results = _benchmark(args.sizes, args.repeats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot(results, args.output_dir / "tersoff_runtime.png")

    summary = {
        "sizes": args.sizes,
        "repeats": args.repeats,
        "tersoff": [
            {
                "atoms": r.atoms,
                "ase_mean": r.ase_mean,
                "ase_std": r.ase_std,
                "ase_min": r.ase_min,
                "ase_max": r.ase_max,
                "numpy_mean": r.numpy_mean,
                "numpy_std": r.numpy_std,
                "numpy_min": r.numpy_min,
                "numpy_max": r.numpy_max,
                "numba_mean": r.numba_mean,
                "numba_std": r.numba_std,
                "numba_min": r.numba_min,
                "numba_max": r.numba_max,
                "speedup": r.speedup,
            }
            for r in results
        ],
    }
    (args.output_dir / "tersoff_speed_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    for row in results:
        print(
            f"{row.atoms:5d} atoms  ASE {row.ase_mean:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_mean:8.3f} ms  ForgeFF (Numba) {row.numba_mean:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )


if __name__ == "__main__":
    main()
