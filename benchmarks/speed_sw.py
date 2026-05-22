"""Benchmark Stillinger-Weber runtime against matscipy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from ase.calculators.calculator import all_changes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forgeff.calculator import make_calculator
from forgeff.potentials.sw.data import SWData
from benchmarks.plot_style import PALETTE, apply_publication_style, save_figure, style_axes


@dataclass
class BenchmarkResult:
    atoms: int
    n: int
    matscipy_mean: float
    matscipy_std: float
    matscipy_min: float
    matscipy_max: float
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
        return self.matscipy_mean / self.numba_mean if self.numba_mean > 0 else float("inf")


def _make_sw_data() -> SWData:
    return SWData(
        species=["Si"],
        epsilon=2.1683,
        sigma=2.0951,
        costheta0=1.0 / 3.0,
        A=7.049556277,
        B=0.6022245584,
        p=4.0,
        a=1.8,
        lambda1=21.0,
        gamma=1.2,
    )


def _matscipy_calc():
    try:
        from matscipy.calculators.manybody import StillingerWeber
        from matscipy.calculators.manybody.calculator import Manybody
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("matscipy is required for this benchmark") from exc

    params = {
        "el": "Si",
        "epsilon": 2.1683,
        "sigma": 2.0951,
        "costheta0": 1.0 / 3.0,
        "A": 7.049556277,
        "B": 0.6022245584,
        "p": 4.0,
        "a": 1.8,
        "lambda1": 21.0,
        "gamma": 1.2,
    }
    sw = StillingerWeber(params)
    return Manybody(
        sw["atom_type"],
        sw["pair_type"],
        sw["F"],
        sw["G"],
        sw["d1F"],
        sw["d2F"],
        sw["d11F"],
        sw["d22F"],
        sw["d12F"],
        sw["d1G"],
        sw["d11G"],
        sw["d2G"],
        sw["d22G"],
        sw["d12G"],
        sw["cutoff"],
    )


def _make_si_atoms(n: int):
    return bulk("Si", "diamond", a=5.43, cubic=True) * (n, n, n)


def _time_callable(func, atoms, repeats: int) -> tuple[float, float, float, float]:
    samples = []
    for _ in range(repeats):
        probe = atoms.copy()
        start = perf_counter()
        func(probe)
        samples.append((perf_counter() - start) * 1000.0)
    values = np.array(samples, dtype=float)
    return float(values.mean()), float(values.std(ddof=0)), float(values.min()), float(values.max())


def _evaluate(calc, atoms) -> None:
    calc.calculate(atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)


def _benchmark(sizes: list[int], repeats: int) -> list[BenchmarkResult]:
    matscipy_calc = _matscipy_calc()
    numpy_calc = make_calculator(_make_sw_data(), engine="numpy")
    numba_data = _make_sw_data()
    numba_calc = make_calculator(numba_data, engine="numba")

    warm = _make_si_atoms(sizes[0])
    _evaluate(matscipy_calc, warm.copy())
    _evaluate(numpy_calc, warm.copy())
    _evaluate(numba_calc, warm.copy())

    results: list[BenchmarkResult] = []
    for n in sizes:
        atoms = _make_si_atoms(n)

        def run_matscipy(probe):
            _evaluate(matscipy_calc, probe)

        def run_numpy(probe):
            _evaluate(numpy_calc, probe)

        def run_numba(probe):
            _evaluate(numba_calc, probe)

        matscipy_mean, matscipy_std, matscipy_min, matscipy_max = _time_callable(run_matscipy, atoms, repeats)
        numpy_mean, numpy_std, numpy_min, numpy_max = _time_callable(run_numpy, atoms, repeats)
        numba_mean, numba_std, numba_min, numba_max = _time_callable(run_numba, atoms, repeats)
        results.append(
            BenchmarkResult(
                len(atoms),
                n,
                matscipy_mean,
                matscipy_std,
                matscipy_min,
                matscipy_max,
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
    matscipy = np.array([r.matscipy_mean for r in results], dtype=float)
    matscipy_min = np.array([r.matscipy_min for r in results], dtype=float)
    matscipy_max = np.array([r.matscipy_max for r in results], dtype=float)
    numpy = np.array([r.numpy_mean for r in results], dtype=float)
    numpy_min = np.array([r.numpy_min for r in results], dtype=float)
    numpy_max = np.array([r.numpy_max for r in results], dtype=float)
    numba = np.array([r.numba_mean for r in results], dtype=float)
    numba_min = np.array([r.numba_min for r in results], dtype=float)
    numba_max = np.array([r.numba_max for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    ax.fill_between(atoms, matscipy_min, matscipy_max, color=PALETTE["reference"], alpha=0.10)
    ax.fill_between(atoms, numpy_min, numpy_max, color=PALETTE["numpy"], alpha=0.10)
    ax.fill_between(atoms, numba_min, numba_max, color=PALETTE["numba"], alpha=0.10)
    ax.plot(atoms, matscipy, "o-", color=PALETTE["reference"], label="matscipy")
    ax.plot(atoms, numpy, "o-", color=PALETTE["numpy"], label="ForgeFF (NumPy)")
    ax.plot(atoms, numba, "o-", color=PALETTE["numba"], label="ForgeFF (Numba)")
    style_axes(ax, title="Stillinger-Weber runtime on Si")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "_static" / "performance",
    )
    args = parser.parse_args()

    results = _benchmark(args.sizes, args.repeats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot(results, args.output_dir / "sw_runtime.png")

    summary = {
        "sizes": args.sizes,
        "repeats": args.repeats,
        "sw": [
            {
                "atoms": r.atoms,
                "matscipy_mean": r.matscipy_mean,
                "matscipy_std": r.matscipy_std,
                "matscipy_min": r.matscipy_min,
                "matscipy_max": r.matscipy_max,
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
    (args.output_dir / "sw_speed_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    for row in results:
        print(
            f"{row.atoms:4d} atoms  matscipy {row.matscipy_mean:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_mean:8.3f} ms  ForgeFF (Numba) {row.numba_mean:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )


if __name__ == "__main__":
    main()
