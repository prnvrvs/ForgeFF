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


@dataclass
class BenchmarkResult:
    atoms: int
    n: int
    matscipy_ms: float
    numpy_ms: float
    numba_ms: float

    @property
    def speedup(self) -> float:
        return self.matscipy_ms / self.numba_ms if self.numba_ms > 0 else float("inf")


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


def _time_callable(func, atoms, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        probe = atoms.copy()
        start = perf_counter()
        func(probe)
        samples.append((perf_counter() - start) * 1000.0)
    return float(np.median(samples))


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

        matscipy_ms = _time_callable(run_matscipy, atoms, repeats)
        numpy_ms = _time_callable(run_numpy, atoms, repeats)
        numba_ms = _time_callable(run_numba, atoms, repeats)
        results.append(BenchmarkResult(len(atoms), n, matscipy_ms, numpy_ms, numba_ms))
    return results


def _plot(results: list[BenchmarkResult], output: Path) -> None:
    atoms = np.array([r.atoms for r in results], dtype=float)
    matscipy = np.array([r.matscipy_ms for r in results], dtype=float)
    numpy = np.array([r.numpy_ms for r in results], dtype=float)
    numba = np.array([r.numba_ms for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(atoms, matscipy, "o-", lw=2, label="matscipy")
    ax.plot(atoms, numpy, "o-", lw=2, label="ForgeFF (Numpy)")
    ax.plot(atoms, numba, "o-", lw=2, label="ForgeFF (Numba)")
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Median evaluation time (ms)")
    ax.set_title("Stillinger-Weber runtime on Si")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--repeats", type=int, default=3)
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
                "matscipy_ms": r.matscipy_ms,
                "numpy_ms": r.numpy_ms,
                "numba_ms": r.numba_ms,
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
            f"{row.atoms:4d} atoms  matscipy {row.matscipy_ms:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_ms:8.3f} ms  ForgeFF (Numba) {row.numba_ms:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )


if __name__ == "__main__":
    main()
