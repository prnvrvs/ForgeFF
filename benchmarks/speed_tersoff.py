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


@dataclass
class BenchmarkResult:
    atoms: int
    n: int
    ase_ms: float
    ase_lo: float
    ase_hi: float
    numpy_ms: float
    numpy_lo: float
    numpy_hi: float
    numba_ms: float
    numba_lo: float
    numba_hi: float

    @property
    def speedup(self) -> float:
        return self.ase_ms / self.numba_ms if self.numba_ms > 0 else float("inf")


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


def _time_callable(func, atoms, repeats: int, *, before=None) -> tuple[float, float, float]:
    samples = []
    for _ in range(repeats):
        probe = atoms.copy()
        if before is not None:
            before()
        start = perf_counter()
        func(probe)
        samples.append((perf_counter() - start) * 1000.0)
    values = np.array(samples, dtype=float)
    return float(np.median(values)), float(values.min()), float(values.max())


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
        ase_ms, ase_lo, ase_hi = _time_callable(
            lambda a: _evaluate(ase_calc, a),
            atoms,
            repeats,
            before=ase_calc.reset,
        )
        numpy_ms, numpy_lo, numpy_hi = _time_callable(lambda a: _evaluate(numpy_calc, a), atoms, repeats)
        numba_ms, numba_lo, numba_hi = _time_callable(lambda a: _evaluate(numba_calc, a), atoms, repeats)
        results.append(
            BenchmarkResult(
                len(atoms),
                n,
                ase_ms,
                ase_lo,
                ase_hi,
                numpy_ms,
                numpy_lo,
                numpy_hi,
                numba_ms,
                numba_lo,
                numba_hi,
            )
        )
    return results


def _plot(results: list[BenchmarkResult], output: Path) -> None:
    atoms = np.array([r.atoms for r in results], dtype=float)
    ase = np.array([r.ase_ms for r in results], dtype=float)
    numpy = np.array([r.numpy_ms for r in results], dtype=float)
    numba = np.array([r.numba_ms for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(atoms, ase, "o-", lw=2, label="ASE")
    ax.plot(atoms, numpy, "o-", lw=2, label="ForgeFF (Numpy)")
    ax.plot(atoms, numba, "o-", lw=2, label="ForgeFF (Numba)")
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Median evaluation time (ms)")
    ax.set_title("Tersoff runtime on distorted Si")
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
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--repeats", type=int, default=3)
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
                "ase_ms": r.ase_ms,
                "ase_lo": r.ase_lo,
                "ase_hi": r.ase_hi,
                "numpy_ms": r.numpy_ms,
                "numpy_lo": r.numpy_lo,
                "numpy_hi": r.numpy_hi,
                "numba_ms": r.numba_ms,
                "numba_lo": r.numba_lo,
                "numba_hi": r.numba_hi,
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
            f"{row.atoms:5d} atoms  ASE {row.ase_ms:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_ms:8.3f} ms  ForgeFF (Numba) {row.numba_ms:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )


if __name__ == "__main__":
    main()
