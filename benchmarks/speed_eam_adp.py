"""Benchmark EAM and ADP runtime scaling for Al supercells.

This script compares the NumPy-backed ASE EAM path against the Numba EAM path
for pure Al, and the same for ADP using the NIST Al-Cu ADP file on pure Al.
It produces two plots:

- EAM runtime vs system size
- ADP runtime vs system size

The goal is to show how the faster JIT path behaves as the system grows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.eam import EAM as ASEEAM

from forgeff.io import read_potential
from forgeff.potentials.eam.numpy.engine import ASEAMEngine
from forgeff.potentials.eam.numba.adp_engine import NumbaADPEngine
from forgeff.potentials.eam.numba.engine import NumbaEAMEngine


ROOT = Path(__file__).resolve().parents[1]
NIST_EAM = ROOT / "tests" / "data_path" / "nist" / "Al99.eam.alloy"
NIST_ADP = ROOT / "tests" / "data_path" / "nist" / "AlCu.adp"


@dataclass
class BenchmarkResult:
    atoms: int
    n: int
    ase_ms: float
    numba_ms: float

    @property
    def speedup(self) -> float:
        return self.ase_ms / self.numba_ms if self.numba_ms > 0 else float("inf")


def _make_al_atoms(n: int) -> Atoms:
    return bulk("Al", "fcc", a=4.05, cubic=True) * (n, n, n)


def _time_callable(func, atoms: Atoms, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        probe = atoms.copy()
        start = perf_counter()
        func(probe)
        samples.append((perf_counter() - start) * 1000.0)
    return float(np.median(samples))


def _benchmark_eam(sizes: list[int], repeats: int) -> list[BenchmarkResult]:
    ase_calc = ASEEAM(potential=str(NIST_EAM))
    numba_engine = NumbaEAMEngine(read_potential(str(NIST_EAM)))

    # Warm up caches / JIT compilation.
    warm = _make_al_atoms(sizes[0])
    ase_calc.calculate(warm.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numba_engine.calculate(warm.copy())

    results: list[BenchmarkResult] = []
    for n in sizes:
        atoms = _make_al_atoms(n)
        ase_ms = _time_callable(
            lambda a: ase_calc.calculate(a, properties=["energy", "forces", "stress"], system_changes=all_changes),
            atoms,
            repeats,
        )
        numba_ms = _time_callable(numba_engine.calculate, atoms, repeats)
        results.append(BenchmarkResult(len(atoms), n, ase_ms, numba_ms))
    return results


def _benchmark_adp(sizes: list[int], repeats: int) -> list[BenchmarkResult]:
    ase_calc = ASEEAM(potential=str(NIST_ADP))
    numba_engine = NumbaADPEngine(read_potential(str(NIST_ADP)))

    warm = _make_al_atoms(sizes[0])
    ase_calc.calculate(warm.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numba_engine.calculate(warm.copy())

    results: list[BenchmarkResult] = []
    for n in sizes:
        atoms = _make_al_atoms(n)
        ase_ms = _time_callable(
            lambda a: ase_calc.calculate(a, properties=["energy", "forces", "stress"], system_changes=all_changes),
            atoms,
            repeats,
        )
        numba_ms = _time_callable(numba_engine.calculate, atoms, repeats)
        results.append(BenchmarkResult(len(atoms), n, ase_ms, numba_ms))
    return results


def _plot(results: list[BenchmarkResult], title: str, output: Path) -> None:
    atoms = np.array([r.atoms for r in results], dtype=float)
    ase = np.array([r.ase_ms for r in results], dtype=float)
    numba = np.array([r.numba_ms for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(atoms, ase, "o-", lw=2, label="ASE / NumPy")
    ax.plot(atoms, numba, "o-", lw=2, label="Numba")
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Median evaluation time (ms)")
    ax.set_title(title)
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
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "_static" / "performance",
    )
    args = parser.parse_args()

    eam = _benchmark_eam(args.sizes, args.repeats)
    adp = _benchmark_adp(args.sizes, args.repeats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot(eam, "EAM runtime on pure Al", args.output_dir / "eam_runtime.png")
    _plot(adp, "ADP runtime on pure Al", args.output_dir / "adp_runtime.png")

    summary = {
        "sizes": args.sizes,
        "repeats": args.repeats,
        "eam": [
            {"atoms": r.atoms, "ase_ms": r.ase_ms, "numba_ms": r.numba_ms, "speedup": r.speedup}
            for r in eam
        ],
        "adp": [
            {"atoms": r.atoms, "ase_ms": r.ase_ms, "numba_ms": r.numba_ms, "speedup": r.speedup}
            for r in adp
        ],
    }
    (args.output_dir / "speed_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("EAM:")
    for row in eam:
        print(f"  {row.atoms:4d} atoms  ASE {row.ase_ms:8.3f} ms  Numba {row.numba_ms:8.3f} ms  speedup {row.speedup:5.2f}x")
    print("ADP:")
    for row in adp:
        print(f"  {row.atoms:4d} atoms  ASE {row.ase_ms:8.3f} ms  Numba {row.numba_ms:8.3f} ms  speedup {row.speedup:5.2f}x")


if __name__ == "__main__":
    main()
