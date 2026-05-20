"""Benchmark EAM and ADP runtime scaling for reference potentials.

This script compares three paths for each reference case:

- ASE reference
- ForgeFF NumPy
- ForgeFF Numba

It produces three plots:

- EAM alloy runtime vs system size
- EAM Finnis-Sinclair runtime vs system size
- ADP runtime vs system size

The goal is to show how the reference path and the two ForgeFF engines scale
as the system grows.
"""

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
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.eam import EAM as ASEEAM

from forgeff.io import read_potential
from forgeff.potentials.eam.numpy.adp_engine import NumpyADPEngine
from forgeff.potentials.eam.numpy.engine import NumpyEAMEngine
from forgeff.potentials.eam.numba.adp_engine import NumbaADPEngine
from forgeff.potentials.eam.numba.engine import NumbaEAMEngine


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NIST_EAM_ALLOY = ROOT / "tests" / "data_path" / "nist" / "Al99.eam.alloy"
NIST_EAM_FS = ROOT / "tests" / "data_path" / "nist" / "Fe_H_Kumar2023.eam.fs"
NIST_ADP = ROOT / "tests" / "data_path" / "nist" / "AlCu.adp"


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


def _make_al_atoms(n: int) -> Atoms:
    return bulk("Al", "fcc", a=4.05, cubic=True) * (n, n, n)


def _make_fe_h_atoms(n: int) -> Atoms:
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (n, n, n)
    h_pos = 0.5 * atoms.cell.sum(axis=0)
    atoms += Atoms("H", positions=[h_pos])
    rng = np.random.default_rng(2025)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    return atoms


def _time_callable(func, atoms: Atoms, repeats: int, *, before=None) -> tuple[float, float, float]:
    samples = []
    for _ in range(repeats):
        probe = atoms.copy()
        if before is not None:
            before()
        start = perf_counter()
        func(probe)
        samples.append((perf_counter() - start) * 1000.0)
    values = np.array(samples, dtype=float)
    median = float(np.median(values))
    return median, float(values.min()), float(values.max())


def _benchmark_eam_alloy(sizes: list[int], repeats: int) -> list[BenchmarkResult]:
    ase_calc = ASEEAM(potential=str(NIST_EAM_ALLOY), form="alloy")
    numpy_engine = NumpyEAMEngine(read_potential(str(NIST_EAM_ALLOY), form="alloy"))
    numba_engine = NumbaEAMEngine(read_potential(str(NIST_EAM_ALLOY), form="alloy"))

    # Warm up caches / JIT compilation.
    warm = _make_al_atoms(sizes[0])
    ase_calc.calculate(warm.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_engine.calculate(warm.copy())
    numba_engine.calculate(warm.copy())

    results: list[BenchmarkResult] = []
    for n in sizes:
        atoms = _make_al_atoms(n)
        ase_ms, ase_lo, ase_hi = _time_callable(
            lambda a: ase_calc.calculate(a, properties=["energy", "forces", "stress"], system_changes=all_changes),
            atoms,
            repeats,
            before=ase_calc.reset,
        )
        numpy_ms, numpy_lo, numpy_hi = _time_callable(
            numpy_engine.calculate,
            atoms,
            repeats,
        )
        numba_ms, numba_lo, numba_hi = _time_callable(numba_engine.calculate, atoms, repeats)
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


def _benchmark_eam_fs(sizes: list[int], repeats: int) -> list[BenchmarkResult]:
    ase_calc = ASEEAM(potential=str(NIST_EAM_FS), form="fs")
    numpy_engine = NumpyEAMEngine(read_potential(str(NIST_EAM_FS), form="fs"))
    numba_engine = NumbaEAMEngine(read_potential(str(NIST_EAM_FS), form="fs"))

    warm = _make_fe_h_atoms(sizes[0])
    ase_calc.calculate(warm.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_engine.calculate(warm.copy())
    numba_engine.calculate(warm.copy())

    results: list[BenchmarkResult] = []
    for n in sizes:
        atoms = _make_fe_h_atoms(n)
        ase_ms, ase_lo, ase_hi = _time_callable(
            lambda a: ase_calc.calculate(a, properties=["energy", "forces", "stress"], system_changes=all_changes),
            atoms,
            repeats,
            before=ase_calc.reset,
        )
        numpy_ms, numpy_lo, numpy_hi = _time_callable(
            numpy_engine.calculate,
            atoms,
            repeats,
        )
        numba_ms, numba_lo, numba_hi = _time_callable(numba_engine.calculate, atoms, repeats)
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


def _benchmark_adp(sizes: list[int], repeats: int) -> list[BenchmarkResult]:
    ase_calc = ASEEAM(potential=str(NIST_ADP))
    numpy_engine = NumpyADPEngine(read_potential(str(NIST_ADP)))
    numba_engine = NumbaADPEngine(read_potential(str(NIST_ADP)))

    warm = _make_al_atoms(sizes[0])
    ase_calc.calculate(warm.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_engine.calculate(warm.copy())
    numba_engine.calculate(warm.copy())

    results: list[BenchmarkResult] = []
    for n in sizes:
        atoms = _make_al_atoms(n)
        ase_ms, ase_lo, ase_hi = _time_callable(
            lambda a: ase_calc.calculate(a, properties=["energy", "forces", "stress"], system_changes=all_changes),
            atoms,
            repeats,
            before=ase_calc.reset,
        )
        numpy_ms, numpy_lo, numpy_hi = _time_callable(numpy_engine.calculate, atoms, repeats)
        numba_ms, numba_lo, numba_hi = _time_callable(numba_engine.calculate, atoms, repeats)
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


def _plot(results: list[BenchmarkResult], title: str, output: Path) -> None:
    atoms = np.array([r.atoms for r in results], dtype=float)
    ase = np.array([r.ase_ms for r in results], dtype=float)
    ase_err = np.vstack(
        [
            np.array([r.ase_ms - r.ase_lo for r in results], dtype=float),
            np.array([r.ase_hi - r.ase_ms for r in results], dtype=float),
        ]
    )
    numpy = np.array([r.numpy_ms for r in results], dtype=float)
    numpy_err = np.vstack(
        [
            np.array([r.numpy_ms - r.numpy_lo for r in results], dtype=float),
            np.array([r.numpy_hi - r.numpy_ms for r in results], dtype=float),
        ]
    )
    numba = np.array([r.numba_ms for r in results], dtype=float)
    numba_err = np.vstack(
        [
            np.array([r.numba_ms - r.numba_lo for r in results], dtype=float),
            np.array([r.numba_hi - r.numba_ms for r in results], dtype=float),
        ]
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(atoms, ase, yerr=ase_err, fmt="o-", lw=2, capsize=4, label="ASE")
    ax.errorbar(atoms, numpy, yerr=numpy_err, fmt="o-", lw=2, capsize=4, label="ForgeFF (Numpy)")
    ax.errorbar(atoms, numba, yerr=numba_err, fmt="o-", lw=2, capsize=4, label="ForgeFF (Numba)")
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

    eam_alloy = _benchmark_eam_alloy(args.sizes, args.repeats)
    eam_fs = _benchmark_eam_fs(args.sizes, args.repeats)
    adp = _benchmark_adp(args.sizes, args.repeats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot(eam_alloy, "EAM alloy runtime on pure Al", args.output_dir / "eam_alloy_runtime.png")
    _plot(eam_fs, "EAM FS runtime on distorted Fe-H", args.output_dir / "eam_fs_runtime.png")
    _plot(adp, "ADP runtime on pure Al", args.output_dir / "adp_runtime.png")

    summary = {
        "sizes": args.sizes,
        "repeats": args.repeats,
        "eam_alloy": [
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
            for r in eam_alloy
        ],
        "eam_fs": [
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
            for r in eam_fs
        ],
        "adp": [
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
            for r in adp
        ],
    }
    (args.output_dir / "speed_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("EAM alloy:")
    for row in eam_alloy:
        print(
            f"  {row.atoms:4d} atoms  ASE {row.ase_ms:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_ms:8.3f} ms  ForgeFF (Numba) {row.numba_ms:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )
    print("EAM FS:")
    for row in eam_fs:
        print(
            f"  {row.atoms:4d} atoms  ASE {row.ase_ms:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_ms:8.3f} ms  ForgeFF (Numba) {row.numba_ms:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )
    print("ADP:")
    for row in adp:
        print(
            f"  {row.atoms:4d} atoms  ASE {row.ase_ms:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_ms:8.3f} ms  ForgeFF (Numba) {row.numba_ms:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )


if __name__ == "__main__":
    main()
