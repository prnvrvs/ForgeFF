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
from forgeff.potentials.eam.numpy.eam_engine import NumpyEAMEngine
from forgeff.potentials.eam.numba.adp_engine import NumbaADPEngine
from forgeff.potentials.eam.numba.eam_engine import NumbaEAMEngine
from benchmarks.plot_style import PALETTE, apply_publication_style, save_figure, style_axes


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


def _make_al_atoms(n: int) -> Atoms:
    return bulk("Al", "fcc", a=4.05, cubic=True) * (n, n, n)


def _make_distorted_al_atoms(n: int) -> Atoms:
    atoms = _make_al_atoms(n)
    strain = np.array(
        [
            [1.00, 0.015, 0.010],
            [0.000, 0.985, 0.012],
            [0.000, 0.000, 1.020],
        ],
        dtype=float,
    )
    atoms.cell = atoms.cell.array @ strain
    rng = np.random.default_rng(2026 + n)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    atoms.wrap()
    return atoms


def _make_fe_h_atoms(n: int) -> Atoms:
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True) * (n, n, n)
    h_pos = 0.5 * atoms.cell.sum(axis=0)
    atoms += Atoms("H", positions=[h_pos])
    rng = np.random.default_rng(2025)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    return atoms


def _time_callable(func, atoms: Atoms, repeats: int, *, before=None) -> tuple[float, float, float, float]:
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
        ase_mean, ase_std, ase_min, ase_max = _time_callable(
            lambda a: ase_calc.calculate(a, properties=["energy", "forces", "stress"], system_changes=all_changes),
            atoms,
            repeats,
            before=ase_calc.reset,
        )
        numpy_mean, numpy_std, numpy_min, numpy_max = _time_callable(
            numpy_engine.calculate,
            atoms,
            repeats,
        )
        numba_mean, numba_std, numba_min, numba_max = _time_callable(numba_engine.calculate, atoms, repeats)
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
        ase_mean, ase_std, ase_min, ase_max = _time_callable(
            lambda a: ase_calc.calculate(a, properties=["energy", "forces", "stress"], system_changes=all_changes),
            atoms,
            repeats,
            before=ase_calc.reset,
        )
        numpy_mean, numpy_std, numpy_min, numpy_max = _time_callable(
            numpy_engine.calculate,
            atoms,
            repeats,
        )
        numba_mean, numba_std, numba_min, numba_max = _time_callable(numba_engine.calculate, atoms, repeats)
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


def _benchmark_adp(sizes: list[int], repeats: int) -> list[BenchmarkResult]:
    ase_calc = ASEEAM(potential=str(NIST_ADP))
    numpy_engine = NumpyADPEngine(read_potential(str(NIST_ADP)))
    numba_engine = NumbaADPEngine(read_potential(str(NIST_ADP)))

    warm = _make_distorted_al_atoms(sizes[0])
    ase_calc.calculate(warm.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numpy_engine.calculate(warm.copy())
    numba_engine.calculate(warm.copy())

    results: list[BenchmarkResult] = []
    for n in sizes:
        atoms = _make_distorted_al_atoms(n)
        ase_mean, ase_std, ase_min, ase_max = _time_callable(
            lambda a: ase_calc.calculate(a, properties=["energy", "forces", "stress"], system_changes=all_changes),
            atoms,
            repeats,
            before=ase_calc.reset,
        )
        numpy_mean, numpy_std, numpy_min, numpy_max = _time_callable(numpy_engine.calculate, atoms, repeats)
        numba_mean, numba_std, numba_min, numba_max = _time_callable(numba_engine.calculate, atoms, repeats)
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


def _plot(results: list[BenchmarkResult], title: str, output: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.fill_between(atoms, ase_min, ase_max, color=PALETTE["reference"], alpha=0.10)
    ax.fill_between(atoms, numpy_min, numpy_max, color=PALETTE["numpy"], alpha=0.10)
    ax.fill_between(atoms, numba_min, numba_max, color=PALETTE["numba"], alpha=0.10)
    ax.plot(atoms, ase, "o-", color=PALETTE["reference"], label="ASE")
    ax.plot(atoms, numpy, "o-", color=PALETTE["numpy"], label="ForgeFF (NumPy)")
    ax.plot(atoms, numba, "o-", color=PALETTE["numba"], label="ForgeFF (Numba)")
    style_axes(ax, title=title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--repeats", type=int, default=5)
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
            for r in eam_alloy
        ],
        "eam_fs": [
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
            for r in eam_fs
        ],
        "adp": [
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
            f"  {row.atoms:4d} atoms  ASE {row.ase_mean:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_mean:8.3f} ms  ForgeFF (Numba) {row.numba_mean:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )
    print("EAM FS:")
    for row in eam_fs:
        print(
            f"  {row.atoms:4d} atoms  ASE {row.ase_mean:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_mean:8.3f} ms  ForgeFF (Numba) {row.numba_mean:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )
    print("ADP:")
    for row in adp:
        print(
            f"  {row.atoms:4d} atoms  ASE {row.ase_mean:8.3f} ms  "
            f"ForgeFF (Numpy) {row.numpy_mean:8.3f} ms  ForgeFF (Numba) {row.numba_mean:8.3f} ms  "
            f"speedup {row.speedup:5.2f}x"
        )


if __name__ == "__main__":
    main()
