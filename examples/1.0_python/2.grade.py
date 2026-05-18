"""
Grader
======
"""

from pathlib import Path
import logging
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from ase.build import bulk
from ase.calculators.eam import EAM as ASEEAM
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.units import fs

from forgeff.grade import Grader
from forgeff.io import read_potential
from forgeff.potentials.eam.data import EAMData
from forgeff.train import Trainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy_logger in (
    "forgeff.loss",
    "forgeff.train.trainer",
    "forgeff.evaluate.evaluator",
    "forgeff.grade.grader",
    "forgeff.optimizers.scipy",
    "forgeff.optimizers.anneal",
    "forgeff.optimizers.ga",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# %%
# 1. Generate the data
# Find the repository root so the example works from any working directory.
root = Path.cwd().resolve()
for candidate in (root, *root.parents):
    if (candidate / "tests/data_path/nist/Al99.eam.alloy").exists():
        root = candidate
        break
else:
    raise FileNotFoundError("Could not locate the repository root.")

# Use the NIST Al EAM file as the reference source potential.
potential_file = root / "tests/data_path/nist/Al99.eam.alloy"
source_potential: EAMData = read_potential(str(potential_file))
reference_calc = ASEEAM(potential=str(potential_file), elements=["Al"], form="alloy")

# Build a small MD trajectory of 100 Al snapshots.
atoms = bulk("Al", cubic=True) * (2, 2, 2)
atoms.calc = reference_calc
MaxwellBoltzmannDistribution(atoms, temperature_K=600)
dyn = VelocityVerlet(atoms, 1.0 * fs)
images_all = []
for step in range(100):
    if step > 0:
        dyn.run(1)
    frame = atoms.copy()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    frame.calc = SinglePointCalculator(frame, energy=energy, forces=forces, stress=stress)
    images_all.append(frame)

# Use the first half for training and keep the second half for testing.
train_indices = np.arange(50, dtype=int)
test_indices = np.arange(50, 100, dtype=int)
images_training = [images_all[i] for i in train_indices]
images_test = [images_all[i] for i in test_indices]

# %%
# 2. Build the initial potential
# Start from a very small coarse tabulated EAM guess.
r_grid = np.linspace(source_potential.r_grid[0], source_potential.r_grid[-1], 3)
rho_grid = np.linspace(source_potential.rho_grid[0], source_potential.rho_grid[-1], 3)
spc = source_potential.species_count
phi_values = np.empty((spc, spc, 3), dtype=float)
rho_values = np.empty((spc, spc, 3), dtype=float)
emb_values = np.empty((spc, 3), dtype=float)
for i in range(spc):
    emb_values[i] = np.interp(rho_grid, source_potential.rho_grid, source_potential.emb_values[i])
    for j in range(spc):
        phi_values[i, j] = np.interp(r_grid, source_potential.r_grid, source_potential.phi_values[i, j])
        rho_values[i, j] = np.interp(r_grid, source_potential.r_grid, source_potential.rho_values[i, j])
pot_data = EAMData(
    potential_name=f"{source_potential.potential_name}-coarse",
    form=source_potential.form,
    r_grid=r_grid,
    rho_grid=rho_grid,
    phi_values=phi_values,
    rho_values=rho_values,
    emb_values=emb_values,
)
pot_data.species = source_potential.species

# %%
# 3. Train the potential
trainer = Trainer(
    pot_data,
    seed=42,
    engine="numba",
    steps=[{"method": "minimize", "kwargs": {"method": "L-BFGS-B", "options": {"maxiter": 3}}}],
)
trainer.train(images_training)

# %%
# 4. Grade the fitted potential
grader = Grader(pot_data, seed=42, engine="numba")
grader.update(images_training)
buffer = StringIO()
with redirect_stdout(buffer):
    images_train_out = grader.grade(images_training)
    images_test_out = grader.grade(images_test)

# %%
# 5. Summarize the grades
print(f"trained on: {train_indices.tolist()}")
print(f"testing on: {test_indices.tolist()}")
train_grades = [image.calc.results["MV_grade"] for image in images_train_out]
test_grades = [image.calc.results["MV_grade"] for image in images_test_out]
print(f"graded training set: {len(train_grades)} configurations")
print(f"graded testing set: {len(test_grades)} configurations")
print(f"training first grade: {train_grades[0]}")
print(f"training minimum grade: {min(train_grades)}")
print(f"training maximum grade: {max(train_grades)}")
print(f"testing first grade: {test_grades[0]}")
print(f"testing minimum grade: {min(test_grades)}")
print(f"testing maximum grade: {max(test_grades)}")
