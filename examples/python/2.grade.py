import logging
from pathlib import Path

import numpy as np

from forgeff.grade import Grader
from forgeff.io import read
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
# Load the shared unary Al dataset from the TOML example folder.
root = Path(__file__).resolve().parents[2]
dataset_path = root / "examples/toml/data/unary/training.cfg"
images_all = read(str(dataset_path), species=[13])

# Use most images for training so the maxvol grading step has enough rows.
train_indices = np.arange(7, dtype=int)
test_indices = np.arange(7, 10, dtype=int)
images_training = [images_all[i] for i in train_indices]
images_test = [images_all[i] for i in test_indices]

# %%
# 2. Build the initial potential
# Start from a very small coarse tabulated EAM guess.
potential_file = root / "tests/data_path/nist/Al99.eam.alloy"
source_potential: EAMData = read_potential(str(potential_file))
r_grid = np.linspace(source_potential.r_grid[0], source_potential.r_grid[-1], 2)
rho_grid = np.linspace(source_potential.rho_grid[0], source_potential.rho_grid[-1], 2)
spc = source_potential.species_count
phi_values = np.empty((spc, spc, len(r_grid)), dtype=float)
rho_values = np.empty((spc, spc, len(r_grid)), dtype=float)
emb_values = np.empty((spc, len(rho_grid)), dtype=float)
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

# %%
# Navigation
# ----------
#
# - Previous: :doc:`/examples/python/1.evaluate`
pass
