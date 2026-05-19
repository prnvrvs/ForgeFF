import logging
from pathlib import Path

import numpy as np

from forgeff.error.cli import print_error_statistics
from forgeff.io import read, read_potential
from forgeff.evaluate import Evaluator
from forgeff.loss import ErrorPrinter
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
images_all = read(str(dataset_path))

# Use the first half for training and keep the second half for testing.
train_indices = np.arange(5, dtype=int)
test_indices = np.arange(5, 10, dtype=int)
images_training = [images_all[i] for i in train_indices]
images_test = [images_all[i] for i in test_indices]

# %%
# 2. Build the initial potential
# Start from a very small coarse tabulated EAM guess.
potential_file = root / "tests/data_path/nist/Al99.eam.alloy"
source_potential: EAMData = read_potential(str(potential_file))
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
initial_params = pot_data.parameters.copy()
trainer = Trainer(
    pot_data,
    seed=42,
    engine="numba",
    steps=[{"method": "minimize", "kwargs": {"method": "L-BFGS-B", "options": {"maxiter": 3}}}],
)
loss = trainer.train(images_training)
trained_params = pot_data.parameters.copy()

# %%
# 4. Print the training statistics
train_eval = Evaluator(pot_data, engine="numba")
images_train_out = train_eval.evaluate(images_training)
train_errors = ErrorPrinter(images_train_out).calculate()
print(f"training indices: {train_indices.tolist()}")
print(f"testing indices: {test_indices.tolist()}")
print(f"initial loss: {loss(initial_params)}")
print(f"final loss: {loss(trained_params)}")
print_error_statistics(train_errors)

# %%
# Navigation
# ----------
#
# - Next: :doc:`/examples/python/1.evaluate`
# - Back to the landing page: :doc:`/example`
pass
