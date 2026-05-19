"""
ASE calculator
==============
"""

# %%
# This example shows how to attach the EAM calculator to a periodic Al cell.
import numpy as np
from ase.build import bulk

from forgeff.calculator import make_calculator
from forgeff.potentials.eam.data import EAMData

# %%
pot_data = EAMData(
    potential_name="demo",
    species_count=1,
    form="alloy",
    r_grid=np.array([2.0, 3.0, 4.0, 5.0]),
    rho_grid=np.array([0.0, 5.0, 10.0, 15.0]),
    phi_values=np.array([[[0.00, -0.05, -0.02, 0.00]]]),
    rho_values=np.array([[[1.20, 0.80, 0.50, 0.10]]]),
    emb_values=np.array([[0.00, -0.20, -0.35, -0.45]]),
)
pot_data.species = [13]

# %%
atoms = bulk("Al", cubic=True)
atoms.calc = make_calculator(pot_data, engine="numba")
atoms.get_potential_energy()

# %%
# Navigation
# ----------
#
# - Previous: :doc:`/examples/python/2.grade`
# - Back to the landing page: :doc:`/example`
pass
