from __future__ import annotations

import numpy as np
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.eam import EAM as ASEEAM

from forgeff.io import read_potential
from forgeff.potentials.eam.numba.adp_engine import NumbaADPEngine
from forgeff.potentials.eam.numba.engine import NumbaEAMEngine


def test_nist_al99_eam_matches_ase() -> None:
    atoms = bulk("Al", "fcc", a=4.05) * (4, 4, 4)
    ase_calc = ASEEAM(potential="tests/data_path/nist/Al99.eam.alloy")
    numba_engine = NumbaEAMEngine(read_potential("tests/data_path/nist/Al99.eam.alloy"))

    ase_calc.calculate(atoms.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numba_res = numba_engine.calculate(atoms.copy())

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-12, atol=1e-12)


def test_nist_alcu_adp_matches_ase_for_pure_al() -> None:
    atoms = bulk("Al", "fcc", a=4.05) * (4, 4, 4)
    ase_calc = ASEEAM(potential="tests/data_path/nist/AlCu.adp")
    numba_engine = NumbaADPEngine(read_potential("tests/data_path/nist/AlCu.adp"))

    ase_calc.calculate(atoms.copy(), properties=["energy", "forces", "stress"], system_changes=all_changes)
    numba_res = numba_engine.calculate(atoms.copy())

    np.testing.assert_allclose(numba_res["energy"], ase_calc.results["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["forces"], ase_calc.results["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_res["stress"], ase_calc.results["stress"], rtol=1e-12, atol=1e-12)
