from __future__ import annotations

import numpy as np
from ase.build import bulk

from forgeff.potentials.ase.custom import CustomPairPotential
from forgeff.potentials.ase.forms import FORMULA_LIBRARY
from forgeff.potentials.ase.numba_pair import NumbaPairPotential


def test_numba_pair_matches_custom_morse() -> None:
    atoms = bulk("Al", "fcc", a=4.05) * (2, 2, 2)
    params = {"De": 0.5, "a": 1.8, "re": 2.9}

    custom = CustomPairPotential(
        expression="De * (exp(-2.0 * a * (r - re)) - 2.0 * exp(-a * (r - re)))",
        parameter_names=["De", "a", "re"],
        cutoff=8.0,
        **params,
    )
    numba_pair = NumbaPairPotential(
        form="morse",
        parameter_names=["De", "a", "re"],
        cutoff=8.0,
        **params,
    )

    custom.calculate(atoms)
    numba_pair.calculate(atoms)

    np.testing.assert_allclose(numba_pair.results["energy"], custom.results["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_pair.results["forces"], custom.results["forces"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_pair.results["stress"], custom.results["stress"], rtol=1e-12, atol=1e-12)


def test_numba_pair_matches_custom_all_builtins() -> None:
    atoms = bulk("Al", "fcc", a=4.05) * (2, 2, 2)

    for form, spec in FORMULA_LIBRARY.items():
        if spec.get("variable", "r") != "r":
            continue

        params = [0.5 * (lo + hi) for lo, hi in spec["default_bounds"]]
        kwargs = dict(zip(spec["params"], params))

        custom = CustomPairPotential(
            expression=spec["formula"],
            parameter_names=spec["params"],
            cutoff=8.0,
            **kwargs,
        )
        numba_pair = NumbaPairPotential(
            form=form,
            parameter_names=spec["params"],
            cutoff=8.0,
            **kwargs,
        )

        custom.calculate(atoms)
        numba_pair.calculate(atoms)

        np.testing.assert_allclose(
            numba_pair.results["energy"],
            custom.results["energy"],
            rtol=1e-10,
            atol=1e-8,
            err_msg=f"energy mismatch for form {form}",
        )
        np.testing.assert_allclose(
            numba_pair.results["forces"],
            custom.results["forces"],
            rtol=1e-10,
            atol=1e-8,
            err_msg=f"forces mismatch for form {form}",
        )
        np.testing.assert_allclose(
            numba_pair.results["stress"],
            custom.results["stress"],
            rtol=1e-10,
            atol=1e-8,
            err_msg=f"stress mismatch for form {form}",
        )
