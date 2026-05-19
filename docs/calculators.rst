ASE calculators
===============

ForgeFF exposes calculators for these supported potential families:

- tabulated EAM
- tabulated ADP
- Tersoff-style many-body potentials
- analytical pair potentials and direct ASE calculators

The public entry point is :func:`forgeff.calculator.make_calculator`.

Basic usage
-----------

.. code-block:: python

    from forgeff.calculator import make_calculator

    atoms.calc = make_calculator(potential_data, engine="numba")

The calculator is selected from the potential data object:

- :class:`forgeff.potentials.eam.data.EAMData` -> EAM calculator
- :class:`forgeff.potentials.eam.adp_data.ADPData` -> ADP calculator
- :class:`forgeff.potentials.ase.data.ASEData` -> generic ASE wrapper

Supported engines
-----------------

EAM
~~~

For EAM, the following backends are currently wired:

- ``numpy``: NumPy-backed EAM engine
- ``numba``: Numba-accelerated EAM engine

Example:

.. code-block:: python

    from forgeff.calculator import make_calculator

    atoms.calc = make_calculator(eam_data, engine="numba")

ADP
~~~

For ADP, the current backend is:

- ``numba``: Numba-accelerated ADP engine

Example:

.. code-block:: python

    from forgeff.calculator import make_calculator

    atoms.calc = make_calculator(adp_data, engine="numba")

Tersoff
~~~~~~~

For Tersoff-style many-body potentials, the current backend is:

- ``numba``: Numba-accelerated Tersoff engine

Example:

.. code-block:: python

    from forgeff.calculator import make_calculator

    atoms.calc = make_calculator(tersoff_data, engine="numba")

Analytical pair potentials and direct ASE calculators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For analytical pair potentials, ForgeFF uses the pair backend selected by
``backend``.

The supported backend names are:

- ``numpy``: Python-backed analytical evaluation
- ``numba``: JIT-backed analytical evaluation

For direct ASE calculators, ForgeFF can also load standard ASE classes such as
``LennardJones``, ``MorsePotential``, ``EMT``, and fully qualified ASE
calculator import paths.

Example:

.. code-block:: python

    from forgeff.calculator import make_calculator

    atoms.calc = make_calculator(ase_data, engine="generic")

Analytical gradients
---------------------

The calculator interface returns energy, forces, and stress, but parameter
Jacobians are still incomplete:

- analytical pair expressions provide distance derivatives
- the optimizer-side parameter Jacobians are mostly numerical placeholders
- the NumPy-backed EAM path currently does not provide analytical parameter
  Jacobians

This is enough for evaluation and for current smoke-test fitting workflows,
but it is not yet a full analytical-gradient implementation for every backend.

Related pages
-------------

- :doc:`toml`
- :doc:`analytical`
- :doc:`theory`
