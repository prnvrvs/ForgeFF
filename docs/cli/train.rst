``forgeff train``
=================

This command trains the potential starting from ``potentials.initial`` based on
``configurations.training``.
The trained potential is written in ``potentials.final``.
After the fit, the command automatically prints a compact error summary for
the trained potential on the same training dataset.

Usage
-----

.. code-block:: bash

    forgeff train forgeff.train.toml

or

.. code-block:: bash

    mpirun -np 4 forgeff train forgeff.train.toml

``forgeff.train.toml``
======================

.. literalinclude:: forgeff.train.toml
    :language: toml

If some of the following parameters are already given in the initial potential,
they are treated as the initial guess, which may or may not be optimized
depending on the above setting.

The exact parameter groups depend on the potential family and are defined by
the potential file itself. For example:

- analytical pair forms use named scalar parameters
- tabulated EAM/ADP potentials use named term arrays

See also
--------

- :doc:`error`
