``forgeff evaluate``
====================

This command calculates energies, forces, and stresses for the configurations written
in ``configurations.initial`` using ``potentials.final``.
The evaluated energies, forces, stresses are written in ``configurations.final``.

Usage
-----

.. code-block:: bash

    forgeff evaluate forgeff.evaluate.toml

``forgeff.evaluate.toml``
=========================

.. literalinclude:: forgeff.evaluate.toml
    :language: toml
