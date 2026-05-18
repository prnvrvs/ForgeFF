``forgeff grade``
=================

This command calculates the extrapolation grades for the configurations written
in ``configurations.initial`` using ``potentials.final`` and ``configurations.training``.
The evaluated extrapolation grades are written in ``configurations.final``.

Usage
-----

.. code-block:: bash

    forgeff grade forgeff.grade.toml

``forgeff.grade.toml``
======================

.. literalinclude:: forgeff.grade.toml
    :language: toml
