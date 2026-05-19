``forgeff error``
=================

After training, this command checks how the fitted potential behaves on one or
more datasets.
It is the direct way to print the same compact error table on any structure
file that ForgeFF can read.

Usage
-----

.. code-block:: bash

    forgeff error final.npy examples/toml/data/unary/training.cfg

You can also point it at a TOML potential:

.. code-block:: bash

    forgeff error initial.toml test.traj

Options
-------

- ``potential``: the fitted potential file, such as ``final.npy`` or
  ``final.toml``.
- ``dataset``: one or more structure files to evaluate, for example ``.cfg``
  or ``.traj`` files.
- ``--engine``: engine name. For public examples, use ``numpy`` or ``numba``.
- ``--species``: optional atomic-number order for ``.cfg`` files that do not
  carry species labels.

Output
------

The command prints a compact error table with:

- total energy error
- energy per atom error
- force-component error
- stress-component error

For example, after training a potential you can run:

.. code-block:: bash

    forgeff error final.npy examples/toml/data/unary/training.cfg

See also
--------

- :doc:`train`
- :doc:`evaluate`
- :doc:`grade`
