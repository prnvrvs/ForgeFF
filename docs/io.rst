File IO
=======

ForgeFF has parsers for `the file formats of the MLIP codes developed by
Shapeev and co-workers <https://gitlab.com/users/ashapeev/projects>`_.

ForgeFF can read and write MLIP-style ``.cfg`` reference-configuration files
through :func:`forgeff.io.mlip.cfg.read_cfg` and
:func:`forgeff.io.mlip.cfg.write_cfg`.

ForgeFF can also read and write potfit-style force configurations with
``#N``/``#C``/``#X``/``#Y``/``#Z``/``#W``/``#E``/``#S``/``#F`` headers through
:func:`forgeff.io.potfit.read_force` and
:func:`forgeff.io.potfit.write_force`.

- ``.cfg``: MLIP-style atomic configurations
- ``.force`` / ``.potfit``: potfit-style force configurations
- ``.toml``: ForgeFF potential definitions for analytical pair, EAM, ADP, SW, and Tersoff forms
- ``.tersoff``: LAMMPS-style Tersoff potential files

``.cfg``
--------

.. automodule:: forgeff.io.mlip.cfg
    :members:

This section covers the ``.cfg`` reader and writer used by MLIP-style
structured reference-configuration workflows.

``.force`` / ``.potfit``
------------------------

.. automodule:: forgeff.io.potfit
    :members:

This section covers the potfit force-file format used by
``ase2force`` / ``vasp2force`` and the matching ForgeFF reader and writer.

``.toml``
---------

ForgeFF also supports TOML-defined potentials through :func:`forgeff.io.read_potential`.
The TOML format is the main user-facing format for ForgeFF potentials and
training workflows. The full TOML specification is documented in
:ref:`toml-spec`.

ForgeFF also reads and writes LAMMPS-style ``.tersoff`` files for the standard
Tersoff family through the same IO layer.

Implementation
~~~~~~~~~~~~~~

.. autofunction:: forgeff.io.read_potential
.. autofunction:: forgeff.io.write_potential
.. autofunction:: forgeff.io.toml.read_potential_toml

This IO page is intentionally short:

- ``.cfg`` describes atomic configurations
- ``.toml`` describes potentials and initial guesses

For a copyable overview of the format and examples, see the TOML spec page
and :download:`examples/toml/potential.toml <../examples/toml/potential.toml>`.
