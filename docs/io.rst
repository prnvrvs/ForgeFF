File IO
=======

ForgeFF has parsers for `the file formats of the MLIP codes developed by
Shapeev and co-workers <https://gitlab.com/users/ashapeev/projects>`_.

- ``.cfg``: Atomic configurations
- ``.toml``: ForgeFF potential definitions for analytical pair, EAM, and ADP forms

``.cfg``
--------

.. automodule:: forgeff.io.mlip.cfg
    :members:

``.toml``
---------

ForgeFF also supports TOML-defined potentials through :func:`forgeff.io.read_potential`.
The TOML format is the main user-facing format for ForgeFF potentials and
training workflows. The full TOML specification is documented in
:ref:`toml-spec`.

This IO page is intentionally short:

- ``.cfg`` describes atomic configurations
- ``.toml`` describes potentials and initial guesses

For a copyable overview of the format and examples, see the TOML spec page
and :download:`examples/toml/potential.toml <../examples/toml/potential.toml>`.
