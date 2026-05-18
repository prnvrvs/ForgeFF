TOML Examples
=============

The `examples/toml` folder is the quickest way to see the format in action.
It shows you the file shapes, the field names, and the small pieces that fit
together into a full potential definition.

It contains three families of fitting examples:

- pairwise analytic potentials
- tabulated EAM potentials
- tabulated ADP potentials
- a tiny shared `training.cfg` input for smoke testing

For a single-file overview of the supported TOML syntax, see
:download:`examples/toml/potential.toml <../../examples/toml/potential.toml>`.

These examples are intentionally small and friendly rather than
production-ready. Their job is to show the structure of the files without
burying you in numbers.

How to read the examples
------------------------

The TOML format is deliberately explicit, but that is what makes it easy to
read back later:

- built-in pair forms use ``form = "..."`` and fixed parameter order
- user-defined pair expressions use ``expression = "..."`` and ``parameter_names``
- the evaluator choice is stored in ``backend``
- ``backend`` is either ``numpy`` or ``numba``
- tabulated EAM/ADP use explicit species names and term blocks
- the shared ``data/training.cfg`` file is just a smoke-test configuration

In other words: the TOML file tells ForgeFF what to fit, what to call the
terms, and where to start.

When you want to run a real fit, replace the shared training input with your
own configuration list and expand the tables or expressions to match your
target material.

Directory layout
-----------------

.. code-block:: text

    examples/toml/
    ├── README.md
    ├── data/
    │   └── training.cfg
    ├── pairwise/
    │   ├── built_in.toml
    │   ├── custom_expression.toml
    │   ├── forgeff.train.toml
    │   └── README.md
    ├── eam/
    │   ├── initial.toml
    │   ├── forgeff.train.toml
    │   └── README.md
    └── adp/
        ├── initial.toml
        ├── forgeff.train.toml
        └── README.md

Pairwise analytic example
-------------------------

`examples/toml/pairwise/built_in.toml` shows a built-in analytical pair form.
It uses the registry-backed `double_morse` form, so you can see how the file
names the equation and lists the initial parameters in one place.

`examples/toml/pairwise/custom_expression.toml` shows a user-defined equation.
Use this pattern when you have a one-off form that is easier to write directly
than to add to the registry.

Tabulated EAM example
---------------------

`examples/toml/eam/initial.toml` shows the minimal alloy EAM structure:

- `[potential]` selects the `eam` family
- `[species]` defines the canonical species order
- `[grids]` defines the uniform radial and density grids
- `[pair.*]` stores the pair table
- `[density.*]` stores the electron-density table
- `[embedding.*]` stores the embedding table

If you remember only one thing, remember this: each named block is one term
that gets fitted.

Tabulated ADP example
---------------------

`examples/toml/adp/initial.toml` extends the EAM layout with:

- `[dipole.*]` for the ADP dipole tables
- `[quadrupole.*]` for the ADP quadrupole tables

That means ADP is just EAM plus two extra tabulated pieces.

Training settings
-----------------

The matching `forgeff.train.toml` files in each subdirectory show how to point
`potentials.initial` at the TOML potential definition without having to
memorize path conventions.

Relative paths inside the example `forgeff.train.toml` files are resolved
relative to the setting file itself. That means the examples work from any
working directory as long as the folder layout is preserved.

The shared `data/training.cfg` file is a tiny smoke-test input. Replace it
with your own configuration list before running a real fit.

This page is the practical companion to the :ref:`toml-spec` page. That page
describes the schema itself. This one shows how the example files are laid out
on disk and how to use them.

Quick start
-----------

To inspect the examples without editing paths, run the CLI against the
provided setting file directly:

.. code-block:: bash

    forgeff train examples/toml/pairwise/forgeff.train.toml
    forgeff train examples/toml/eam/forgeff.train.toml
    forgeff train examples/toml/adp/forgeff.train.toml

Each file is intentionally minimal. In practice you will replace the shared
`data/training.cfg` path with the actual configuration list you want to fit
against.

If you want to start from NIST potentials instead, the repo also includes:

- `tests/data_path/nist/Al99.eam.alloy`
- `tests/data_path/nist/AlCu.adp`
