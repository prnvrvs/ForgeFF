.. _toml-spec:

TOML Specification
==================

ForgeFF uses TOML as the main user-facing format for potential definitions and
training settings. The goal is to keep the files readable for humans while
still making the potential family, term order, and initial guess explicit for
the code. The runtime engine lives in the training setting file under
``[common].engine``.

This page is the authoritative reference for the ForgeFF TOML schema. For
worked examples, see:

- :doc:`examples/toml`
- :doc:`example`

If you want a starter file, ``forgeff template`` can generate a valid TOML
template for analytical, EAM, ADP, Stillinger-Weber, or Tersoff potentials.

Template generation
-------------------

The template command keeps the potential file small and readable while giving
you a valid starting point for each family:

.. code-block:: bash

   forgeff template analytical --species Al Cu --form morse
   forgeff template eam --species Fe Ni --form alloy
   forgeff template adp --species Al Cu
   forgeff template sw --species Si
   forgeff template tersoff --species Si C

Use ``--output`` when you want the template written to a file instead of
printed to the terminal:

.. code-block:: bash

   forgeff template analytical --species Al Cu --form morse --output initial.toml

The generated file is only the potential definition. The matching
``forgeff.train.toml`` still carries ``[common].engine`` and the rest of the
runtime settings.

If you need a LAMMPS-compatible output file for EAM or ADP, see
:doc:`cli/export`. That page covers the supported formats and command-line
usage.

The example set includes unary and binary layouts for the supported families:

- pairwise: unary and binary
- EAM: unary alloy, binary alloy, unary FS, and binary FS
- ADP: unary and binary
- Tersoff: explicit species-triple tables and templates

Overview
--------

Think of the TOML file as the conversation between you and ForgeFF. It tells
the code what kind of model you want, which equations or tables to use, and
what values should seed the fit.

The TOML schema supports three potential families:

- analytical pair potentials
- tabulated EAM potentials
- tabulated ADP potentials
- Stillinger-Weber potentials
- Tersoff potentials

The general structure is:

- ``[potential]``: selects the family and calculation mode
- ``[species]``: fixes the canonical species order for EAM/ADP
- ``[triplet.*]``: stores Tersoff species-triple tables
- ``[grids]``: defines uniform radial and density grids for tabulated models
- term blocks such as ``[pair.*]``, ``[density.*]``, ``[embedding.*]``,
  ``[dipole.*]``, and ``[quadrupole.*]``
- ``[parameters]``: optional container for analytical starting values

Tabulated EAM and ADP term blocks also accept ``optimize = false`` to freeze
that block during fitting. By default, blocks are optimized. The same freeze
logic is available in Python by setting ``pot.optimized`` to the block names
you want to update.

Multispecies analytical pair terms use the same idea: set ``optimize = false``
on a ``[pair.*]`` block to keep that pair fixed while training the remaining
pair blocks.

For example, a binary Morse fit can freeze the pretrained ``Al-Al`` block and
keep the cross terms trainable:

.. code-block:: toml

    [potential]
    family = "analytical"
    form = "morse"
    cutoff = 8.0

    [species]
    order = ["Al", "Cu"]

    [pair.AlAl]
    initial = [0.20, 1.50, 2.75]
    optimize = false

    [pair.AlCu]
    initial = [0.18, 1.45, 2.80]

    [pair.CuCu]
    initial = [0.22, 1.60, 2.90]

Field guide
-----------

Here is the short version of what each part means:

- ``family`` says which model family you are fitting.
  - ``analytical``: one formula per term
  - ``eam``: tabulated embedded-atom model
  - ``adp``: tabulated EAM plus angular corrections
- ``sw``: Stillinger-Weber potential
  - ``tersoff``: Tersoff potential
- ``form`` says which flavor inside the family you want.
  - for analytical pair potentials, this is the equation name
  - for EAM, this is usually ``alloy`` or ``fs``
  - for ADP, the current runtime layout follows the alloy-style density form
  - for SW, the standard unary parameter layout is used
- ``species`` fixes the order of elements in the tables.
- ``grids`` holds the x-axes for tabulated functions.
- term blocks hold the actual numbers or formulas.
- ``initial`` is the starting guess for the fit.

The runtime engine is set in the training setting file, for example
``examples/toml/eam/alloy/forgeff.train.toml`` uses ``[common].engine =
"numba"``.

If the file format is ambiguous, use an explicit format hint, just like ASE.

For analytical pair potentials, the potential file only stores the formula or
parameter guess:

- ``ASE`` for direct ASE calculators when the built-in form has an ASE
  counterpart
- ``numpy`` for the Python/sympy-backed path
- ``numba`` for the JIT pair engine when the form is supported

For tabulated EAM and ADP potentials, the runtime engine is set separately in
the training setting file. EAM supports ``ASE``, **ForgeFF NumPy**, and
**ForgeFF Numba**. ADP supports **ForgeFF NumPy** and **ForgeFF Numba**.

Stillinger-Weber
-----------------

ForgeFF also supports a native Stillinger-Weber potential family with a
potfit-style multispecies layout. The potential file defines the species order
and then provides one pair block per unique species pair plus one lambda block
per center/neighbor triple:

.. code-block:: toml

    [potential]
    family = "sw"
    costheta0 = 0.3333333333333333

    [species]
    order = ["Al", "Cu"]

    [pair.AlAl]
    initial = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 7.0]

    [pair.AlCu]
    initial = [1.1, 2.1, 3.1, 0.0, 4.1, 5.1, 6.1, 7.1]

    [pair.CuCu]
    initial = [1.2, 2.2, 3.2, 0.0, 4.2, 5.2, 6.2, 7.2]

    [lambda.AlAlAl]
    initial = [0.1]

    [lambda.AlAlCu]
    initial = [0.2]

    [lambda.AlCuCu]
    initial = [0.3]

    [lambda.CuAlAl]
    initial = [0.4]

    [lambda.CuAlCu]
    initial = [0.5]

    [lambda.CuCuCu]
    initial = [0.6]

The matching ``forgeff.train.toml`` then sets ``[common].engine`` to either
``numpy`` or ``numba``. The public rule is the same as the other native
families: ``numpy`` for the reference implementation and ``numba`` for the
accelerated path.

Tersoff
-------

ForgeFF also supports a native Tersoff family with a species-order table and
explicit triple blocks. Each block stores the 14 parameters for one
``(i, j, k)`` species combination:

.. code-block:: toml

    [potential]
    family = "tersoff"
    cutoff_skin = 0.3

    [species]
    order = ["Si"]

    [triplet.SiSiSi]
    initial = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]

The runtime engine still lives in the training setting file and uses
``engine = "numba"`` for now.

Path handling
-------------

Relative paths in training, evaluation, and grading setting files are resolved
relative to the setting file itself. That makes the examples easier to move
around, because they still know where their data lives.

Analytical pair potentials
--------------------------

ForgeFF supports two analytical pair modes:

- built-in named forms from the registry
- user-defined expressions

The full equation list and parameter order for the built-in forms lives in
:doc:`analytical`. That page is the place to check when you want to
know exactly what a name like ``double_morse`` means.

Example:

.. code-block:: toml

    [potential]
    family = "analytical"
    form = "double_morse"
    cutoff = 8.0
    initial = [0.03, 5.0, 2.75, 0.01, 2.0, 3.5, 0.0]

In this example, ``double_morse`` tells ForgeFF which formula to use, and
``initial`` gives the first values for that formula in the same order listed on
:doc:`analytical`.

The runtime engine is set in the matching ``forgeff.train.toml`` file.

Custom expressions
~~~~~~~~~~~~~~~~~~

If the equation is not in the registry, define it explicitly. That keeps the
file self-contained and avoids creating a new registry entry for a one-off
form:

.. code-block:: toml

    [potential]
    family = "analytical"
    expression = "A*(exp(-2*a*(r-r0)) - 2*exp(-a*(r-r0))) + B/r**12"
    parameter_names = ["A", "a", "r0", "B"]
    cutoff = 8.0

    [parameters]
    initial = [0.04, 4.5, 2.85, 1e-4]

Here the formula is written directly in the file, so the model is fully
described without needing a built-in name.

EAM theory and TOML mapping
---------------------------

The EAM family uses pair interactions plus embedding energy:

.. math::

    E = \frac{1}{2} \sum_{i \ne j} \phi_{\alpha_i \alpha_j}(r_{ij})
        + \sum_i F_{\alpha_i}(\rho_i)

The local density is typically:

.. math::

    \rho_i = \sum_{j \ne i} \rho_{\alpha_i \alpha_j}(r_{ij})

ForgeFF currently supports two EAM density conventions, and the TOML file
states which one you are using:

- ``form = "alloy"``
- ``form = "fs"``

Alloy EAM
~~~~~~~~~

Alloy EAM is the simpler and more common case. The density contribution from a
neighbor depends on the neighbor species only:

.. math::

    \rho_i = \sum_{j \ne i} \rho_{\alpha_j}(r_{ij})

In TOML this is the simplest alloy layout:

.. code-block:: toml

    [potential]
    family = "eam"
    form = "alloy"

    [species]
    order = ["Al"]

    [grids]
    r = [0.1, 0.2, 0.3]
    rho = [0.0, 1.0, 2.0]

    [pair.AlAl]
    values = [0.1, 0.2, 0.3]

    [density.Al]
    values = [0.4, 0.5, 0.6]

    [embedding.Al]
    values = [0.7, 0.8, 0.9]

Read this as follows:

- ``pair.AlAl`` is the pair interaction for Al with Al.
- ``density.Al`` is the density contribution carried by Al neighbors.
- ``embedding.Al`` is the embedding curve for Al.

A matching worked example is listed in :doc:`examples/toml` under the EAM
section.

If you want the same alloy layout for a binary system, see the binary Alloy
example in :doc:`examples/toml`.

Finnis-Sinclair EAM
~~~~~~~~~~~~~~~~~~~

Finnis-Sinclair is a little more general. The density contribution can depend
on both the central species and the neighbor species:

.. math::

    \rho_i = \sum_{j \ne i} \rho_{\alpha_i \alpha_j}(r_{ij})

In TOML this means the density table is fully species-pair dependent.

A matching worked example is listed in :doc:`examples/toml` under the EAM
section.

If you want the same Finnis-Sinclair layout for a unary system, see the unary
FS example in :doc:`examples/toml`.

If you want to record the intended runtime engine, add it to the matching
training setting file under ``[common].engine``.

What the data object stores
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``phi_values`` holds the pair tables.
- ``rho_values`` holds the density tables. For alloy EAM this is one density
  curve per species; for Finnis-Sinclair it is a full species-pair matrix.
- ``emb_values`` holds the embedding tables.

Analytical multispecies layouts
-------------------------------

ForgeFF also supports analytical expressions in place of sampled tables for
the EAM terms. The layout stays explicit, but the values are generated from a
formula instead of loaded from a dense grid.

For a multispecies pairwise setup, define one block per species pair and keep
the shared form at the top level. The explicit ``[species].order`` list tells
ForgeFF how to map the pair names to atom types.

The current pairwise TOML layout supports this pattern for built-in analytical
forms such as ``lj`` and ``morse``. Direct ``ASE`` fitting remains a single
global calculator and is not used for per-pair multispecies fits.
If ``engine = "ASE"`` is requested for an analytical form that ASE does not
implement, ForgeFF warns and falls back to the ForgeFF-native ``numpy`` path.
Multispecies pairwise fits with ``engine = "ASE"`` are rejected with a warning
because that path is not implemented.

This is the point where ForgeFF follows the potfit-style idea of explicit pair
channels, but we are still keeping the per-pair form question open for later
decision. The current TOML layout supports one shared pair form plus explicit
per-pair parameter blocks.

For EAM alloy, the analytical version looks like this:

.. code-block:: toml

    [potential]
    family = "eam"
    form = "alloy"

    [species]
    order = ["Al", "Cu"]

    [grids]
    r = [0.1, 0.2, 0.3, 0.4]
    rho = [0.0, 1.0, 2.0, 3.0]

    [pair.AlAl]
    initial = [0.20, 1.50, 2.75]

    [pair.AlCu]
    initial = [0.18, 1.40, 2.85]

    [pair.CuCu]
    initial = [0.22, 1.60, 2.65]

    [density.Al]
    expression = "A * exp(-beta * r)"
    parameter_names = ["A", "beta"]
    initial = [1.0, 2.0]

    [density.Cu]
    expression = "A * exp(-beta * r)"
    parameter_names = ["A", "beta"]
    initial = [0.9, 1.8]

    [embedding.Al]
    expression = "F0 * sqrt(rho)"
    parameter_names = ["F0"]
    initial = [0.15]

    [embedding.Cu]
    expression = "F0 * sqrt(rho)"
    parameter_names = ["F0"]
    initial = [0.18]

Two-step EAM fitting is also supported. A common pattern is to fit the pair
term first, then freeze that pair block and fit the density and embedding
terms in a second pass:

.. code-block:: toml

    # Stage 1: fit pair terms, keep density/embedding fixed.
    [potential]
    family = "eam"
    form = "alloy"

    [species]
    order = ["Al"]

    [grids]
    r = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    rho = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]

    [pair.AlAl]
    values = [0.00, -0.05, -0.02, -0.01, -0.005, 0.00]

    [density.Al]
    values = [1.20, 0.80, 0.50, 0.30, 0.15, 0.05]
    optimize = false

    [embedding.Al]
    values = [0.00, -0.20, -0.35, -0.45, -0.55, -0.65]
    optimize = false

.. code-block:: toml

    # Stage 2: freeze the pair block and fit density/embedding.
    [potential]
    family = "eam"
    form = "alloy"

    [species]
    order = ["Al"]

    [grids]
    r = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    rho = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]

    [pair.AlAl]
    values = [0.00, -0.05, -0.02, -0.01, -0.005, 0.00]
    optimize = false

    [density.Al]
    values = [1.20, 0.80, 0.50, 0.30, 0.15, 0.05]

    [embedding.Al]
    values = [0.00, -0.20, -0.35, -0.45, -0.55, -0.65]

The same pattern works for multispecies alloy and Finnis-Sinclair fits: keep
the pretrained ``pair.*`` blocks fixed and optimize only the new density or
embedding blocks in the next stage.

For EAM Finnis-Sinclair, the pair blocks stay the same, but the density table
becomes fully species-pair dependent:

.. code-block:: toml

    [potential]
    family = "eam"
    form = "fs"

    [species]
    order = ["Al", "Cu"]

    [grids]
    r = [0.1, 0.2, 0.3, 0.4]
    rho = [0.0, 1.0, 2.0, 3.0]

    [pair.AlAl]
    initial = [0.20, 1.50, 2.75]

    [pair.AlCu]
    initial = [0.18, 1.40, 2.85]

    [pair.CuCu]
    initial = [0.22, 1.60, 2.65]

    [density.AlAl]
    expression = "A * exp(-beta * r)"
    parameter_names = ["A", "beta"]
    initial = [1.0, 2.0]

    [density.AlCu]
    expression = "A * exp(-beta * r)"
    parameter_names = ["A", "beta"]
    initial = [0.8, 1.7]

    [density.CuAl]
    expression = "A * exp(-beta * r)"
    parameter_names = ["A", "beta"]
    initial = [0.8, 1.7]

    [density.CuCu]
    expression = "A * exp(-beta * r)"
    parameter_names = ["A", "beta"]
    initial = [0.9, 1.8]

    [embedding.Al]
    expression = "F0 * sqrt(rho)"
    parameter_names = ["F0"]
    initial = [0.15]

    [embedding.Cu]
    expression = "F0 * sqrt(rho)"
    parameter_names = ["F0"]
    initial = [0.18]

These analytical EAM layouts follow the same ``[common].engine`` rule as the
tabulated examples: keep the runtime engine in the matching training setting
file, not in the potential file.

ADP theory and TOML mapping
---------------------------

ADP extends EAM with angular terms, usually represented by dipole and
quadrupole contributions. In practice, it keeps the EAM backbone and adds
directional structure on top of it.

In ForgeFF the tabulated ADP schema keeps the EAM blocks and adds:

- ``[dipole.*]``
- ``[quadrupole.*]``

This is the TOML layout:

.. code-block:: toml

    [potential]
    family = "adp"
    form = "alloy"

    [species]
    order = ["Al", "Cu"]

    [grids]
    r = [0.1, 0.2, 0.3]
    rho = [0.0, 1.0, 2.0]

    [pair.AlCu]
    values = [0.1, 0.2, 0.3]

    [density.Al]
    values = [0.4, 0.5, 0.6]

    [embedding.Al]
    values = [0.7, 0.8, 0.9]

    [dipole.AlCu]
    values = [0.01, 0.02, 0.03]

    [quadrupole.AlCu]
    values = [0.02, 0.03, 0.04]

The extra ``dipole`` and ``quadrupole`` blocks are what make ADP more
directional than plain EAM.

Initial guesses
---------------

- analytical forms use ``initial`` or ``[parameters].initial``
- tabulated forms use the term ``values`` arrays themselves
- user-defined expressions use the ordered parameter names plus an initial vector

If you are fitting from scratch, think of ``initial`` as the first guess the
optimizer will see.

Embedding terms
---------------

Embedding terms use the same analytical registry as pairwise terms. In
practice, ``sqrt`` is the familiar embedding form, but any registered
analytical form or user-defined expression can be used if it makes sense as a
function of ``rho``.

Example:

.. code-block:: toml

    [embedding.Al]
    form = "sqrt"
    initial = [0.1, 1.0]

    [embedding.Cu]
    expression = "A * exp(-B * rho) + C"
    parameter_names = ["A", "B", "C"]
    initial = [0.2, 1.5, 0.0]

Practical rules
---------------

- Use uniform grids for tabulated EAM/ADP.
- Keep the species order explicit so the file is readable later.
- Use built-in analytical forms when you want a stable parameter order.
- Use user-defined expressions when you need a one-off equation.
- Use ``[common].engine = "numba"`` in the training setting when the supported
  form has a JIT path and you want more speed.

See also
--------

- :doc:`io`
- :doc:`examples/toml`
- :doc:`theory`
