.. _toml-spec:

TOML Specification
==================

ForgeFF uses TOML as the main user-facing format for potential definitions and
training settings. The goal is to keep the file readable for humans while
still making the potential family, term order, backend, and initial guess
explicit for the code.

This page is the authoritative reference for the ForgeFF TOML schema. For
worked examples, see:

- :doc:`examples/toml`
- :doc:`example`

The example set includes unary and binary layouts for the supported families:

- pairwise: unary and binary
- EAM: unary alloy, binary alloy, unary FS, and binary FS
- ADP: unary and binary

Overview
--------

Think of the TOML file as the conversation between you and ForgeFF. It tells
the code what kind of model you want, which equations or tables to use, and
what values should seed the fit.

The TOML schema supports three potential families:

- analytical pair potentials
- tabulated EAM potentials
- tabulated ADP potentials

The general structure is:

- ``[potential]``: selects the family and calculation mode
- ``[species]``: fixes the canonical species order for EAM/ADP
- ``[grids]``: defines uniform radial and density grids for tabulated models
- term blocks such as ``[pair.*]``, ``[density.*]``, ``[embedding.*]``,
  ``[dipole.*]``, and ``[quadrupole.*]``
- ``[parameters]``: optional container for analytical starting values

Field guide
-----------

Here is the short version of what each part means:

- ``family`` says which model family you are fitting.
  - ``analytical``: one formula per term
  - ``eam``: tabulated embedded-atom model
  - ``adp``: tabulated EAM plus angular corrections
- ``form`` says which flavor inside the family you want.
  - for analytical pair potentials, this is the equation name
  - for EAM and ADP, this is usually ``alloy`` or ``fs``
- ``backend`` says how the code should evaluate the potential when that makes
  sense.
  - analytical pair forms: ``numpy`` or ``numba``
  - EAM: ``numpy`` or ``numba``
  - ADP: ``numba``
- ``species`` fixes the order of elements in the tables.
- ``grids`` holds the x-axes for tabulated functions.
- term blocks hold the actual numbers or formulas.
- ``initial`` is the starting guess for the fit.

For analytical pair potentials, ``backend`` selects the evaluator:

- ``numpy`` for the Python/sympy-backed path
- ``numba`` for the JIT pair backend when the form is supported

For EAM and ADP, ``backend`` is optional metadata. When present it is recorded
with the potential and can be set to ``numpy`` or ``numba``.

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
    backend = "numpy"
    cutoff = 8.0
    initial = [0.03, 5.0, 2.75, 0.01, 2.0, 3.5, 0.0]

In this example, ``double_morse`` tells ForgeFF which formula to use, and
``initial`` gives the first values for that formula in the same order listed on
:doc:`analytical`.

The ``backend`` can be set to ``"numba"`` when you want the faster JIT pair
backend for supported built-in forms.

Custom expressions
~~~~~~~~~~~~~~~~~~

If the equation is not in the registry, define it explicitly. That keeps the
file self-contained and avoids creating a new registry entry for a one-off
form:

.. code-block:: toml

    [potential]
    family = "analytical"
    backend = "numpy"
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
    backend = "numpy"

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

If you want to record the intended runtime backend, you can add a ``backend``
field to the ``[potential]`` block as metadata, for example ``numba``.

What the data object stores
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``phi_values`` holds the pair tables.
- ``rho_values`` holds the density tables.
- ``emb_values`` holds the embedding tables.

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
    backend = "numba"

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
- Use ``backend = "numba"`` when the supported form has a JIT path and you
  want more speed.

See also
--------

- :doc:`io`
- :doc:`examples/toml`
- :doc:`theory`
