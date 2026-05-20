Theory
======

ForgeFF supports semi-empirical interatomic potentials in the EAM and ADP
families, plus analytical pair forms. The current TOML workflow is designed to
make the potential family explicit rather than hiding it behind file names.

EAM basics
----------

The Embedded Atom Method writes the energy as a sum of pair interactions and
an embedding term:

.. math::

    E = \frac{1}{2} \sum_{i \ne j} \phi_{\alpha_i \alpha_j}(r_{ij})
        + \sum_i F_{\alpha_i}(\rho_i)

where:

- :math:`\phi_{\alpha_i \alpha_j}(r)` is the pair term between species
  :math:`\alpha_i` and :math:`\alpha_j`
- :math:`F_{\alpha_i}(\rho)` is the embedding energy for species
  :math:`\alpha_i`
- :math:`\rho_i` is the local electron density at atom :math:`i`

The local density is typically written as:

.. math::

    \rho_i = \sum_{j \ne i} \rho_{\alpha_i \alpha_j}(r_{ij})

In ForgeFF, the tabulated EAM data object stores these parts explicitly:

- ``phi_values`` for pair functions
- ``rho_values`` for density functions
- ``emb_values`` for embedding functions

Alloy EAM
---------

The alloy form assumes the density contribution from a neighbor depends on the
neighbor species only:

.. math::

    \rho_i = \sum_{j \ne i} \rho_{\alpha_j}(r_{ij})

This is the simplest and most common binary/alloy EAM setup. In the current
TOML schema, this corresponds to using ``form = "alloy"`` and storing the
density table per species.

Finnis-Sinclair EAM
-------------------

The Finnis-Sinclair, or ``fs``, form makes the density contribution depend on
both species:

.. math::

    \rho_i = \sum_{j \ne i} \rho_{\alpha_i \alpha_j}(r_{ij})

This is more general than alloy EAM and is useful when the density contribution
depends on both the central and neighboring species.

In ForgeFF this is represented by:

- ``form = "fs"`` in the potential schema
- a full ``rho_values[species_i, species_j, r]`` table

What the code stores
--------------------

The current data model mirrors this distinction:

- :class:`forgeff.potentials.eam.data.EAMData` stores ``phi_values``,
  ``rho_values``, and ``emb_values``
- ``form = "alloy"`` uses the simplified density interpretation
- ``form = "fs"`` uses the full species-pair density interpretation

ADP basics
----------

The Angular-Dependent Potential extends EAM with angular corrections, usually
via dipole and quadrupole terms. Conceptually, ADP adds more structure to the
local environment than scalar density alone.

One common ADP decomposition writes the total energy as:

.. math::

    E_{\mathrm{ADP}} = E_{\mathrm{EAM}} + E_{\mathrm{dip}} + E_{\mathrm{quad}}

with the EAM backbone

.. math::

    E_{\mathrm{EAM}} = \frac{1}{2} \sum_{i \ne j} \phi_{\alpha_i \alpha_j}(r_{ij})
        + \sum_i F_{\alpha_i}(\rho_i)

and angular moments built from directional neighbor sums. A common choice is:

.. math::

    \mathbf{\mu}_i = \sum_{j \ne i} \mathbf{\hat r}_{ij}\, g_{\alpha_i \alpha_j}(r_{ij})

.. math::

    \mathbf{\Lambda}_i = \sum_{j \ne i}
    \left(\mathbf{\hat r}_{ij} \otimes \mathbf{\hat r}_{ij}
    - \frac{1}{3}\mathbf{I}\right)
    h_{\alpha_i \alpha_j}(r_{ij})

where :math:`\mathbf{\hat r}_{ij}` is the unit vector from atom :math:`i` to
:math:`j`, :math:`g` is the dipole radial weight, and :math:`h` is the
quadrupole radial weight. The angular energy then takes the schematic form

.. math::

    E_{\mathrm{dip}} = \frac{1}{2} \sum_i \mathbf{\mu}_i \cdot \mathbf{\mu}_i

.. math::

    E_{\mathrm{quad}} = \frac{1}{2} \sum_i \mathbf{\Lambda}_i : \mathbf{\Lambda}_i

This is the level of structure ForgeFF keeps in the ADP tables: pair terms,
scalar density, embedding, and the extra dipole and quadrupole channels.

In ForgeFF the tabulated ADP model keeps the EAM terms and adds:

- ``dipole_values``
- ``quadrupole_values``

The ADP data object therefore contains:

- pair functions
- density functions
- embedding functions
- dipole functions
- quadrupole functions

Embedding functions in ForgeFF may use the built-in analytical registry or a
user-defined expression. The common choice is ``sqrt``, but the schema does not
restrict you to that one form if another analytical shape is more appropriate
for :math:`F(\rho)`.

Relationship between families
-----------------------------

The practical hierarchy is:

- pair potentials: only :math:`\phi(r)`
- EAM: :math:`\phi(r) + F(\rho)`
- alloy EAM: EAM with species-dependent density functions
- FS EAM: more general EAM with pair-dependent density functions
- ADP: EAM plus angular corrections

Stillinger-Weber
-----------------

Stillinger-Weber is a three-body potential with a pair term and a triplet
term. In its common form, the energy is written as:

.. math::

    E = \sum_{i < j} \phi_{\alpha_i \alpha_j}(r_{ij})
        + \sum_{i < j < k} \lambda_{\alpha_i \alpha_j \alpha_k}
          \, g(r_{ij})\, g(r_{ik})\, h(\theta_{jik})

The pair part controls the two-body attraction and repulsion, while the
three-body term penalizes bond-angle distortions. In ForgeFF’s current SW
layout:

- each unique species pair has its own pair parameter block
- each ordered species triple has its own lambda block
- the runtime engine is native ForgeFF ``numpy`` or ``numba``

This matches the potfit-style multispecies organization used by the parser and
the example templates.

Extrapolation grading
---------------------

ForgeFF uses a MaxVol-based active-set grade to ask a simple question:
can the new configuration be rebuilt from the training basis, or is it already
outside what the fit has seen?

The practical idea is:

1. turn the training structures into a matrix
2. pick the most informative rows as the active set
3. try to express the new configuration in terms of that active set
4. look at the largest coefficient

If that largest coefficient is around 1, the structure is close to the
training set. If it is much larger than 1, the structure is drifting toward
extrapolation.

Where the Jacobian comes from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The grade is built from a Jacobian matrix. Here the word *Jacobian* simply
means “how sensitive the energy is to each parameter.”

For one configuration, ForgeFF asks the calculator for:

.. math::

    \frac{\partial E}{\partial p_1},\;
    \frac{\partial E}{\partial p_2},\;
    \ldots,\;
    \frac{\partial E}{\partial p_P}

where :math:`p_1, \ldots, p_P` are the fitted parameters. The calculator
already knows these derivatives because the engine implements
``jac_energy()``.

So for one configuration, the Jacobian row is simply:

.. math::

    j = \left[
    \frac{\partial E}{\partial p_1},
    \frac{\partial E}{\partial p_2},
    \ldots,
    \frac{\partial E}{\partial p_P}
    \right]

If you have :math:`N` training configurations, those rows are stacked into a
matrix:

.. math::

    J \in \mathbb{R}^{N \times P}

MaxVol then picks a square active-set submatrix from those training rows, and a
new configuration is compared against it.

For a new configuration, ForgeFF solves a least-squares problem to find the
coefficients that best rebuild its Jacobian row from the active set. The grade
is then the largest absolute coefficient.

.. math::

    \gamma = \max_k |c_k|

Beginner example with a tiny Morse toy model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Imagine a toy Morse fit with only three parameters:

- ``D_e``
- ``a``
- ``r_e``

For one configuration, the engine might report the sensitivity vector

.. math::

    j = \begin{bmatrix}
    \frac{\partial E}{\partial D_e} &
    \frac{\partial E}{\partial a} &
    \frac{\partial E}{\partial r_e}
    \end{bmatrix}

That means:

- if ``D_e`` changes a little, the energy changes by the first number
- if ``a`` changes a little, the energy changes by the second number
- if ``r_e`` changes a little, the energy changes by the third number

Now suppose MaxVol selected these training rows:

.. math::

    J_{\mathrm{act}} =
    \begin{bmatrix}
    1.0 & 0.2 & 0.0 \\
    0.1 & 1.0 & 0.3 \\
    0.0 & 0.2 & 1.0
    \end{bmatrix}

Then the least-squares reconstruction of :math:`j` from the active set gives
some coefficients :math:`c = [c_1, c_2, c_3]`.

For example, if the solver returns

.. math::

    c = \begin{bmatrix} 0.7 & 1.4 & 0.2 \end{bmatrix}

So the grade is:

.. math::

    \gamma = \max(0.7, 1.4, 0.2) = 1.4

That tells you the configuration is already outside the safe interpolation
region in at least one direction.

Interpretation:

- :math:`\gamma \lesssim 1`: the configuration is well represented by the
  training active set.
- :math:`\gamma > 1`: the configuration is increasingly extrapolative.
- larger values mean the new structure is asking the fitted potential to
  explain geometry beyond what the training set spans.

In configuration mode, ForgeFF computes one grade per structure. In
neighborhood mode, it computes grades per local environment and then keeps the
maximum value for the structure.

How this maps to TOML
---------------------

- ``family = "eam"`` with ``form = "alloy"`` gives the alloy-EAM style layout
- ``family = "eam"`` with ``form = "fs"`` gives the full FS-style layout
- ``family = "adp"`` uses the same base layout as EAM and adds dipole and
  quadrupole terms

For worked examples, see the TOML reference pages under ``examples/toml/`` and
the format discussion in :doc:`io` and :doc:`credit`.
