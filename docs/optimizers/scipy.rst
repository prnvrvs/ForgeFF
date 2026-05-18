SciPy optimizers
================

.. |scipy-minimize| replace:: ``scipy.optimize.minimize``
.. _scipy-minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

.. |scipy-DA| replace:: ``scipy.optimize.dual_annealing``
.. _scipy-DA: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html

.. |scipy-DE| replace:: ``scipy.optimize.differential_evolution``
.. _scipy-DE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

The gradient-based local optimizers and the global optimizers in SciPy are
available to determine potential parameters.

Local optimizers
----------------

We can give the method name for |scipy-minimize|_.

For CLI (``forgeff.train.toml`` for ``forgeff train``):

.. code-block:: toml

    [[steps]]
    method = "BFGS"
    optimized = ["parameters"]

For Python API:

.. code-block:: python

    from forgeff.train import Trainer

    method = "BFGS"
    optimized = ["parameters"]
    Trainer(..., steps=[{"method": method, "optimized": optimized}])

Methods like ``BFGS`` and ``Nelder-Mead`` can be specified.
Optimizers with constraints such as ``L-BFGS-B`` are also available,
but since the fixed parameters are handled on the ForgeFF side,
they are not particularly recommended.

Global optimizers
-----------------

SciPy global optimizers can be specified.

- |scipy-DA|_: ``DA``
- |scipy-DE|_: ``DE``

Supported keyword arguments
---------------------------

The SciPy methods accept the keywords that ``scipy.optimize`` expects. The
most common ones are:

- ``method``: local solver name for ``minimize``
- ``jac``: analytical Jacobian toggle for local optimizers
- ``bounds``: optional parameter bounds

See also
--------

- :doc:`anneal`
- :doc:`ga`
