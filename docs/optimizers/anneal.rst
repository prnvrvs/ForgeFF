Simulated annealing optimizer
=============================

The simulated annealing optimizer is available as a potfit-style global search
method for potential parameters. It works on the flat parameter vector
exposed by the potential data object, proposes random moves, accepts downhill
moves, and can occasionally accept uphill moves early in the annealing
schedule.

Usage
-----

For CLI (``forgeff.train.toml`` for ``forgeff train``):

.. code-block:: toml

    [[steps]]
    method = "SA"
    optimized = ["parameters"]

    [steps.kwargs]
    anneal_temp = "auto"
    maxiter = 100
    steps_per_temperature = 20

For Python API:

.. code-block:: python

    from forgeff.train import Trainer

    steps = [
        {
            "method": "SA",
            "optimized": ["parameters"],
            "kwargs": {
                "anneal_temp": "auto",
                "maxiter": 100,
                "steps_per_temperature": 20,
            },
        }
    ]
    Trainer(..., steps=steps)

Supported keyword arguments
---------------------------

- ``anneal_temp``: starting temperature or ``"auto"``
- ``temperature``: alias for ``anneal_temp``
- ``seed``: random seed
- ``maxiter``: number of temperature levels
- ``steps_per_temperature``: number of sweep cycles per temperature
- ``cooling``: temperature multiplier after each level
- ``step_scale``: initial proposal size for each parameter

Like the other optimizers, simulated annealing works on the current potential
parameter vector and writes back the improved parameters at the end of the run.

Example
-------

See the standalone walkthrough in :doc:`/examples/python/3.anneal`.

See also
--------

- :doc:`scipy`
- :doc:`ga`
