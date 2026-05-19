Genetic algorithm optimizer
===========================

The genetic algorithm optimizer is a population-based global search method
for potential parameters. It evolves a set of candidate parameter vectors
through selection, crossover, mutation, and optional local refinement.

Usage
-----

For CLI (``forgeff.train.toml`` for ``forgeff train``):

.. code-block:: toml

    [[steps]]
    method = "GA"
    optimized = ["parameters"]

    [steps.kwargs]
    population_size = 40
    generations = 100

For Python API:

.. code-block:: python

    from forgeff.train import Trainer

    steps = [
        {
            "method": "GA",
            "optimized": ["parameters"],
            "kwargs": {
                "population_size": 40,
                "generations": 100,
            },
        }
    ]
    Trainer(..., steps=steps)

Supported keyword arguments
---------------------------

The GA implementation accepts the usual population settings:

- ``population_size``: number of candidates in the population
- ``mutation_rate``: probability of mutating one parameter
- ``elitism_rate``: fraction of the best candidates that survive
- ``crossover_probability``: probability of combining two parents
- ``superhuman``: whether to refine elites with a local optimizer

Example
-------

See the standalone walkthrough in :doc:`/examples/python/4.ga`.

See also
--------

- :doc:`anneal`
- :doc:`scipy`
