Template wizard
===============

The ``forgeff -t`` wizard writes a starter TOML file for one of the supported
potential families. It is meant as a quick way to bootstrap a new
``initial.toml`` and matching ``forgeff.train.toml`` for the examples and
your own fits.

Usage
-----

.. code-block:: bash

   forgeff -t

The wizard prompts for the dataset, species, family, and training options.
For fully scripted output, the internal template helpers still power the
documentation examples and tests, but the public entry point is the wizard.

The generated files are the potential definition and the matching training
setting. ``forgeff.train.toml`` keeps ``[common].engine`` separate from the
potential description.
