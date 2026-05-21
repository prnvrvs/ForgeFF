``forgeff template``
====================

The ``forgeff template`` command writes a starter TOML file for one of the
supported potential families. It is meant as a quick way to bootstrap a new
``initial.toml`` for the examples and your own fits.

Usage
-----

.. code-block:: bash

   forgeff template analytical --species Al Cu --form morse
   forgeff template eam --species Al Cu --form alloy
   forgeff template sw --species Si
   forgeff template tersoff --species Si C

The command prints the template to stdout unless ``--output`` is given:

.. code-block:: bash

   forgeff template sw --species Al Cu --output initial.toml

The generated file is only the potential definition. The matching
``forgeff.train.toml`` still lives in the training settings workflow and keeps
``[common].engine`` separate from the potential description.
