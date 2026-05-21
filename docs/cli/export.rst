``forgeff export``
==================

The ``forgeff export`` command converts a fitted potential file into a
LAMMPS-compatible EAM or ADP file.

It accepts the same potential inputs as the rest of ForgeFF:

- ``.toml`` potential descriptions
- fitted ``.npy`` checkpoints
- NIST-style EAM/ADP files

Usage
-----

.. code-block:: bash

   forgeff export final.npy final.eam.alloy
   forgeff export final.npy final.fs
   forgeff export final.npy final.adp

For ambiguous NIST inputs such as ``.txt`` files, pass an explicit format hint
just like ASE:

.. code-block:: bash

   forgeff export potential.txt final.adp --form adp

The export path is intended for tabulated EAM and ADP potentials. The output is
written with ASE's LAMMPS-compatible writer so it can be used directly by
LAMMPS or read back through ASE.
