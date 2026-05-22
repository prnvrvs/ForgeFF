``forgeff export``
==================

The ``forgeff export`` command converts a potential file into a
LAMMPS-compatible EAM, ADP, or Tersoff file.

It accepts:

- ``family = "eam"`` -> ``.eam.alloy`` or ``.fs``
- ``family = "adp"`` -> ``.adp``
- ``family = "tersoff"`` -> ``.tersoff``
- ForgeFF fitted ``.npy`` checkpoints for the same families

Usage
-----

.. code-block:: bash

   forgeff export initial.toml final.eam.alloy
   forgeff export initial.toml final.fs
   forgeff export initial.toml final.adp
   forgeff export initial.toml final.tersoff
   forgeff export final.npy final.eam.alloy
   forgeff export final.npy final.fs
   forgeff export final.npy final.adp
   forgeff export initial.toml final.eam.alloy --nr 3001 --nrho 2001

For tabulated EAM and ADP potentials, ``--nr`` and ``--nrho`` resample the
export onto a uniform grid before writing the LAMMPS table. ForgeFF samples
the source spline with explicit endpoint slopes before writing the table.
When omitted, the export uses the source grid length.

The export path is intended for tabulated EAM, ADP, and standard Tersoff
potentials. The output is written in LAMMPS-compatible format so it can be
used directly by LAMMPS, or read back through ForgeFF or ASE where supported.
