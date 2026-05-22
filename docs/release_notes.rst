Release Notes
=============

This page summarizes the launch-facing changes currently in ForgeFF.

Latest updates
--------------

- LAMMPS EAM and ADP export now forwards header metadata into the body writer and resamples tabulated data with explicit endpoint slopes.
- The EAM and ADP small-cell self-image handling has been fixed so total energy and site-energy bookkeeping stay consistent with ASE.
- SW export and calculator edge cases for empty structures and similar robustness issues have been cleaned up.
- Loss, optimizer, and parser edge cases were fixed so mixed datasets, Jacobians, and configuration parsing behave predictably.
- The performance plots and benchmark styling have been refreshed for a cleaner presentation.

Export notes
------------

- EAM/alloy, EAM/fs, and ADP export use a uniform sampling grid before writing the LAMMPS table.
- Tersoff export follows the LAMMPS triangular parameter ordering for all supported species counts.

See also
--------

- :doc:`cli/export`
- :doc:`performance`
- :doc:`toml`
