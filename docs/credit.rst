Credit
======

ForgeFF started from the MOTEP codebase and keeps evolving into a broader
semi-empirical fitting tool with a cleaner Python/TOML interface. The goal is
not just to copy the old workflow, but to make it easier to read, easier to
edit, and easier to reuse in Python.

What we keep from potfit
------------------------

ForgeFF borrows the analytical pair-form idea and the workflow mindset from
potfit. potfit does implement pair potentials as a first-class family, with a
fixed layout per species pair and per potential family. That upstream work is
the reference point for the fitting style used here, especially for:

- explicit analytical pair forms
- readable configuration-driven fitting
- term-by-term potential construction
- tabulated EAM and ADP workflows

Links:

- potfit: https://www.potfit.net/wiki/
- MOTEP: https://github.com/imw-md/motep
- matscipy: https://github.com/libAtoms/matscipy

ForgeFF keeps the potfit ideas that matter for semi-empirical fitting:

- analytical pair forms with explicit parameter order
- pair potentials as a first-class family
- tabulated EAM and ADP fitting
- uniform radial and density grids for tabulated models
- a term-based view of the potential
- extrapolation-grade style analysis for active-set selection
- configuration-driven fitting instead of hard-coded internals
- Stillinger-Weber reference behavior cross-checked against matscipy for the unary limit and potfit for multispecies layout
- potfit-style I/O for converting external simulation outputs into reference
  configurations, including VASP/OUTCAR-to-force-file workflows

What ForgeFF adds
-----------------

- TOML-based configuration for potentials and training settings.
- Built-in analytical pair forms with explicit parameter order.
- Custom analytical pair expressions.
- ForgeFF-native NumPy and Numba EAM/ADP paths.
- Native Stillinger-Weber NumPy and Numba engines.
- ASE-backed calculator support for wrapped calculators.
- A more direct mapping from the file format to the fitted terms.
- I/O paths that keep the reference-config workflow explicit and scriptable,
  similar to potfit's OUTCAR conversion tools.

Current scope
-------------

Today ForgeFF is best described as a Python package for semi-empirical
potential fitting with a potfit-style workflow and a clearer TOML-first
interface. The docs now cover:

- the TOML example landing page and the real training launcher
- pairwise, EAM, ADP, and analytical example families
- the public engine choices and their dark-mode documentation styling
- CLI commands such as ``forgeff train``, ``forgeff evaluate``, ``forgeff grade``,
  ``forgeff error``, ``forgeff export``, and ``forgeff template``
- the maintainer and credit pages with current project links

ForgeFF follows that style and credits that design lineage. If you are looking
for the codebase ForgeFF grew from, that is MOTEP.
