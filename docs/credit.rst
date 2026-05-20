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

What ForgeFF adds
-----------------

- TOML-based configuration for potentials and training settings.
- Built-in analytical pair forms with explicit parameter order.
- Custom analytical pair expressions.
- ForgeFF-native NumPy and Numba EAM/ADP paths.
- Native Stillinger-Weber NumPy and Numba engines.
- ASE-backed calculator support for wrapped calculators.
- A more direct mapping from the file format to the fitted terms.

Practical summary
-----------------

If you want a potfit-like workflow in Python, the path is simple:

- use TOML to define the potential family and the initial guess
- use tabulated EAM/ADP for alloy fitting
- use built-in analytical forms or user-defined expressions for pair terms
- use the NumPy engine first, then the Numba engine where available for speed

If you are looking for the original potfit project, ForgeFF is best described
as a Python package that follows that style and credits that design lineage.
If you are looking for the codebase ForgeFF grew from, that is MOTEP.
