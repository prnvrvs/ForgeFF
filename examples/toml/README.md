# ForgeFF TOML Examples

This folder contains copyable TOML examples for the supported potential
families.

The examples are split into three groups:

- `pairwise/`: built-in and user-defined pair potentials
- `eam/`: tabulated EAM tables
- `adp/`: tabulated ADP tables
- `data/`: a tiny shared `training.cfg` smoke-test input

For a single-file overview of the TOML schema, see `potential.toml` in this
folder.

Each group contains:

- `initial.toml`: the potential definition passed to `forgeff train`
- `forgeff.train.toml`: a minimal training setting file

Relative paths in the setting files are resolved relative to the setting file
itself, so the examples can be run from any working directory.

The `training.cfg` entry in each `forgeff.train.toml` file is a placeholder.
Replace it with your own configuration list when running a real fit.
