# ForgeFF TOML Examples

This folder contains copyable TOML examples for the supported potential
families.

The examples are split into three groups:

- `pairwise/`: built-in and user-defined pair potentials, split into
  `unary/` and `binary/` leaves for each style
- `eam/`: tabulated EAM tables, including alloy and Finnis-Sinclair layouts,
  each in its own subfolder
- `adp/`: tabulated ADP tables in a dedicated subfolder
- `data/unary/`: a 10-frame unary Al training dataset from the EAM reference
- `data/binary/`: a 10-frame binary Al-Cu training dataset from the ADP reference

For a single-file overview of the TOML schema, see `potential.toml` in this
folder.

If you want to run a training example directly, use `train.py` in this
folder. It is the TOML-based training launcher, and you can point it at the
unary or binary example settings:

- unary: `python examples/toml/train.py --setting examples/toml/eam/alloy/forgeff.train.toml`
- binary: `python examples/toml/train.py --setting examples/toml/eam/alloy_binary/forgeff.train.toml`

The `examples/python` side is separate. It is the Python-based tutorial set
for training, evaluation, grading, and calculator usage. Its example folders
mirror the TOML layout under `examples/python/pairwise/`,
`examples/python/eam/`, and `examples/python/adp/`.

Each example subfolder contains:

- `initial.toml`: the potential definition passed to `forgeff train`
- `forgeff.train.toml`: a minimal training setting file

For a direct Python entry point, see `train.py` in this folder. It runs one
example training job from a setting file path.

Relative paths in the setting files are resolved relative to the setting file
itself, so the examples can be run from any working directory.

The shared datasets live under `data/unary/` and `data/binary/`. The unary
dataset is used for the Al-only examples, and the binary dataset is used for
the Al-Cu examples. The binary pairwise examples fit one parameter block per
species combination.
