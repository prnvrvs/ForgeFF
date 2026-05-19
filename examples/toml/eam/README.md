# EAM Examples

This folder shows four tabulated EAM layouts, each in its own subfolder:

- `alloy/`: a simple alloy-style Al example
- `alloy_binary/`: the same alloy-style layout on a binary Al-Cu example
- `fs/`: a Finnis-Sinclair example for Al-Cu
- `fs_unary/`: the same Finnis-Sinclair layout on a unary Al example

Each subfolder contains:

- `initial.toml`
- `forgeff.train.toml`

The files are intentionally small so the schema is easy to read:

- `alloy/initial.toml` uses one species and one table per EAM term
- `alloy_binary/initial.toml` shows the same alloy terms with two species
- `fs/initial.toml` uses two species and species-pair density tables
- `fs_unary/initial.toml` shows the FS term layout on one species

The relative paths inside each `forgeff.train.toml` file are resolved against
the setting file itself. The shared training inputs live in
`../data/unary/training.cfg` for the Al-only examples and
`../data/binary/training.cfg` for the binary examples.
