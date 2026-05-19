# ADP Example

This example shows a tabulated ADP potential for Al-Cu in its own subfolder.

It includes:

- pair tables
- density tables
- embedding tables
- dipole tables
- quadrupole tables

The example files are:

- `alcu/initial.toml`
- `alcu/forgeff.train.toml`

The values are small and illustrative. Replace them with sampled tables or
your own initial guess before running a real fit.

The relative paths inside `forgeff.train.toml` are resolved against the
setting file itself. The shared training input lives in
`../data/binary/training.cfg`.
