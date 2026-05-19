# Al-Cu ADP Example

This folder contains a tabulated ADP potential for Al-Cu:

- `initial.toml`
- `forgeff.train.toml`

The `initial.toml` file extends the EAM layout with dipole and quadrupole
tables. The training setting points to `../../data/binary/training.cfg`, so
you can replace that path with your own configuration list before running a
real fit.
