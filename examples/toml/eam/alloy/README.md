# Alloy EAM Example

This folder contains a simple alloy-style tabulated EAM potential:

- `initial.toml`
- `forgeff.train.toml`

The `initial.toml` file uses one species and one table per EAM term. The
training setting points to `../../data/unary/training.cfg`, so you can replace
that path with your own configuration list before running a real fit.
