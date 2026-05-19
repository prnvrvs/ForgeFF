# Custom Pairwise Expression Example

This folder contains a user-defined analytical pair potential:

- `initial.toml`
- `forgeff.train.toml`

The `initial.toml` file writes the equation directly instead of using a
built-in form name. The training setting points to
`../../data/unary/training.cfg`, so you can replace that path with your own
configuration list before running a real fit.
