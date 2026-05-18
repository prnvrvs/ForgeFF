# Pairwise Example

This example shows two ways to define a pair potential:

- `built_in.toml`: a built-in analytical form from the registry
- `custom_expression.toml`: an explicit user equation

The training settings in `forgeff.train.toml` point to the built-in form.
To use the explicit expression, change `potentials.initial` accordingly.

The relative paths inside `forgeff.train.toml` are resolved against the
setting file itself. The shared training input lives in `../data/training.cfg`.
