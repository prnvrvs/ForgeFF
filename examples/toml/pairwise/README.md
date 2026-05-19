# Pairwise Example

This folder now keeps one pairwise example per subfolder:

- `morse/`: the simple built-in Morse pair form
- `double_morse/`: the two-well built-in pair form
- `custom_expression/`: an explicit user equation

Each subfolder contains:

- `initial.toml`
- `forgeff.train.toml`

The relative paths inside each `forgeff.train.toml` file are resolved against
the setting file itself. The shared training inputs live in
`../data/unary/training.cfg` and `../data/binary/training.cfg`.
