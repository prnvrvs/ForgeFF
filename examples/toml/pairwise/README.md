# Pairwise Examples

This folder keeps one pairwise example family per subfolder, and each family
has a unary and binary variant:

- `morse/unary/`
- `morse/binary/`
- `morse/binary_frozen/`
- `double_morse/unary/`
- `double_morse/binary/`
- `custom_expression/unary/`
- `custom_expression/binary/`

Each leaf folder contains:

- `initial.toml`
- `forgeff.train.toml`

The unary variants use a single `[pair.AlAl]` block. The binary variants add
`[pair.AlCu]` and `[pair.CuCu]` so each combination can be fitted
independently.

The `morse/binary_frozen/` example shows the same binary layout with a frozen
pretrained `[pair.AlAl]` block.

The relative paths inside each `forgeff.train.toml` file are resolved against
the setting file itself. From each leaf folder, the shared training inputs
resolve to `../../../data/unary/training.cfg` and
`../../../data/binary/training.cfg`.
