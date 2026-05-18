# EAM Example

This example shows a tabulated EAM potential for Al.

The file is intentionally small so the schema is easy to read:

- one species
- one pair table
- one density table
- one embedding table

Use this as a template for larger alloy tables by expanding the `species`
list and adding the canonical term blocks.

The relative paths inside `forgeff.train.toml` are resolved against the
setting file itself. The shared training input lives in `../data/training.cfg`.
