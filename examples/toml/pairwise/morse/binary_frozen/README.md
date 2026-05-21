# Morse Binary Frozen Example

This folder contains a binary Morse example with a frozen pretrained
`[pair.AlAl]` block.

- `initial.toml`
- `forgeff.train.toml`

The training setting uses the binary dataset under `../../../data/binary/`.
The `pair.AlAl` block stays fixed, while `pair.AlCu` and `pair.CuCu` remain
trainable.
