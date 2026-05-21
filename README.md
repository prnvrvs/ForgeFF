# [ForgeFF: semi-empirical potential fitting in Python](https://github.com/prnvrvs/ForgeFF)

[![Latest release](https://img.shields.io/github/v/release/prnvrvs/ForgeFF?display_name=tag)](https://github.com/prnvrvs/ForgeFF/releases/latest)
[![Release downloads](https://img.shields.io/github/downloads/prnvrvs/ForgeFF/total)](https://github.com/prnvrvs/ForgeFF/releases)
[![GitHubActions](https://github.com/prnvrvs/ForgeFF/actions/workflows/tests.yml/badge.svg)](https://github.com/prnvrvs/ForgeFF/actions?query=workflow%3ATests)

ForgeFF is a Python toolkit for fitting semi-empirical interatomic potentials.
It keeps the model equations, parameter order, and fitting workflow explicit so
the fundamentals stay easy to inspect and explain.

Install from PyPI with:

```bash
pip install ForgeFF
```

The recommended user-facing format is TOML:
- custom analytic pair potentials
- built-in analytical forms such as Morse and double-Morse
- tabulated EAM and ADP term blocks
- beginner-friendly examples for training, evaluation, grading, and calculators
