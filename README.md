# [ForgeFF: semi-empirical potential fitting in Python](https://github.com/prnvrvs/ForgeFF)

[![PyPI version](https://badge.fury.io/py/forgeff.svg)](https://badge.fury.io/py/forgeff)
[![Downloads](https://static.pepy.tech/badge/forgeff/month)](https://pepy.tech/project/forgeff)
[![GitHubActions](https://github.com/prnvrvs/ForgeFF/actions/workflows/tests.yml/badge.svg)](https://github.com/prnvrvs/ForgeFF/actions?query=workflow%3ATests)

ForgeFF is a Python toolkit for fitting semi-empirical interatomic potentials.
It keeps the model equations, parameter order, and fitting workflow explicit so
the fundamentals stay easy to inspect and explain.

The recommended user-facing format is TOML:
- custom analytic pair potentials
- built-in analytical forms such as Morse and double-Morse
- tabulated EAM and ADP term blocks
- beginner-friendly examples for training, evaluation, grading, and calculators
