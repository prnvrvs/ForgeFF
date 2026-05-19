"""TOML potential IO."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any

import numpy as np
import tomllib
from ase.data import atomic_numbers

from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.ase.forms import evaluate_expression, evaluate_form, get_form_spec
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_label(value: Any) -> str:
    return "".join(ch for ch in str(value) if ch.isalnum()).lower()


def _species_number(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    label = str(value)
    if label not in atomic_numbers:
        raise ValueError(f"Unknown species label in TOML potential: {value!r}")
    return int(atomic_numbers[label])


def _species_labels(species: list[int]) -> list[str]:
    from ase.data import chemical_symbols

    return [chemical_symbols[int(num)] for num in species]


def _resolve_index(name: Any, labels: list[str]) -> int:
    if isinstance(name, (int, np.integer)):
        idx = int(name)
        if idx < 0 or idx >= len(labels):
            raise ValueError(f"Species index out of range: {idx}")
        return idx

    target = _normalize_label(name)
    for idx, label in enumerate(labels):
        if _normalize_label(label) == target:
            return idx
    raise ValueError(f"Unknown species label in term: {name!r}")


def _term_array(term: dict[str, Any], expected_size: int, *, key: str = "values") -> np.ndarray:
    values = term.get(key, term.get("initial"))
    if values is None:
        raise ValueError(f"Missing '{key}' array for term {term!r}")
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != expected_size:
        raise ValueError(f"Expected {expected_size} values, got {arr.size}")
    return arr


def _term_parameters(term: dict[str, Any], default_names: list[str]) -> tuple[list[str], np.ndarray]:
    parameter_names = _as_list(term.get("parameter_names", default_names))
    if not parameter_names:
        values = term.get("parameters", term.get("initial"))
        if values is None:
            return [], np.array([], dtype=float)
        values = np.asarray(_as_list(values), dtype=float).reshape(-1)
        if values.size != 0:
            raise ValueError(
                "Analytical TOML terms with no parameter names must not define values."
            )
        return [], values
    values = term.get("parameters", term.get("initial"))
    if values is None:
        values = [0.0] * len(parameter_names)
    values = np.asarray(_as_list(values), dtype=float).reshape(-1)
    if values.size != len(parameter_names):
        raise ValueError(
            f"Length of analytical parameters ({values.size}) does not match "
            f"'parameter_names' ({len(parameter_names)})."
        )
    return parameter_names, values


def _analytic_term_array(term: dict[str, Any], grid: np.ndarray) -> np.ndarray:
    if "expression" in term:
        parameter_names, values = _term_parameters(term, [])
        variable = str(term.get("variable", "r"))
        return evaluate_expression(
            str(term["expression"]),
            variable=variable,
            parameter_names=parameter_names,
            x=grid,
            parameters=values,
        )

    form = term.get("form")
    if form is None:
        raise ValueError("Analytical term requires either 'values' or 'form'/'expression'.")
    spec = get_form_spec(str(form))
    parameter_names, values = _term_parameters(term, spec["params"])
    variable = str(term.get("variable", spec.get("variable", "r")))
    return evaluate_form(
        str(form),
        x=grid,
        parameters=values,
        variable=variable,
    )


def _pair_from_key(name: str, labels: list[str]) -> tuple[int, int]:
    normalized = _normalize_label(name)
    matches: list[tuple[int, int]] = []
    for i, left in enumerate(labels):
        left_norm = _normalize_label(left)
        for j, right in enumerate(labels):
            right_norm = _normalize_label(right)
            if normalized == left_norm + right_norm:
                matches.append((i, j))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous pair term name {name!r}; please use explicit 'species' entries.")

    # Fallback for separator-delimited names such as "Al-Ni" or "Al_Ni".
    tokens = [token for token in name.replace("-", " ").replace("_", " ").split() if token]
    if len(tokens) == 2:
        return _resolve_index(tokens[0], labels), _resolve_index(tokens[1], labels)

    raise ValueError(
        f"Could not infer pair species from term name {name!r}; "
        "please add an explicit 'species = [...]' entry."
    )


def _one_species_from_term(name: str, labels: list[str]) -> int:
    target = _normalize_label(name)
    for idx, label in enumerate(labels):
        if _normalize_label(label) == target:
            return idx
    tokens = [token for token in name.replace("-", " ").replace("_", " ").split() if token]
    if len(tokens) == 1:
        return _resolve_index(tokens[0], labels)
    raise ValueError(
        f"Could not infer species from term name {name!r}; "
        "please add an explicit 'species = ...' entry."
    )


def _parse_species(data: dict[str, Any], potential: dict[str, Any]) -> list[int]:
    species_block = data.get("species", {})
    species = species_block.get("order", potential.get("species", []))
    species = _as_list(species)
    if not species:
        raise ValueError("TOML potential requires a non-empty [species].order list.")
    return [_species_number(item) for item in species]


def _grid_from(data: dict[str, Any], potential: dict[str, Any], name: str) -> np.ndarray:
    grids = data.get("grids", {})
    value = grids.get(name, potential.get(f"{name}_grid", potential.get(name)))
    if value is None:
        raise ValueError(f"TOML potential is missing the '{name}' grid.")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Grid '{name}' must be a one-dimensional array.")
    return arr


def _term_key_pairs(species_count: int, form: str) -> list[tuple[int, int]]:
    if form == "alloy":
        return [(i, i) for i in range(species_count)]
    return [(i, j) for i in range(species_count) for j in range(species_count)]


def _check_coverage(
    seen: set[tuple[int, int]],
    expected: list[tuple[int, int]],
    *,
    label: str,
) -> None:
    missing = [pair for pair in expected if pair not in seen]
    if missing:
        raise ValueError(f"TOML potential is missing required {label} terms for species pairs: {missing}")


def _read_custom_toml(data: dict[str, Any], potential: dict[str, Any]) -> ASEData:
    engine = str(potential.get("engine", "numpy"))
    engine_alias = engine.lower()
    calculator_kwargs = {}
    if "expression" in potential:
        if engine_alias == "numba":
            warnings.warn(
                "Custom analytical expressions do not support engine='numba'; falling back to engine='numpy'.",
                RuntimeWarning,
                stacklevel=2,
            )
            engine = "numpy"
            engine_alias = "numpy"
        calculator_kwargs["expression"] = potential["expression"]
    elif "form" in potential:
        spec = get_form_spec(str(potential["form"]))
        calculator_kwargs["form"] = str(potential["form"])
        calculator_kwargs.setdefault("parameter_names", spec["params"])
        if engine_alias == "numpy":
            calculator_kwargs["expression"] = spec["formula"]
            calculator_kwargs.setdefault("variable", spec.get("variable", "r"))
        elif engine_alias != "numba":
            raise ValueError(f"Unknown analytical engine {engine!r}. Use 'numpy' or 'numba'.")
    for key in ("parameter_names", "variable", "cutoff", "rc"):
        if key in potential:
            calculator_kwargs[key] = potential[key]
    if "calculator_kwargs" in potential and isinstance(potential["calculator_kwargs"], dict):
        calculator_kwargs.update(potential["calculator_kwargs"])

    parameter_names = _as_list(calculator_kwargs.get("parameter_names", []))
    if not parameter_names:
        raise ValueError("Custom TOML potentials require 'parameter_names'.")
    calculator_kwargs["parameter_names"] = parameter_names

    initial = data.get("parameters", {}).get("initial", potential.get("initial"))
    initial = _as_list(initial)
    if initial and len(initial) != len(parameter_names):
        raise ValueError("Length of 'initial' must match 'parameter_names'.")

    ase_data = ASEData(
        engine=str(engine),
        calculator_kwargs=calculator_kwargs,
    )
    for idx, name in enumerate(parameter_names):
        value = initial[idx] if idx < len(initial) else None
        ase_data.add_parameter(name, (), value)
    return ase_data


def _populate_eam_arrays(
    data: dict[str, Any],
    potential: dict[str, Any],
    *,
    family: str,
) -> EAMData | ADPData:
    species = _parse_species(data, potential)
    labels = _species_labels(species)
    r_grid = _grid_from(data, potential, "r")
    rho_grid = _grid_from(data, potential, "rho")
    form = str(potential.get("form", "alloy")).lower()
    if form not in {"alloy", "fs"}:
        raise ValueError("EAM/ADP TOML 'form' must be either 'alloy' or 'fs'.")

    spc = len(species)
    nr = len(r_grid)
    nrho = len(rho_grid)

    phi = np.zeros((spc, spc, nr), dtype=float)
    rho = np.zeros((spc, spc, nr), dtype=float)
    emb = np.zeros((spc, nrho), dtype=float)

    pair_terms = data.get("pair", {})
    density_terms = data.get("density", {})
    embedding_terms = data.get("embedding", {})
    dipole_terms = data.get("dipole", {})
    quadrupole_terms = data.get("quadrupole", {})

    seen_pairs: set[tuple[int, int]] = set()
    for name, term in pair_terms.items():
        term = dict(term)
        if "species" in term:
            if len(_as_list(term["species"])) != 2:
                raise ValueError(f"Pair term {name!r} must define exactly two species.")
            i, j = (_resolve_index(s, labels) for s in _as_list(term["species"]))
        else:
            i, j = _pair_from_key(name, labels)
        values = _term_array(term, nr) if "values" in term else _analytic_term_array(term, r_grid)
        phi[i, j] = values
        phi[j, i] = values
        seen_pairs.add((i, j))
        seen_pairs.add((j, i))

    _check_coverage(seen_pairs, _term_key_pairs(spc, form), label="pair")

    seen_density: set[tuple[int, int]] = set()
    if form == "alloy":
        for name, term in density_terms.items():
            term = dict(term)
            if "species" in term:
                species_idx = _resolve_index(_as_list(term["species"])[0], labels)
            else:
                species_idx = _one_species_from_term(name, labels)
            values = _term_array(term, nr) if "values" in term else _analytic_term_array(term, r_grid)
            rho[:, species_idx, :] = values
            seen_density.add((species_idx, species_idx))
    else:
        for name, term in density_terms.items():
            term = dict(term)
            if "species" in term:
                pair = _as_list(term["species"])
                if len(pair) != 2:
                    raise ValueError(f"FS density term {name!r} must define exactly two species.")
                i, j = (_resolve_index(s, labels) for s in pair)
            else:
                i, j = _pair_from_key(name, labels)
            values = _term_array(term, nr) if "values" in term else _analytic_term_array(term, r_grid)
            rho[i, j] = values
            seen_density.add((i, j))

    _check_coverage(seen_density, _term_key_pairs(spc, form), label="density")

    seen_embedding: set[int] = set()
    for name, term in embedding_terms.items():
        term = dict(term)
        if "species" in term:
            species_idx = _resolve_index(_as_list(term["species"])[0], labels)
        else:
            species_idx = _one_species_from_term(name, labels)
        values = _term_array(term, nrho) if "values" in term else _analytic_term_array(term, rho_grid)
        emb[species_idx] = values
        seen_embedding.add(species_idx)

    missing_embedding = [idx for idx in range(spc) if idx not in seen_embedding]
    if missing_embedding:
        raise ValueError(f"TOML potential is missing required embedding terms for species indices: {missing_embedding}")

    if family == "eam":
        pot = EAMData(
            potential_name=str(potential.get("potential_name", "")),
            form=form,
            r_grid=r_grid,
            rho_grid=rho_grid,
            phi_values=phi,
            rho_values=rho,
            emb_values=emb,
        )
    else:
        dipole = np.zeros((spc, spc, nr), dtype=float)
        quadrupole = np.zeros((spc, spc, nr), dtype=float)
        seen_dipole: set[tuple[int, int]] = set()
        for name, term in dipole_terms.items():
            term = dict(term)
            if "species" in term:
                pair = _as_list(term["species"])
                if len(pair) != 2:
                    raise ValueError(f"Dipole term {name!r} must define exactly two species.")
                i, j = (_resolve_index(s, labels) for s in pair)
            else:
                i, j = _pair_from_key(name, labels)
            values = _term_array(term, nr) if "values" in term else _analytic_term_array(term, r_grid)
            dipole[i, j] = values
            dipole[j, i] = values
            seen_dipole.add((i, j))
            seen_dipole.add((j, i))
        _check_coverage(seen_dipole, _term_key_pairs(spc, form), label="dipole")

        seen_quadrupole: set[tuple[int, int]] = set()
        for name, term in quadrupole_terms.items():
            term = dict(term)
            if "species" in term:
                pair = _as_list(term["species"])
                if len(pair) != 2:
                    raise ValueError(f"Quadrupole term {name!r} must define exactly two species.")
                i, j = (_resolve_index(s, labels) for s in pair)
            else:
                i, j = _pair_from_key(name, labels)
            values = _term_array(term, nr) if "values" in term else _analytic_term_array(term, r_grid)
            quadrupole[i, j] = values
            quadrupole[j, i] = values
            seen_quadrupole.add((i, j))
            seen_quadrupole.add((j, i))
        _check_coverage(seen_quadrupole, _term_key_pairs(spc, form), label="quadrupole")
        pot = ADPData(
            potential_name=str(potential.get("potential_name", "")),
            form=form,
            r_grid=r_grid,
            rho_grid=rho_grid,
            phi_values=phi,
            rho_values=rho,
            emb_values=emb,
            dipole_values=dipole,
            quadrupole_values=quadrupole,
        )

    pot.species = np.asarray(species, dtype=np.int32)
    return pot


def read_potential_toml(filename: str | Path):
    """Read a TOML-defined potential."""
    with Path(filename).open("rb") as handle:
        data = tomllib.load(handle)

    potential = data.get("potential", {})
    family = str(potential.get("family", "")).lower()
    form = str(potential.get("form", "alloy")).lower()
    default_engine = "numba" if family == "adp" or (family == "eam" and form == "fs") else "numpy"
    engine = str(potential.get("engine", default_engine))
    engine_alias = engine.lower()

    if family in {"eam", "adp"}:
        result = _populate_eam_arrays(data, potential, family=family)
        result.engine = engine
        return result

    if family == "analytical" or engine_alias in {"numpy", "numba"}:
        return _read_custom_toml(data, potential)

    # Infer family from the presence of ADP-only tables or EAM tables.
    if any(key in data for key in ("dipole", "quadrupole")):
        return _populate_eam_arrays(data, potential, family="adp")
    if any(key in data for key in ("pair", "density", "embedding")):
        return _populate_eam_arrays(data, potential, family="eam")

    raise ValueError(
        "Could not infer TOML potential family. "
        "Set [potential].family to 'analytical', 'eam', or 'adp'."
    )
