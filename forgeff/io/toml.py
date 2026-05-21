"""TOML potential IO."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any

import numpy as np
import tomllib
from ase.data import atomic_numbers, chemical_symbols

from forgeff.potentials.ase.data import ASEData
from forgeff.potentials.ase.forms import evaluate_expression, evaluate_form, get_form_spec
from forgeff.potentials.eam.adp_data import ADPData
from forgeff.potentials.eam.data import EAMData
from forgeff.potentials.tersoff.data import TersoffData, TersoffParameters
from forgeff.potentials.sw.data import PAIR_PARAMETER_COUNT, SWData


_ASE_ANALYTICAL_CALCULATORS = {
    "lj": "LennardJones",
    "morse": "MorsePotential",
}

_PAIR_SHARED_FORMS = {
    "lj",
    "bornmayer",
    "morse",
    "doublemorse",
    "powerdecay",
    "expdecay",
    "constant",
    "coul",
    "exponential",
    "hbnd",
    "buck",
    "eopp",
    "csw",
    "csw2",
    "ms",
    "born",
    "softshell",
    "expplus",
    "mexpdecay",
    "strmm",
    "poly5",
    "zero",
}


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


def _parse_sw_species(data: dict[str, Any], potential: dict[str, Any]) -> list[str]:
    species_block = data.get("species", {})
    species = species_block.get("order", potential.get("species", ["Si"]))
    species = _as_list(species)
    if not species:
        species = ["Si"]
    labels: list[str] = []
    for item in species:
        if isinstance(item, (int, np.integer)):
            labels.append(str(chemical_symbols[int(item)]))
        else:
            labels.append(str(item))
    return labels


def _triple_from_key(name: str, labels: list[str]) -> tuple[int, int, int]:
    normalized = _normalize_label(name)
    matches: list[tuple[int, int, int]] = []
    for i, left in enumerate(labels):
        left_norm = _normalize_label(left)
        for j, mid in enumerate(labels):
            mid_norm = _normalize_label(mid)
            for k, right in enumerate(labels):
                right_norm = _normalize_label(right)
                if normalized == left_norm + mid_norm + right_norm:
                    matches.append((i, j, k))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous triple term name {name!r}; please use explicit 'species' entries.")

    tokens = [token for token in name.replace("-", " ").replace("_", " ").split() if token]
    if len(tokens) == 3:
        return (
            _resolve_index(tokens[0], labels),
            _resolve_index(tokens[1], labels),
            _resolve_index(tokens[2], labels),
        )

    raise ValueError(
        f"Could not infer triple species from term name {name!r}; "
        "please add an explicit 'species = [...]' entry."
    )


def _parse_tersoff_species(data: dict[str, Any], potential: dict[str, Any]) -> list[str]:
    species_block = data.get("species", {})
    species = species_block.get("order", potential.get("species", []))
    species = _as_list(species)
    if not species:
        raise ValueError("Tersoff TOML potential requires a non-empty [species].order list.")
    labels: list[str] = []
    for item in species:
        if isinstance(item, (int, np.integer)):
            labels.append(str(chemical_symbols[int(item)]))
        else:
            labels.append(str(item))
    return labels


def _tersoff_triplet_coverage(species_count: int) -> list[tuple[int, int, int]]:
    return [(i, j, k) for i in range(species_count) for j in range(species_count) for k in range(species_count)]


def _sw_pair_coverage(species_count: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(species_count) for j in range(i, species_count)]


def _sw_lambda_coverage(species_count: int) -> list[tuple[int, int, int]]:
    return [
        (i, j, k)
        for i in range(species_count)
        for j in range(species_count)
        for k in range(j, species_count)
    ]


def _grid_from(data: dict[str, Any], potential: dict[str, Any], name: str) -> np.ndarray:
    grids = data.get("grids", {})
    value = grids.get(name, potential.get(f"{name}_grid", potential.get(name)))
    if value is None:
        raise ValueError(f"TOML potential is missing the '{name}' grid.")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Grid '{name}' must be a one-dimensional array.")
    return arr


def _pair_key_pairs(species_count: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(species_count) for j in range(i, species_count)]


def _density_key_pairs(species_count: int, form: str) -> list[tuple[int, int]]:
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


def _term_optimize_flag(term: dict[str, Any]) -> bool:
    if "optimize" not in term:
        return True
    return bool(term["optimize"])


def _read_custom_toml(data: dict[str, Any], potential: dict[str, Any]) -> ASEData:
    if "pair" in data and data["pair"]:
        return _read_multispecies_pair_toml(data, potential)

    engine = str(potential.get("engine", "numpy"))
    engine_alias = engine.lower()
    calculator_kwargs = {}
    if "expression" in potential:
        if engine_alias in {"numba", "ase"}:
            warnings.warn(
                "Custom analytical expressions do not support engine='numba' or engine='ASE'; falling back to engine='numpy'.",
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
        elif engine_alias == "ase":
            form_key = str(potential["form"]).lower()
            if form_key not in _ASE_ANALYTICAL_CALCULATORS:
                warnings.warn(
                    f"ASE does not support analytical form {potential['form']!r}; "
                    "falling back to engine='numpy'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                engine = "numpy"
                engine_alias = "numpy"
                calculator_kwargs["expression"] = spec["formula"]
                calculator_kwargs.setdefault("variable", spec.get("variable", "r"))
            else:
                calculator_kwargs["calculator"] = _ASE_ANALYTICAL_CALCULATORS[form_key]
        elif engine_alias != "numba":
            raise ValueError(f"Unknown analytical engine {engine!r}. Use 'ASE', 'numpy', or 'numba'.")
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


def _read_sw_toml(data: dict[str, Any], potential: dict[str, Any]) -> SWData:
    species = _parse_sw_species(data, potential)
    labels = species
    pair_terms = data.get("pair", {})
    lambda_terms = data.get("lambda", {})

    if pair_terms or lambda_terms:
        if not pair_terms or not lambda_terms:
            raise ValueError("Multispecies Stillinger-Weber requires both [pair.*] and [lambda.*] blocks.")

        spc = len(species)
        pair_parameters = np.zeros((spc, spc, PAIR_PARAMETER_COUNT), dtype=float)
        lambda_values = np.zeros((spc, spc, spc), dtype=float)
        seen_pairs: set[tuple[int, int]] = set()
        seen_lambdas: set[tuple[int, int, int]] = set()

        for name, term in pair_terms.items():
            term = dict(term)
            if "species" in term:
                pair = _as_list(term["species"])
                if len(pair) != 2:
                    raise ValueError(f"SW pair term {name!r} must define exactly two species.")
                i, j = (_resolve_index(s, labels) for s in pair)
            else:
                i, j = _pair_from_key(name, labels)
            values = _term_array(term, PAIR_PARAMETER_COUNT)
            pair_parameters[i, j] = values
            pair_parameters[j, i] = values
            seen_pairs.add((i, j))
            seen_pairs.add((j, i))

        for name, term in lambda_terms.items():
            term = dict(term)
            if "species" in term:
                triple = _as_list(term["species"])
                if len(triple) != 3:
                    raise ValueError(f"SW lambda term {name!r} must define exactly three species.")
                i, j, k = (_resolve_index(s, labels) for s in triple)
            else:
                i, j, k = _triple_from_key(name, labels)
            value = _term_array(term, 1)[0]
            j, k = sorted((j, k))
            lambda_values[i, j, k] = value
            lambda_values[i, k, j] = value
            seen_lambdas.add((i, j, k))
            seen_lambdas.add((i, k, j))

        missing_pairs = [pair for pair in _sw_pair_coverage(spc) if pair not in seen_pairs]
        if missing_pairs:
            raise ValueError(f"SW TOML is missing pair terms for species pairs: {missing_pairs}")
        missing_lambdas = [triplet for triplet in _sw_lambda_coverage(spc) if triplet not in seen_lambdas]
        if missing_lambdas:
            raise ValueError(f"SW TOML is missing lambda terms for species triples: {missing_lambdas}")

        return SWData(
            species=species,
            pair_parameters=pair_parameters,
            lambda_values=lambda_values,
            costheta0=float(potential.get("costheta0", 1.0 / 3.0)),
        )

    sw_kwargs: dict[str, Any] = {"species": species}
    for key in ("epsilon", "sigma", "costheta0", "A", "B", "p", "a", "lambda1", "gamma", "optimized"):
        if key in potential:
            sw_kwargs[key] = potential[key]
    return SWData(**sw_kwargs)


def _read_multispecies_pair_toml(data: dict[str, Any], potential: dict[str, Any]) -> ASEData:
    species = _parse_species(data, potential)
    labels = _species_labels(species)
    engine = str(potential.get("engine", "numpy"))
    engine_alias = engine.lower()
    if engine_alias == "ase":
        warnings.warn(
            "ASE does not support multispecies analytical pair fitting; stopping.",
            RuntimeWarning,
            stacklevel=2,
        )
        raise ValueError("engine='ASE' is not supported for multispecies analytical pair fitting.")

    pair_terms = data.get("pair", {})
    if not pair_terms:
        raise ValueError("Multispecies analytical pair TOML requires at least one [pair.*] block.")

    shared_form = potential.get("form")
    shared_expression = potential.get("expression")
    if shared_form is None and shared_expression is None:
        raise ValueError("Multispecies analytical pair TOML requires a top-level [potential].form or .expression.")

    if shared_form is not None:
        shared_form = str(shared_form)
        if _normalize_label(shared_form) not in _PAIR_SHARED_FORMS:
            raise ValueError(f"Unsupported analytical pair form for multispecies fitting: {shared_form!r}")

    shared_parameter_names = _as_list(potential.get("parameter_names", []))
    if shared_expression is not None and not shared_parameter_names:
        raise ValueError("Multispecies analytical expressions require [potential].parameter_names.")

    calculator_kwargs: dict[str, Any] = {
        "species": species,
        "pair_terms": [],
    }
    if shared_form is not None:
        calculator_kwargs["form"] = shared_form
    if shared_expression is not None:
        calculator_kwargs["expression"] = str(shared_expression)
        calculator_kwargs["parameter_names"] = shared_parameter_names
    for key in ("cutoff", "rc"):
        if key in potential:
            calculator_kwargs[key] = potential[key]

    ase_data = ASEData(engine=str(engine), calculator_kwargs=calculator_kwargs)

    seen_pairs: set[tuple[int, int]] = set()
    for name, term in pair_terms.items():
        term = dict(term)
        if "species" in term:
            pair = _as_list(term["species"])
            if len(pair) != 2:
                raise ValueError(f"Pair term {name!r} must define exactly two species.")
            i, j = (_resolve_index(s, labels) for s in pair)
        else:
            i, j = _pair_from_key(name, labels)

        term_form = term.get("form", shared_form)
        term_expression = term.get("expression", shared_expression)
        if term_form is None and term_expression is None:
            raise ValueError(f"Pair term {name!r} must define a form or expression.")
        if shared_form is not None and term_form is not None and str(term_form) != str(shared_form):
            raise ValueError(f"Pair term {name!r} must use the shared analytical form {shared_form!r}.")
        if shared_expression is not None and term_expression is not None and str(term_expression) != str(shared_expression):
            raise ValueError(f"Pair term {name!r} must use the shared analytical expression.")

        if term_expression is not None:
            parameter_names = _as_list(term.get("parameter_names", shared_parameter_names))
            if not parameter_names:
                raise ValueError(f"Multispecies pair term {name!r} requires 'parameter_names'.")
        else:
            spec = get_form_spec(str(shared_form))
            parameter_names = _as_list(term.get("parameter_names", spec["params"]))
        values = term.get("initial", term.get("values"))
        if values is None:
            values = [0.0] * len(parameter_names)
        values = _as_list(values)
        if len(values) != len(parameter_names):
            raise ValueError(
                f"Length of multispecies pair term {name!r} initial values must match parameter names."
            )

        prefix = f"{labels[i]}{labels[j]}"
        pair_entry = {
            "name": name,
            "species": [int(species[i]), int(species[j])],
            "prefix": prefix,
            "parameter_names": list(parameter_names),
            "form": str(term_form) if term_form is not None else None,
            "expression": str(term_expression) if term_expression is not None else None,
            "variable": str(term.get("variable", potential.get("variable", "r"))),
            "cutoff": term.get("cutoff", potential.get("cutoff")),
            "rc": term.get("rc", potential.get("rc")),
        }
        calculator_kwargs["pair_terms"].append(pair_entry)
        for idx, param_name in enumerate(parameter_names):
            ase_data.add_parameter(f"{prefix}_{param_name}", (), values[idx])
        seen_pairs.add((i, j))
        seen_pairs.add((j, i))

    _check_coverage(seen_pairs, _pair_key_pairs(len(species)), label="pair")
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
    optimized_blocks: list[str] = []
    block_mode = False

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
        if not _term_optimize_flag(term):
            block_mode = True
        if _term_optimize_flag(term):
            optimized_blocks.append(f"pair.{labels[min(i, j)]}{labels[max(i, j)]}")

    _check_coverage(seen_pairs, _pair_key_pairs(spc), label="pair")

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
            if not _term_optimize_flag(term):
                block_mode = True
            if _term_optimize_flag(term):
                optimized_blocks.append(f"density.{labels[species_idx]}")
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
            if not _term_optimize_flag(term):
                block_mode = True
            if _term_optimize_flag(term):
                optimized_blocks.append(f"density.{labels[i]}{labels[j]}")

    _check_coverage(seen_density, _density_key_pairs(spc, form), label="density")

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
        if not block_mode:
            block_mode = not _term_optimize_flag(term)
        if _term_optimize_flag(term):
            optimized_blocks.append(f"embedding.{labels[species_idx]}")

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
            if not _term_optimize_flag(term):
                block_mode = True
            if _term_optimize_flag(term):
                optimized_blocks.append(f"dipole.{labels[min(i, j)]}{labels[max(i, j)]}")
        _check_coverage(seen_dipole, _pair_key_pairs(spc), label="dipole")

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
            if not _term_optimize_flag(term):
                block_mode = True
            if _term_optimize_flag(term):
                optimized_blocks.append(f"quadrupole.{labels[min(i, j)]}{labels[max(i, j)]}")
        _check_coverage(seen_quadrupole, _pair_key_pairs(spc), label="quadrupole")
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
    if block_mode:
        pot.optimized = optimized_blocks
    return pot


def _read_tersoff_toml(data: dict[str, Any], potential: dict[str, Any]) -> TersoffData:
    species = _parse_tersoff_species(data, potential)
    labels = species
    triplet_terms = data.get("triplet", {})
    if not triplet_terms:
        raise ValueError("Tersoff TOML potential requires at least one [triplet.*] block.")

    parameters: dict[tuple[str, str, str], TersoffParameters] = {}
    seen_triplets: set[tuple[int, int, int]] = set()
    for name, term in triplet_terms.items():
        term = dict(term)
        if "species" in term:
            triple = _as_list(term["species"])
            if len(triple) != 3:
                raise ValueError(f"Tersoff triplet term {name!r} must define exactly three species.")
            key = tuple(str(s) for s in triple)
            i, j, k = (_resolve_index(s, labels) for s in triple)
        else:
            i, j, k = _triple_from_key(name, labels)
            key = (labels[i], labels[j], labels[k])
        values = _term_array(term, 14)
        parameters[key] = TersoffParameters.from_list(values)
        seen_triplets.add((i, j, k))

    missing_triplets = [triplet for triplet in _tersoff_triplet_coverage(len(species)) if triplet not in seen_triplets]
    if missing_triplets:
        raise ValueError(f"Tersoff TOML is missing triplet terms for species triples: {missing_triplets}")

    return TersoffData.from_parameter_dict(
        parameters,
        species=species,
        cutoff_skin=float(potential.get("cutoff_skin", potential.get("skin", 0.3))),
    )


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

    if family == "tersoff":
        return _read_tersoff_toml(data, potential)
    if family == "sw":
        return _read_sw_toml(data, potential)
    if family in {"eam", "adp"}:
        result = _populate_eam_arrays(data, potential, family=family)
        result.engine = engine
        return result

    if family == "analytical" or engine_alias in {"ase", "numpy", "numba"}:
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
