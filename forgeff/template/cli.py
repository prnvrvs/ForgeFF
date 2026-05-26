"""ForgeFF template wizard."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols
from ase.io import write as ase_write

from forgeff.io import read as read_images
from forgeff.potentials.ase.forms import get_form_spec


def _toml_list(values: list[str | float | int]) -> str:
    formatted: list[str] = []
    for value in values:
        if isinstance(value, str):
            formatted.append(f'"{value}"')
        elif isinstance(value, int):
            formatted.append(str(value))
        else:
            formatted.append(format(float(value), ".12g"))
    return "[" + ", ".join(formatted) + "]"


def _species_block(species: list[str]) -> str:
    return "[species]\norder = " + _toml_list(species) + "\n"


def _pair_labels(species: list[str]) -> list[tuple[str, str]]:
    return [(species[i], species[j]) for i in range(len(species)) for j in range(i, len(species))]


def _ordered_pairs(species: list[str]) -> list[tuple[str, str]]:
    return [(left, right) for left in species for right in species]


def _sw_lambda_labels(species: list[str]) -> list[tuple[str, str, str]]:
    return [
        (species[i], species[j], species[k])
        for i in range(len(species))
        for j in range(len(species))
        for k in range(j, len(species))
    ]


def _tersoff_triplet_labels(species: list[str]) -> list[tuple[str, str, str]]:
    return [
        (species[i], species[j], species[k])
        for i in range(len(species))
        for j in range(len(species))
        for k in range(len(species))
    ]


def _format_blocks(blocks: list[str]) -> str:
    return "\n".join(blocks).rstrip() + "\n"


def _toml_inline_table(mapping: dict[str, float]) -> str:
    items = ", ".join(f"{key} = {format(float(value), '.12g')}" for key, value in mapping.items())
    return "{ " + items + " }"


def _toml_lines_for_key_value(key: str, value: str | float | int | list[str] | list[str | float | int]) -> str:
    if isinstance(value, list):
        return f"{key} = {_toml_list(value)}"
    if isinstance(value, str):
        return f'{key} = "{value}"'
    if isinstance(value, int):
        return f"{key} = {value}"
    return f"{key} = {format(float(value), '.12g')}"


def _normalize_species_labels(images: list[object]) -> list[str]:
    seen: list[int] = []
    for atoms in images:
        numbers = np.asarray(getattr(atoms, "numbers", []), dtype=int).reshape(-1)
        for number in numbers.tolist():
            if number not in seen:
                seen.append(int(number))
    if not seen:
        raise ValueError("Could not infer species from the dataset.")
    return [chemical_symbols[int(number)] for number in seen]


def _species_need_manual_input(species: list[str]) -> bool:
    return any(label == "X" for label in species)


def _infer_species_from_dataset(dataset: str) -> list[str]:
    images = read_images(dataset)
    return _normalize_species_labels(images)


def _dataset_is_present(dataset: str | None) -> bool:
    return bool(dataset) and Path(dataset).exists()


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{text}{suffix}: ").strip()
    return default if not value and default is not None else value


def _prompt_float(text: str, default: float) -> float:
    while True:
        raw = _prompt(text, format(default, ".12g"))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def _prompt_choice(text: str, choices: list[str], default: str | None = None) -> str:
    allowed = {choice.lower(): choice for choice in choices}
    while True:
        raw = _prompt(text + f" ({', '.join(choices)})", default)
        choice = raw.lower()
        if choice in allowed:
            return allowed[choice]
        print(f"Please choose one of: {', '.join(choices)}")


def _prompt_optional_float(text: str, default: float | None = None) -> float | None:
    suffix = f" [{format(default, '.12g')}]" if default is not None else ""
    raw = input(f"{text}{suffix}: ").strip()
    if not raw:
        return default
    while True:
        try:
            return float(raw)
        except ValueError:
            raw = input("Please enter a valid number: ").strip()
            if not raw:
                return default


def _ensure_writable_path(path: str, force: bool) -> bool:
    if not Path(path).exists():
        return True
    if force:
        return True
    overwrite = _prompt_choice(f"Output file {path} exists. Overwrite?", ["no", "yes"], default="no")
    return overwrite == "yes"


def build_training_template(
    *,
    species: list[str],
    dataset: str,
    initial: str,
    final: str,
    engine: str,
    energy_weight: float,
    forces_weight: float,
    stress_weight: float,
    species_energy_offset_mode: str,
    species_energy_offsets: dict[str, float],
    optimizer: str,
    maxiter: int,
    tol: float,
) -> str:
    blocks = [
        "[common]",
        "seed = 42",
        f'engine = "{engine}"',
        _toml_lines_for_key_value("species", species),
        "",
        "[configurations]",
        _toml_lines_for_key_value("training", [dataset]),
        "",
        "[potentials]",
        _toml_lines_for_key_value("initial", initial),
        _toml_lines_for_key_value("final", final),
        "",
        "[loss]",
        _toml_lines_for_key_value("energy_weight", energy_weight),
        _toml_lines_for_key_value("forces_weight", forces_weight),
        _toml_lines_for_key_value("stress_weight", stress_weight),
        _toml_lines_for_key_value("species_energy_offset_mode", species_energy_offset_mode),
    ]
    if species_energy_offsets:
        blocks.append(f"species_energy_offsets = {_toml_inline_table(species_energy_offsets)}")
    blocks.extend(
        [
            "",
            "[[steps]]",
            'method = "minimize"',
            "",
            "[steps.kwargs]",
            f'method = "{optimizer}"',
            _toml_lines_for_key_value("tol", tol),
            "",
            "[steps.kwargs.options]",
            _toml_lines_for_key_value("maxiter", maxiter),
        ]
    )
    return _format_blocks(blocks)


def _analytical_template(species: list[str], form: str, expression: str | None, parameter_names: list[str] | None) -> str:
    blocks = [
        "[potential]",
        'family = "analytical"',
    ]
    if expression is not None:
        if not parameter_names:
            parameter_names = ["A", "a", "r0", "B"]
        blocks.append(f'expression = "{expression}"')
        if parameter_names:
            blocks.append(f"parameter_names = {_toml_list(parameter_names)}")
    else:
        blocks.append(f'form = "{form}"')
    blocks.append("cutoff = 8.0")
    blocks.append("")
    blocks.append(_species_block(species).rstrip())

    if expression is not None:
        nparams = len(parameter_names or [])
    else:
        nparams = len(get_form_spec(form)["params"])

    for left, right in _pair_labels(species):
        blocks.extend(
            [
                f"[pair.{left}{right}]",
                f"initial = {_toml_list([0.0] * nparams)}",
                "",
            ]
        )
    return _format_blocks(blocks)


def _eam_template(species: list[str], form: str) -> str:
    blocks = [
        "[potential]",
        'family = "eam"',
        f'form = "{form}"',
        "",
        _species_block(species).rstrip(),
        "",
        "[grids]",
        "r = [0.1, 0.2, 0.3]",
        "rho = [0.0, 1.0, 2.0]",
        "",
    ]

    for left, right in _pair_labels(species):
        blocks.extend(
            [
                f"[pair.{left}{right}]",
                "values = [0.0, 0.0, 0.0]",
                "",
            ]
        )

    if form == "alloy":
        for item in species:
            blocks.extend(
                [
                    f"[density.{item}]",
                    "values = [0.0, 0.0, 0.0]",
                    "",
                ]
            )
    else:
        for left in species:
            for right in species:
                blocks.extend(
                    [
                        f"[density.{left}{right}]",
                        "values = [0.0, 0.0, 0.0]",
                        "",
                    ]
                )

    for item in species:
        blocks.extend(
            [
                f"[embedding.{item}]",
                "values = [0.0, 0.0, 0.0]",
                "",
            ]
        )
    return _format_blocks(blocks)


def _adp_template(species: list[str]) -> str:
    blocks = [
        "[potential]",
        'family = "adp"',
        'form = "alloy"',
        "",
        _species_block(species).rstrip(),
        "",
        "[grids]",
        "r = [0.1, 0.2, 0.3]",
        "rho = [0.0, 1.0, 2.0]",
        "",
    ]
    for left, right in _pair_labels(species):
        blocks.extend(
            [
                f"[pair.{left}{right}]",
                "values = [0.0, 0.0, 0.0]",
                "",
            ]
        )
    for item in species:
        blocks.extend(
            [
                f"[density.{item}]",
                "values = [0.0, 0.0, 0.0]",
                "",
            ]
        )
    for item in species:
        blocks.extend(
            [
                f"[embedding.{item}]",
                "values = [0.0, 0.0, 0.0]",
                "",
            ]
        )
    for left, right in _pair_labels(species):
        blocks.extend(
            [
                f"[dipole.{left}{right}]",
                "values = [0.0, 0.0, 0.0]",
                "",
                f"[quadrupole.{left}{right}]",
                "values = [0.0, 0.0, 0.0]",
                "",
            ]
        )
    return _format_blocks(blocks)


def _sw_template(species: list[str]) -> str:
    blocks = [
        "[potential]",
        'family = "sw"',
        'costheta0 = 0.3333333333333333',
        "",
        _species_block(species).rstrip(),
        "",
    ]
    for left, right in _pair_labels(species):
        blocks.extend(
            [
                f"[pair.{left}{right}]",
                "initial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
                "",
            ]
        )
    for left, middle, right in _sw_lambda_labels(species):
        blocks.extend(
            [
                f"[lambda.{left}{middle}{right}]",
                "initial = [0.0]",
                "",
            ]
        )
    return _format_blocks(blocks)


def _tersoff_template(species: list[str]) -> str:
    blocks = [
        "[potential]",
        'family = "tersoff"',
        "cutoff_skin = 0.3",
        "",
        _species_block(species).rstrip(),
        "",
    ]
    for left, middle, right in _tersoff_triplet_labels(species):
        blocks.extend(
            [
                f"[triplet.{left}{middle}{right}]",
                "initial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
                "",
            ]
        )
    return _format_blocks(blocks)


def build_template(
    family: str,
    species: list[str],
    *,
    form: str | None = None,
    expression: str | None = None,
    parameter_names: list[str] | None = None,
) -> str:
    family = family.lower()
    if family == "analytical":
        return _analytical_template(species, form or "morse", expression, parameter_names)
    if family == "eam":
        return _eam_template(species, form or "alloy")
    if family == "adp":
        return _adp_template(species)
    if family == "sw":
        return _sw_template(species)
    if family == "tersoff":
        return _tersoff_template(species)
    raise ValueError(f"Unknown template family {family!r}")


def _dataset_has_energy(images: list[object]) -> bool:
    for atoms in images:
        calc = getattr(atoms, "calc", None)
        if calc is None:
            continue
        results = getattr(calc, "results", None) or {}
        targets = getattr(calc, "targets", None) or {}
        if "energy" in results or "energy" in targets:
            return True
    return False


def _wizard_species(dataset: str, species_override: list[str] | None) -> list[str]:
    if species_override:
        return list(species_override)
    return _infer_species_from_dataset(dataset)


def _prompt_species_labels() -> list[str]:
    while True:
        raw = _prompt("Species labels (space-separated)")
        labels = [label for label in raw.split() if label]
        if labels:
            return labels
        print("Please enter at least one species label.")


def _prompt_split_fractions() -> tuple[float, float]:
    while True:
        train_fraction = _prompt_float("Train fraction", 1.0)
        test_default = max(0.0, 1.0 - train_fraction)
        test_fraction = _prompt_optional_float(
            "Test fraction (leave blank for the remainder)",
            test_default,
        )
        if test_fraction is None:
            test_fraction = test_default
        if not (0.0 <= train_fraction <= 1.0 and 0.0 <= test_fraction <= 1.0):
            print("Fractions must be between 0 and 1.")
            continue
        if abs((train_fraction + test_fraction) - 1.0) > 1.0e-8:
            print("Train and test fractions must sum to 1.")
            continue
        return train_fraction, test_fraction


def _split_images(images: list[object], train_fraction: float) -> tuple[list[object], list[object]]:
    if not images:
        return [], []
    train_count = int(round(len(images) * train_fraction))
    train_count = max(0, min(len(images), train_count))
    return images[:train_count], images[train_count:]


def _prompt_family_defaults() -> tuple[str, str | None, str | None, list[str] | None]:
    family = _prompt_choice("Potential family", ["pair", "eam", "adp", "sw", "tersoff"], default="pair")
    form: str | None = None
    expression: str | None = None
    parameter_names: list[str] | None = None

    if family == "pair":
        pair_mode = _prompt_choice("Pair potential type", ["analytical", "custom"], default="analytical")
        family = "analytical"
        if pair_mode == "custom":
            expression = _prompt("Custom analytical expression")
            raw_names = _prompt("Parameter names (space-separated)", "").split()
            parameter_names = raw_names or None
        else:
            form = _prompt("Analytical form", "morse")
    elif family == "eam":
        form = _prompt_choice("EAM family", ["alloy", "fs"], default="alloy")
        embedding_mode = _prompt_choice("Embedding mode", ["tabulated", "analytical"], default="tabulated")
        if embedding_mode == "analytical":
            print("Analytical EAM embedding is not implemented yet; generating a tabulated template.")
    elif family == "adp":
        form = "alloy"
    return family, form, expression, parameter_names


def _wizard_generate(args: argparse.Namespace) -> tuple[str, str]:
    dataset = getattr(args, "dataset", None)
    if not dataset:
        dataset = _prompt("Dataset path (press Enter to skip, or type skip)", "skip")
    species: list[str] | None = None
    if dataset.lower() in {"", "skip", "none", "n", "no"}:
        dataset = ""
        images: list[object] = []
        print("Dataset step skipped.")
    else:
        if not Path(dataset).exists():
            print(f"Dataset {dataset!r} not found. Skipping dataset-specific prompts.")
            images = []
        else:
            images = read_images(dataset)
            species = _wizard_species(dataset, list(args.species) if args.species else None)
            if _species_need_manual_input(species):
                species = None
                print(
                    "Dataset does not expose chemical species labels; "
                    "they will be requested later."
                )
            else:
                print(f"Dataset contains {len(images)} structures and {len(species)} species: {', '.join(species)}")

    train_fraction = getattr(args, "train_fraction", None)
    test_fraction = getattr(args, "test_fraction", None)
    if images and train_fraction is None:
        train_fraction, test_fraction = _prompt_split_fractions()
    elif not images:
        train_fraction, test_fraction = 1.0, 0.0
    else:
        if test_fraction is None:
            test_fraction = max(0.0, 1.0 - train_fraction)
        if not (0.0 <= train_fraction <= 1.0 and 0.0 <= test_fraction <= 1.0):
            raise ValueError("Fractions must be between 0 and 1.")
        if abs((train_fraction + test_fraction) - 1.0) > 1.0e-8:
            raise ValueError("Train and test fractions must sum to 1.")

    train_images, test_images = _split_images(images, train_fraction)
    dataset_path = Path(dataset) if dataset else Path("training.xyz")
    train_dataset = dataset
    test_dataset = str(dataset_path.with_name(f"{dataset_path.stem}.test.xyz"))
    if dataset and (train_fraction < 1.0 or test_fraction > 0.0):
        train_dataset = str(dataset_path.with_name(f"{dataset_path.stem}.train.xyz"))
        if _ensure_writable_path(train_dataset, getattr(args, "force", False)):
            ase_write(train_dataset, train_images)
        if test_fraction > 0.0:
            if _ensure_writable_path(test_dataset, getattr(args, "force", False)):
                ase_write(test_dataset, test_images)
        print(f"Train split: {train_dataset}")
        if test_fraction > 0.0:
            print(f"Test split: {test_dataset}")
    elif not dataset:
        train_dataset = _prompt("Training dataset path", "training.xyz")

    family = getattr(args, "family", None)
    form = getattr(args, "form", None)
    expression = getattr(args, "expression", None)
    parameter_names = list(args.parameter_names) if getattr(args, "parameter_names", None) is not None else None
    if family is None:
        family, form, expression, parameter_names = _prompt_family_defaults()
    elif family == "pair":
        family = "analytical"
        if expression is None:
            form = form or "morse"

    if species is None:
        species = list(args.species) if getattr(args, "species", None) else _prompt_species_labels()

    if getattr(args, "energy_offset_mode", None) is not None:
        offset_mode = args.energy_offset_mode
    elif images and _dataset_has_energy(images):
        offset_mode = _prompt_choice(
            "Species energy correction mode",
            ["off", "manual", "regression"],
            default="regression",
        )
    else:
        offset_mode = "off"

    offsets: dict[str, float] = {}
    if offset_mode == "manual":
        for label in species:
            offsets[label] = _prompt_float(f"Energy offset for {label}", 0.0)

    energy_weight = args.energy_weight if args.energy_weight is not None else _prompt_float("Energy weight", 1.0)
    forces_weight = args.forces_weight if args.forces_weight is not None else _prompt_float("Forces weight", 0.01)
    stress_weight = args.stress_weight if args.stress_weight is not None else _prompt_float("Stress weight", 0.001)
    optimizer = args.optimizer or _prompt("Optimizer", "L-BFGS-B")
    maxiter = args.maxiter if args.maxiter is not None else int(_prompt("Max iterations", "100"))
    tol = args.tol if args.tol is not None else _prompt_float("Optimizer tolerance", 1.0e-6)

    family = family.lower()
    final_name = "final.toml" if family == "analytical" else "final.npy"
    initial_output = args.initial_output or "initial.toml"
    train_output = args.train_output or args.output or "forgeff.train.toml"
    if not species:
        species = ["Al"]

    initial_text = build_template(
        family,
        species,
        form=form,
        expression=expression,
        parameter_names=parameter_names,
    )
    train_text = build_training_template(
        species=species,
        dataset=train_dataset,
        initial=initial_output,
        final=final_name,
        engine="numpy",
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        species_energy_offset_mode=offset_mode,
        species_energy_offsets=offsets,
        optimizer=optimizer,
        maxiter=maxiter,
        tol=tol,
    )
    if _ensure_writable_path(initial_output, getattr(args, "force", False)):
        Path(initial_output).write_text(initial_text, encoding="utf-8")
    if _ensure_writable_path(train_output, getattr(args, "force", False)):
        Path(train_output).write_text(train_text, encoding="utf-8")
    return initial_text, train_text


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument(
        "family",
        nargs="?",
        choices=["analytical", "eam", "adp", "sw", "tersoff"],
        default=None,
        help="Potential family to generate a TOML template for.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Species labels in the order they should appear in the template.",
    )
    parser.add_argument(
        "--form",
        default=None,
        help="Family-specific form name, such as 'morse' or 'alloy'.",
    )
    parser.add_argument(
        "--expression",
        default=None,
        help="Analytical custom expression to write into the template.",
    )
    parser.add_argument(
        "--parameter-names",
        nargs="+",
        default=None,
        help="Parameter names for an analytical custom expression.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the template to this file instead of stdout.",
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Interactively generate a matching initial TOML and forgeff.train.toml.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset path to read species from in wizard mode.",
    )
    parser.add_argument(
        "--initial-output",
        default="initial.toml",
        help="Initial potential output path in wizard mode.",
    )
    parser.add_argument(
        "--train-output",
        default=None,
        help="Training setting output path in wizard mode.",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=None,
        help="Energy loss weight in wizard mode.",
    )
    parser.add_argument(
        "--forces-weight",
        type=float,
        default=None,
        help="Forces loss weight in wizard mode.",
    )
    parser.add_argument(
        "--stress-weight",
        type=float,
        default=None,
        help="Stress loss weight in wizard mode.",
    )
    parser.add_argument(
        "--energy-offset-mode",
        choices=["off", "manual", "regression"],
        default=None,
        help="Species energy offset mode in wizard mode.",
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        help="Optimizer method in wizard mode.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Maximum optimizer iterations in wizard mode.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Optimizer tolerance in wizard mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )


def run(args: argparse.Namespace) -> None:
    """Run."""
    if getattr(args, "wizard", False) or getattr(args, "dataset", None):
        _wizard_generate(args)
        return
    if getattr(args, "family", None) is None:
        raise ValueError("A potential family is required unless wizard mode is enabled.")
    species = list(args.species or (["Si"] if args.family == "sw" else ["Al"]))
    template = build_template(
        args.family,
        species,
        form=args.form,
        expression=args.expression,
        parameter_names=list(args.parameter_names) if args.parameter_names is not None else None,
    )
    if args.output:
        path = Path(args.output)
        if path.exists() and not args.force:
            raise FileExistsError(f"Template output file already exists: {path}")
        path.write_text(template, encoding="utf-8")
        return
    print(template, end="")


def main() -> None:
    """Command."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    add_arguments(parser)
    args = parser.parse_args()
    run(args)
