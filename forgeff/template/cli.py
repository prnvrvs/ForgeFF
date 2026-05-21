"""`forgeff template`."""

from __future__ import annotations

import argparse
from pathlib import Path
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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument(
        "family",
        choices=["analytical", "eam", "adp", "sw", "tersoff"],
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
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )


def run(args: argparse.Namespace) -> None:
    """Run."""
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
