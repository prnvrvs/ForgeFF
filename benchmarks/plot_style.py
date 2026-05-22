"""Shared plotting style helpers for benchmark figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    "reference": "#0f172a",
    "numpy": "#2563eb",
    "numba": "#d97706",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#94a3b8",
            "axes.labelcolor": "#0f172a",
            "axes.titleweight": "semibold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#cbd5e1",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.45,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 2.4,
            "lines.markersize": 5.5,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
        }
    )


def style_axes(ax, *, title: str, xlabel: str = "Number of atoms", ylabel: str = "Mean evaluation time (ms)") -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


def save_figure(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
