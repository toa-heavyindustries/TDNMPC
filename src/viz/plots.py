"""Visualization functions for NMPC outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def plot_timeseries(df: pd.DataFrame, cols: list[str], out: Path) -> None:
    """Plot given columns as a timeseries and save to disk."""

    if not cols:
        raise ValueError("No columns provided for plotting")

    ax = df[cols].plot(figsize=(8, 4))
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.figure.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out)
    plt.close(ax.figure)


def plot_convergence(hist: pd.DataFrame, out: Path) -> None:
    """Plot ADMM residual convergence curves."""

    required = {"primal_residual", "dual_residual"}
    if not required.issubset(hist.columns):
        missing = required - set(hist.columns)
        raise ValueError(f"History missing columns: {missing}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist.index, hist["primal_residual"], label="primal")
    ax.plot(hist.index, hist["dual_residual"], label="dual")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_convergence_multi(hist: pd.DataFrame, out: Path) -> None:
    """Plot ADMM residuals across all steps in a single figure.

    Expects columns: 'step', 'iter', 'primal_residual', 'dual_residual'.
    Creates a two-panel plot (primal/dual) with per-step trajectories overlaid.
    """

    required = {"step", "iter", "primal_residual", "dual_residual"}
    if not required.issubset(hist.columns):
        missing = required - set(hist.columns)
        raise ValueError(f"History missing columns: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    steps = list(sorted(hist["step"].unique()))

    # Plot per step with transparency
    for s in steps:
        sub = hist.loc[hist["step"] == s]
        axes[0].plot(sub["iter"], sub["primal_residual"], alpha=0.35)
        axes[1].plot(sub["iter"], sub["dual_residual"], alpha=0.35)

    for ax, title in zip(axes, ("primal", "dual")):
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("residual")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
