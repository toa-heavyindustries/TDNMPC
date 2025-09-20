"""Plot residual timeseries and save figures for a given run directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing logs.csv/summary.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir
    logs = pd.read_csv(run_dir / "logs.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(logs["step"], logs["residual_max"], label="residual_max")
    ax.plot(logs["step"], logs["residual_mean"], label="residual_mean")
    ax.set_xlabel("step")
    ax.set_ylabel("residual")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "residuals.png")
    plt.close(fig)

    # Parse vector columns and plot per-interface time series
    if {"tso_vector", "dso_vector"}.issubset(logs.columns):
        try:
            tso_vectors = logs["tso_vector"].apply(literal_eval)
            dso_vectors = logs["dso_vector"].apply(literal_eval)
            n_if = len(tso_vectors.iloc[0])
            ts = logs["step"].to_numpy()
            # TSO vectors
            fig, ax = plt.subplots(figsize=(9, 4))
            for i in range(n_if):
                ax.plot(ts, [v[i] for v in tso_vectors], label=f"tso_{i}")
            ax.set_xlabel("step")
            ax.set_ylabel("TSO boundary (kW)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(run_dir / "tso_interfaces.png")
            plt.close(fig)

            # DSO vectors
            fig, ax = plt.subplots(figsize=(9, 4))
            for i in range(n_if):
                ax.plot(ts, [v[i] for v in dso_vectors], label=f"dso_{i}")
            ax.set_xlabel("step")
            ax.set_ylabel("DSO boundary (kW)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(run_dir / "dso_interfaces.png")
            plt.close(fig)
        except Exception:
            pass


if __name__ == "__main__":
    main()
