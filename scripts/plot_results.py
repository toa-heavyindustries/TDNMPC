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

def plot_voltage_heatmap(logs: pd.DataFrame, run_dir: Path) -> None:
    """Plot a heatmap of DSO node voltages over time."""
    if "dso_voltages" not in logs.columns or logs["dso_voltages"].isnull().all():
        return

    try:
        # Extract the voltage data for the first control step of each simulation step for the first DSO.
        voltage_series = []
        # The column is read as a string, so we need to evaluate it back to a Python object.
        voltages_col = logs["dso_voltages"].apply(literal_eval)

        for step_voltages in voltages_col:
            if not isinstance(step_voltages, list) or not step_voltages:
                continue
            # Reconstruct DataFrame from split format for the first DSO
            dso0_horizon_dict = step_voltages[0]
            dso0_horizon_df = pd.DataFrame(**dso0_horizon_dict)
            # Get the voltage at the first time step of the horizon
            voltage_series.append(dso0_horizon_df.iloc[0])

        if not voltage_series:
            return

        heatmap_df = pd.concat(voltage_series, axis=1)
        heatmap_df.columns = logs.index[:len(heatmap_df.columns)]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_df, aspect="auto", cmap="viridis", vmin=0.95, vmax=1.05)
        fig.colorbar(im, ax=ax, label="Voltage (p.u.)")

        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("DSO 0 Bus Index")
        ax.set_title("DSO 0 Nodal Voltage Heatmap")

        fig.tight_layout()
        fig.savefig(run_dir / "voltage_heatmap_dso0.png")
        plt.close(fig)
    except Exception as e:
        print(f"Failed to generate voltage heatmap: {e}")

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

    plot_voltage_heatmap(logs, run_dir)

if __name__ == "__main__":
    main()
