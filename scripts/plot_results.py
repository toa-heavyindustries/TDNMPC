"""Plot residual timeseries and save figures for a given run directory.

Extended to also:
  - Visualize LV snapshots summary if present (bar charts of KPIs)
  - Visualize local timeseries run (bus voltage heatmap and SoC trajectory)
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path

import json
import numpy as np
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

def plot_pcc_trajectory(logs: pd.DataFrame, run_dir: Path) -> None:
    """Plot actual, forecast, and envelope for the PCC power trajectory."""
    if not all(col in logs.columns for col in ["dso_vector", "dso_h", "envelope_upper", "envelope_lower"]):
        return

    try:
        # Use literal_eval to parse string-formatted list data
        dso_vector = logs["dso_vector"].apply(literal_eval)
        dso_h = logs["dso_h"].apply(literal_eval)
        env_up = logs["envelope_upper"].apply(literal_eval)
        env_low = logs["envelope_lower"].apply(literal_eval)

        # Focus on the first interface (index 0)
        if_idx = 0

        actual_traj = [v[if_idx] for v in dso_vector]
        upper_bound = [v[if_idx] for v in env_up]
        lower_bound = [v[if_idx] for v in env_low]
        steps = logs["step"].to_numpy()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual trajectory and envelope
        ax.plot(steps, actual_traj, "-o", markersize=4, label="Actual Power")
        ax.fill_between(steps, lower_bound, upper_bound, color="gray", alpha=0.3, label="Envelope")

        # Plot a representative forecast (e.g., from the last step)
        forecast_step = len(logs) - 1
        if forecast_step >= 0:
            forecast_horizon = dso_h.iloc[forecast_step]
            if forecast_horizon and len(forecast_horizon) > if_idx:
                forecast_values = [h[if_idx] for h in forecast_horizon]
                forecast_steps = range(forecast_step, forecast_step + len(forecast_values))
                ax.plot(forecast_steps, forecast_values, "--d", markersize=4, label=f"Forecast at t={forecast_step}")

        ax.set_xlabel("Simulation Step")
        ax.set_ylabel(f"PCC Interface {if_idx} Power (kW)")
        ax.set_title(f"PCC Interface {if_idx}: Actual vs. Forecast & Envelope")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / f"pcc_trajectory_if{if_idx}.png")
        plt.close(fig)

    except Exception as e:
        print(f"Failed to generate PCC trajectory plot: {e}")

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir
    logs_path = run_dir / "logs.csv"
    logs = pd.read_csv(logs_path) if logs_path.exists() else pd.DataFrame()

    if not logs.empty and {"step","residual_max","residual_mean"}.issubset(logs.columns):
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

    if not logs.empty:
        plot_voltage_heatmap(logs, run_dir)
        plot_pcc_trajectory(logs, run_dir)

    # LV snapshots: plot KPIs from summary if present
    lv_summary = run_dir / "lv_snapshots.summary.json"
    if lv_summary.exists():
        try:
            data = json.loads(lv_summary.read_text())
            cases = list(data.keys())
            vm_min = [data[c]["vm_min"] for c in cases]
            vm_max = [data[c]["vm_max"] for c in cases]
            viol = [data[c]["voltage_violations"] for c in cases]

            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(cases))
            ax.bar(x - 0.2, vm_min, width=0.4, label="vm_min")
            ax.bar(x + 0.2, vm_max, width=0.4, label="vm_max")
            ax.axhline(0.95, color="green", linestyle="--", linewidth=1)
            ax.axhline(1.05, color="green", linestyle="--", linewidth=1)
            ax.set_xticks(x, cases, rotation=20)
            ax.set_ylim(0.9, max(1.08, max(vm_max)+0.01))
            ax.set_ylabel("Voltage (p.u.)")
            ax.set_title("LV Snapshots: Voltage Range")
            ax.legend()
            fig.tight_layout()
            fig.savefig(run_dir / "lv_voltage_range.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(cases, viol, color="#cc6677")
            ax.set_ylabel("Count")
            ax.set_title("LV Snapshots: Voltage Violations")
            fig.tight_layout()
            fig.savefig(run_dir / "lv_voltage_violations.png")
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"Failed to plot LV snapshots: {e}")

    # Timeseries local run: plot voltage heatmap and SoC if present
    res_bus_dir = run_dir / "res_bus"
    if res_bus_dir.exists():
        try:
            vm_pickle = res_bus_dir / "vm_pu.p"
            if vm_pickle.exists():
                df = pd.read_pickle(vm_pickle)
                fig, ax = plt.subplots(figsize=(9, 4))
                im = ax.imshow(df.T, aspect="auto", cmap="viridis", vmin=0.95, vmax=1.05)
                fig.colorbar(im, ax=ax, label="V (p.u.)")
                ax.set_xlabel("Time step")
                ax.set_ylabel("Bus index")
                ax.set_title("Timeseries: Bus Voltage Heatmap")
                fig.tight_layout()
                fig.savefig(run_dir / "ts_voltage_heatmap.png")
                plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"Failed to plot timeseries voltages: {e}")

    soc_csv = run_dir / "soc.csv"
    if soc_csv.exists():
        try:
            soc = pd.read_csv(soc_csv)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(soc.index, soc.iloc[:, 0], label="SoC (MWh)")
            ax.set_xlabel("Time step")
            ax.set_ylabel("SoC (MWh)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(run_dir / "ts_soc.png")
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"Failed to plot SoC: {e}")

if __name__ == "__main__":
    main()
