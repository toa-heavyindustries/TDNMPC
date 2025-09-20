"""Plot residual timeseries and save figures for a given run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

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

    # Quick-check plot for envelope bounds range per step (optional)
    try:
        upper = logs["envelope_upper"].apply(lambda x: len(str(x)))
        lower = logs["envelope_lower"].apply(lambda x: len(str(x)))
    except KeyError:
        return


if __name__ == "__main__":
    main()

