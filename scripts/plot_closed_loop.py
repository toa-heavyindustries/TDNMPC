"""Generate plots for closed-loop simulation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="Path to closed_loop.csv or equivalent")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = pd.read_csv(args.history)

    out_dir = args.history.parent

    fig, ax = plt.subplots(figsize=(8, 4))
    for bus, group in df.groupby("bus"):
        ax.plot(group["step"], group["p_target"], label=f"bus {bus} target")
        ax.plot(group["step"], group["p_tso"], linestyle="--", label=f"bus {bus} tso")
    ax.set_xlabel("step")
    ax.set_ylabel("Power (MW)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "closed_loop_power.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], df["residual"], label="residual")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("Residual (MW)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "closed_loop_residual.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], df["soc"], label="SoC")
    ax.set_xlabel("step")
    ax.set_ylabel("State of charge (arb.)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "closed_loop_soc.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
