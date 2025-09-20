"""Compute evaluation metrics for a run directory and save metrics.json."""

from __future__ import annotations

import argparse
import json
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import coupling_rmse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing logs.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir
    logs = pd.read_csv(run_dir / "logs.csv")

    # Parse vector columns
    tso = logs["tso_vector"].apply(literal_eval)
    dso = logs["dso_vector"].apply(literal_eval)
    n_if = len(tso.iloc[0])

    # Flatten residuals across steps and interfaces for RMSE
    diffs = []
    for idx in range(len(logs)):
        tv = np.asarray(tso.iloc[idx], dtype=float)
        dv = np.asarray(dso.iloc[idx], dtype=float)
        diffs.extend((tv - dv).tolist())
    diffs_series = pd.Series(diffs, dtype=float)

    metrics = {
        "rmse": coupling_rmse(diffs_series),
        "mean_residual": float(logs["residual_mean"].mean()),
        "final_residual": float(logs["residual_max"].iloc[-1]),
        "steps": int(logs["step"].max() + 1 if "step" in logs else len(logs)),
        "interfaces": int(n_if),
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

