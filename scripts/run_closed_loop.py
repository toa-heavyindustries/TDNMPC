"""Run a simple closed-loop TSO/DSO simulation and save KPIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sim import run_closed_loop
from utils import ensure_run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="Number of simulation steps")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Power oscillation amplitude (MW)")
    parser.add_argument("--solver", default="glpk", help="Pyomo solver name")
    parser.add_argument("--feeder-peak", type=float, default=20.0, help="Feeder peak MW")
    parser.add_argument("--dt-min", type=float, default=5.0, help="Time step in minutes")
    parser.add_argument("--load-csv", type=Path, default=None, help="Optional CSV with load reference column")
    parser.add_argument("--tag", default=None, help="Run directory tag")
    parser.add_argument("--out", type=Path, default=Path("closed_loop.csv"), help="Relative output file name")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    load_profile = None
    if args.load_csv is not None:
        df = pd.read_csv(args.load_csv)
        numeric_cols = df.select_dtypes(include="number")
        if numeric_cols.empty:
            raise ValueError("load CSV must contain at least one numeric column")
        load_profile = numeric_cols.iloc[:, 0].to_numpy()[: args.steps]

    result = run_closed_loop(
        steps=args.steps,
        amplitude=args.amplitude,
        feeder_peak_mw=args.feeder_peak,
        solver=args.solver,
        dt_min=args.dt_min,
        load_profile=load_profile,
    )

    run_dir = ensure_run_dir(args.tag)
    out_path = run_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result.history.to_csv(out_path, index=False)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result.summary, indent=2))

    envelopes_meta = [env.to_dict() for env in result.envelopes]
    (out_path.with_suffix(".envelopes.json")).write_text(json.dumps(envelopes_meta, indent=2))


if __name__ == "__main__":
    main()
