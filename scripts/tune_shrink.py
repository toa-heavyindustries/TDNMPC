"""Monte-Carlo sweep to size envelope shrink for desired quantiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim import run_closed_loop
from utils import ensure_run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="Steps per simulation run")
    parser.add_argument("--cases", type=int, default=20, help="Number of Monte Carlo samples")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Noise std dev for each sample (MW)",
    )
    parser.add_argument("--amplitude", type=float, default=1.0, help="Base sinusoid amplitude (MW)")
    parser.add_argument("--solver", default="gurobi", choices=["gurobi", "ipopt"], help="Pyomo solver")
    parser.add_argument("--target-quantile", type=float, default=0.95, help="Quantile for shrink sizing")
    parser.add_argument("--dt-min", type=float, default=5.0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", type=Path, default=Path("shrink_sweep.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(42)

    results = []
    sigma_vals = []
    for idx in range(args.cases):
        load_profile = rng.normal(scale=args.amplitude, size=args.steps)
        res = run_closed_loop(
            steps=args.steps,
            amplitude=0.0,
            solver=args.solver,
            load_profile=load_profile,
            noise_std=args.noise_std,
            dt_min=args.dt_min,
        )
        summary = res.summary
        sigma_vals.append(summary["sigma_p95"])
        results.append({
            "case": idx,
            "sigma_p95": summary["sigma_p95"],
            "max_abs_residual": summary["max_abs_residual"],
            "voltage_violations": summary["voltage_violations"],
        })

    sigma_array = np.asarray(sigma_vals)
    target_sigma = float(np.quantile(sigma_array, args.target_quantile))

    run_dir = ensure_run_dir(args.tag)
    out_path = run_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "target_quantile": args.target_quantile,
                "sigma_values": sigma_vals,
                "recommended_shrink": target_sigma,
                "cases": results,
            },
            indent=2,
        )
    )

 
if __name__ == "__main__":
    main()
