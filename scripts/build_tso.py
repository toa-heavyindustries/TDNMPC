"""Build a DC TSO case, mark boundary buses, and evaluate power balance."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from tso.network import build_tso_case, dc_power_flow, mark_boundary_buses
from utils import ensure_run_dir

LOGGER = logging.getLogger("tso")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bus", type=int, default=30, help="Number of buses in the synthetic case")
    parser.add_argument(
        "--boundary",
        nargs="*",
        type=int,
        default=None,
        help="Boundary bus ids (space separated)",
    )
    parser.add_argument("--tag", default=None, help="Run directory tag")
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("data/tso_case.json"),
        help="Path to export the DC case as JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    case = build_tso_case(args.n_bus)
    LOGGER.info("Built TSO case with %s buses", len(case["buses"]))

    if args.boundary:
        case = mark_boundary_buses(case, args.boundary)
        LOGGER.info("Marked boundary buses: %s", case["boundary"].tolist())

    solution = dc_power_flow(case)
    residual = case["admittance"] @ solution["theta"] - case["injections"]
    LOGGER.info("Power balance residual infinity-norm: %.2e", np.linalg.norm(residual, ord=np.inf))

    run_dir = ensure_run_dir(args.tag)
    export_path = args.export
    export_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(  # lightweight binary store for reproducibility
        run_dir / "tso_case.npz",
        buses=case["buses"],
        slack=np.array([case["slack"]]),
        boundary=case["boundary"],
        admittance=case["admittance"],
        injections=case["injections"],
        theta=solution["theta"],
    )

    export_data = {
        "buses": case["buses"].tolist(),
        "slack": int(case["slack"]),
        "boundary": case["boundary"].tolist(),
        "admittance": case["admittance"].tolist(),
        "injections": case["injections"].tolist(),
    }
    export_path.write_text(json.dumps(export_data, indent=2))
    LOGGER.info("Case exported to %s", export_path)


if __name__ == "__main__":
    main()

