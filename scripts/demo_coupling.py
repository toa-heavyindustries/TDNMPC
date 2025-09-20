"""Demonstrate TSO-DSO coupling mechanics with synthetic signals."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from coord.interface import (
    aggregate_dsos_to_tso,
    coupling_residuals,
    define_coupling,
    push_tso_signals_to_dsos,
)
from dso.network import build_ieee33
from tso.network import build_tso_case, mark_boundary_buses
from utils import ensure_run_dir

LOGGER = logging.getLogger("coupling")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bus", type=int, default=6, help="Number of TSO buses")
    parser.add_argument("--tag", default=None, help="Run directory tag")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    tso_case = build_tso_case(args.n_bus)
    boundary_ids = [1, min(2, args.n_bus - 1)]
    tso_case = mark_boundary_buses(tso_case, boundary_ids)

    dso_nets = [build_ieee33(), build_ieee33()]
    mapping = {boundary_ids[0]: (0, 5), boundary_ids[1]: (1, 10)}

    coupler = define_coupling(tso_case, dso_nets, mapping)
    LOGGER.info("Coupling interfaces: %s", coupler["tso_buses"].tolist())

    signals = np.linspace(5.0, 10.0, coupler["n_interfaces"])
    push_tso_signals_to_dsos(signals, dso_nets)

    for net in dso_nets:
        coupling = net.get("coupling")
        if not coupling:
            continue
        coupling["p_response"] = 0.95 * coupling["p_tso"]

    aggregated = aggregate_dsos_to_tso(dso_nets)
    residual = coupling_residuals(signals, aggregated)
    LOGGER.info("Residual stats: %s", residual)

    run_dir = ensure_run_dir(args.tag)
    (run_dir / "coupling.json").write_text(json.dumps(residual, indent=2))
    LOGGER.info("Residual metrics stored at %s", run_dir / "coupling.json")


if __name__ == "__main__":
    main()

