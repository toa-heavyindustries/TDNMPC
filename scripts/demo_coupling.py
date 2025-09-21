"""Demonstrate TSO-DSO coupling mechanics with synthetic signals."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from coord.interface import (
    aggregate_dsos_to_tso,
    coupling_residuals,
    define_coupling,
    push_tso_signals_to_dsos,
)
from dso.network import build_ieee33
from sim import plan_baseline_coupled_system
from tso.network import build_tso_case, mark_boundary_buses
from utils import ensure_run_dir

LOGGER = logging.getLogger("coupling")


def try_int(value: str) -> int | str:
    """Cast ``value`` to ``int`` when possible, otherwise return the raw string."""

    try:
        return int(value)
    except ValueError:
        return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["baseline", "synthetic"],
        default="baseline",
        help="Use the baseline pandapower system or the legacy synthetic case",
    )
    parser.add_argument("--case", default="case39", help="Pandapower TSO case name")
    parser.add_argument(
        "--baseline-boundary",
        nargs="*",
        help="Optional boundary bus identifiers for the baseline case",
    )
    parser.add_argument(
        "--feeder-type",
        choices=["mv", "lv"],
        default="mv",
        help="Distribution feeder template for the baseline mode",
    )
    parser.add_argument(
        "--feeder-peak",
        type=float,
        default=30.0,
        help="Target peak MW per feeder in baseline mode",
    )
    parser.add_argument(
        "--feeder-cosphi",
        type=float,
        default=0.95,
        help="Target power factor per feeder in baseline mode",
    )
    parser.add_argument(
        "--n-bus",
        type=int,
        default=6,
        help="Number of TSO buses for synthetic mode",
    )
    parser.add_argument("--tag", default=None, help="Run directory tag")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.mode == "baseline":
        boundary_labels = None
        if args.baseline_boundary:
            boundary_labels = [try_int(bus) for bus in args.baseline_boundary]

        plan = plan_baseline_coupled_system(
            case_name=args.case,
            boundary_buses=boundary_labels,
            feeder_type=args.feeder_type,
            feeder_peak_mw=args.feeder_peak,
            feeder_cos_phi=args.feeder_cosphi,
        )

        tso_case = plan.tso
        dso_nets = [attachment.feeder.net for attachment in plan.attachments]
        mapping = {
            attachment.boundary_bus: (idx, attachment.feeder.root_bus)
            for idx, attachment in enumerate(plan.attachments)
        }

        LOGGER.info(
            "Baseline plan uses boundary buses %s",
            ", ".join(plan.tso.boundary_bus_names),
        )
    else:
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
