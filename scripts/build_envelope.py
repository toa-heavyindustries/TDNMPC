"""Generate boundary envelopes for the baseline coupled system."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dso.network import ac_power_flow
from interface.envelope import BoxEnvelope, build_box_envelope, export_envelopes
from models.lindistflow import linearize_lindistflow, validate_linearization
from sim import plan_baseline_coupled_system
from utils import ensure_run_dir

LOGGER = logging.getLogger("envelope")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case39", help="Pandapower TSO case name")
    parser.add_argument(
        "--boundary",
        nargs="*",
        help="Optional boundary bus labels for the baseline plan",
    )
    parser.add_argument("--feeder-type", choices=["mv", "lv"], default="mv")
    parser.add_argument("--feeder-peak", type=float, default=20.0)
    parser.add_argument("--p-margin", type=float, default=None)
    parser.add_argument("--q-margin", type=float, default=None)
    parser.add_argument("--tag", default=None, help="Run directory tag")
    parser.add_argument("--samples", type=int, default=30, help="Sensitivity validation samples")
    parser.add_argument("--tol", type=float, default=0.03, help="Sensitivity validation tolerance")
    parser.add_argument("--delta-mw", type=float, default=0.5)
    parser.add_argument("--delta-mvar", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/boundary.json"))
    return parser.parse_args(argv)


def build_envelope_collection(args: argparse.Namespace) -> list[BoxEnvelope]:
    boundary_labels = None
    if args.boundary:
        parsed = []
        for token in args.boundary:
            try:
                parsed.append(int(token))
            except ValueError:
                parsed.append(token)
        boundary_labels = parsed

    plan = plan_baseline_coupled_system(
        case_name=args.case,
        boundary_buses=boundary_labels,
        feeder_type=args.feeder_type,
        feeder_peak_mw=args.feeder_peak,
    )

    envelopes: list[BoxEnvelope] = []

    for attachment in plan.attachments:
        feeder = attachment.feeder
        net = feeder.net
        env = build_box_envelope(
            feeder,
            net=net,
            measurement=None,
            p_margin=args.p_margin,
            q_margin=args.q_margin,
        )

        base_pf = ac_power_flow(net)
        sens = linearize_lindistflow(
            net,
            base_pf,
            perturbation_mw=args.delta_mw,
            perturbation_mvar=args.delta_mvar,
        )
        stats = validate_linearization(
            net,
            sens,
            n_samples=args.samples,
            tol=args.tol,
            max_delta_mw=args.delta_mw,
            max_delta_mvar=args.delta_mvar,
        )
        env.metadata.update(
            {
                "boundary_bus": attachment.boundary_bus,
                "boundary_bus_name": attachment.boundary_bus_name,
                "lin_mape": stats.get("mape"),
                "lin_max_abs": stats.get("max_abs"),
                "lin_passed": stats.get("passed"),
            }
        )
        envelopes.append(env)

    return envelopes


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    envelopes = build_envelope_collection(args)
    run_dir = ensure_run_dir(args.tag)
    out_path = args.out if args.out.is_absolute() else run_dir / args.out
    export_envelopes(envelopes, out_path)
    LOGGER.info("Envelopes exported to %s", out_path)

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "count": len(envelopes),
        "summary": [env.metadata for env in envelopes],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
