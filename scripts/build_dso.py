"""Build a DSO network, export its data, and validate LinDistFlow sensitivities."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dso import ac_power_flow, build_ieee33, export_net
from models import linearize_lindistflow, validate_linearization
from utils import ensure_run_dir

LOGGER = logging.getLogger("dso")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["ieee33"], default="ieee33", help="Network case")
    parser.add_argument("--out", type=Path, default=Path("data/ieee33.json"), help="Export path")
    parser.add_argument("--tag", default=None, help="Output run directory tag")
    parser.add_argument("--samples", type=int, default=20, help="Validation sample size")
    parser.add_argument("--tol", type=float, default=0.03, help="Validation tolerance")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    net = build_ieee33()
    LOGGER.info("Network built with %s buses and %s lines", len(net.bus), len(net.line))

    base_pf = ac_power_flow(net)
    LOGGER.info("Base power flow computed. Min voltage: %.3f pu", base_pf["vm_pu"].min())

    sens = linearize_lindistflow(net, base_pf)

    export_net(net, args.out)
    LOGGER.info("Network exported to %s", args.out)

    run_dir = ensure_run_dir(args.tag)
    metrics_path = run_dir / "lin_check.json"
    metrics = validate_linearization(net, sens, n_samples=args.samples, tol=args.tol)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Validation metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()

