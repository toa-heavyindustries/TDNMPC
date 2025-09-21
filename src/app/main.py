"""Command-line entrypoint for running NMPC simulations.

Usage examples (via uv):
  uv run codex-nmpc --config smoke_test_b1
"""

from __future__ import annotations

import argparse
import sys

from sim.runner import simulate_scenario
from utils.config import list_available_configs, resolve_config_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NMPC simulation from a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Config file path or name under config/. "
            "Examples: --config config/demo.yaml or --config demo"
        ),
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List discovered configs under config/ and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_configs:
        for p in list_available_configs():
            print(p)
        return 0

    cfg_path = resolve_config_path(args.config)
    simulate_scenario(cfg_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
