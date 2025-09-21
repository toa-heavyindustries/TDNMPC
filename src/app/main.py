"""Command-line entrypoint for running NMPC simulations.

Usage examples (via uv):
  uv run codex-nmpc --config cfg/smoke_test_b1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sim.runner import simulate_scenario


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NMPC simulation from a config file.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML/JSON configuration file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    simulate_scenario(args.config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

