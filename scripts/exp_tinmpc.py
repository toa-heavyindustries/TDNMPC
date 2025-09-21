"""Run a 24h TI-NMPC cooperation experiment from a config.

This wires into the existing scenario/runner stack which already supports
time-invariant envelopes and OUR/B3 algorithms (see config/*). Use this script
as a convenience wrapper that tags the run directory and prints the summary.

Usage examples:
  # Soft TI envelope (OUR)
  uv run python scripts/exp_tinmpc.py --cfg config/smoke_test_our.yaml --tag our_demo

  # Hard TI envelope (B3)
  uv run python scripts/exp_tinmpc.py --cfg config/smoke_test_b3.yaml --tag b3_demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.runner import simulate
from utils.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", type=Path, default=Path("config/smoke_test_our.yaml"), help="Scenario YAML config")
    p.add_argument("--tag", default=None, help="Run directory tag override")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.cfg)
    if args.tag:
        cfg.setdefault("run", {})["tag"] = args.tag
    result: dict[str, Any] = simulate(cfg)
    print(json.dumps({"run_dir": str(result.get("run_dir"))}, indent=2))


if __name__ == "__main__":
    main()
