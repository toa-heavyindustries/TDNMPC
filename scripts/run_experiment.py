"""Run an NMPC scenario using a configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.runner import _simulate
from utils.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, type=Path, help="Path to scenario configuration file")
    parser.add_argument("--tag", help="Override run tag")
    parser.add_argument("--base", help="Override run base directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.cfg)
    run_cfg = cfg.setdefault("run", {})
    if args.tag:
        run_cfg["tag"] = args.tag
    if args.base:
        run_cfg["base"] = args.base

    result = _simulate(cfg)
    print(json.dumps({"run_dir": str(result["run_dir"]), "final_residual": result["final_residual"]}, indent=2))


if __name__ == "__main__":
    main()

