"""Run an NMPC scenario (single or batched) from configuration."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from sim.runner import simulate
from utils import ensure_run_dir
from utils.batch import run_batch
from utils.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, type=Path, help="Scenario configuration file (prefer config/)")
    parser.add_argument("--tag", help="Override run tag")
    parser.add_argument("--base", help="Override run base directory")
    parser.add_argument("--seeds", type=int, nargs="*", help="Batch seeds to execute")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.cfg)
    run_cfg = cfg.setdefault("run", {})
    if args.tag:
        run_cfg["tag"] = args.tag
    if args.base:
        run_cfg["base"] = args.base

    seeds = args.seeds or []

    if seeds:
        base_dir = Path(run_cfg.get("base", "runs"))
        batch_tag = run_cfg.get("tag", "batch") + "_batch"
        batch_dir = ensure_run_dir(batch_tag, base=base_dir)

        def simulator(seed: int) -> dict[str, Any]:
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.setdefault("run", {})
            run_tag = run_cfg.get("tag", "run")
            cfg_copy["run"].update({"tag": f"{run_tag}-s{seed}", "base": str(base_dir)})
            cfg_copy["seed"] = seed
            result = simulate(cfg_copy)
            return {
                "run_dir": str(result["run_dir"]),
                "final_residual": result["final_residual"],
            }

        df = run_batch(simulator, seeds, batch_dir)
        print(df.to_json(orient="records"))
    else:
        result = simulate(cfg)
        payload = {
            "run_dir": str(result["run_dir"]),
            "final_residual": result["final_residual"],
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
