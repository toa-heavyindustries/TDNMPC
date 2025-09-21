"""Run ADMM consensus and export convergence artifacts.

Two modes:
  1) Toy signals (uses cfg.signals with target sequences)
  2) Model-backed (uses cfg.tso/cfg.dsos via sim.scenario)

Usage:
  uv run python scripts/run_admm.py --cfg config/demo.yaml --tag admm_demo

Outputs under runs/<tag>/:
  - admm_history.csv
  - admm_convergence.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coord.admm import ADMMConfig, run_admm
from sim.scenario import build_scenario
from utils import ensure_run_dir
from utils.config import load_config
from viz.plots import plot_convergence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", type=Path, required=True, help="Scenario YAML with signals or tso/dsos")
    p.add_argument("--tag", default=None, help="Run directory tag")
    return p.parse_args(argv)


def _toy_updates(state: dict[str, Any]):
    # Scenario built in toy mode by providing cfg.signals
    scenario = state["scenario"]
    rho = float(scenario.rho)

    def tso_step(v: np.ndarray) -> np.ndarray:
        target = scenario.tso_targets[scenario.step_ref["idx"]]
        return (target + rho * v) / (1.0 + rho)

    def dso_step(v: np.ndarray) -> np.ndarray:
        target = scenario.dso_targets[scenario.step_ref["idx"]]
        return (target + rho * v) / (1.0 + rho)

    size = int(scenario.tso_targets[0].shape[0])
    cfg = ADMMConfig(
        size=size,
        rho=rho,
        max_iters=scenario.controller.config.admm.max_iters,  # type: ignore[attr-defined]
        tol_primal=scenario.controller.config.admm.tol_primal,  # type: ignore[attr-defined]
        tol_dual=scenario.controller.config.admm.tol_dual,  # type: ignore[attr-defined]
    )
    return tso_step, dso_step, cfg


def _model_updates(state: dict[str, Any]):
    # Use the functions already wired in the controller config
    scenario = state["scenario"]
    conf = scenario.controller.config  # type: ignore[attr-defined]

    def tso_step(v: np.ndarray) -> np.ndarray:
        x, _meta = conf.tso_solver(v)  # returns (vec, meta)
        return x

    def dso_step(v: np.ndarray) -> np.ndarray:
        z, _meta = conf.dso_solver(v)
        return z

    cfg = conf.admm
    return tso_step, dso_step, cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.cfg)
    run_dir = ensure_run_dir(args.tag)

    state = {"scenario": build_scenario(cfg)}
    scenario = state["scenario"]

    # Decide mode based on presence of toy signals
    if scenario.tso_targets is not None and scenario.dso_targets is not None:
        tso_step, dso_step, admm_cfg = _toy_updates(state)
    else:
        tso_step, dso_step, admm_cfg = _model_updates(state)

    result = run_admm(tso_step, dso_step, admm_cfg)
    hist = pd.DataFrame(result["history"]) if result.get("history") else pd.DataFrame()

    # Persist artifacts
    hist_path = run_dir / "admm_history.csv"
    if not hist.empty:
        hist.set_index("iter").to_csv(hist_path)
        try:
            plot_convergence(hist.set_index("iter"), run_dir / "admm_convergence.png")
        except Exception:
            pass

    summary = {
        "converged": bool(result.get("converged")),
        "iterations": int(result.get("iterations", 0)),
        "final_primal": float(hist["primal_residual"].iloc[-1]) if not hist.empty else None,
        "final_dual": float(hist["dual_residual"].iloc[-1]) if not hist.empty else None,
    }
    (run_dir / "admm_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()

