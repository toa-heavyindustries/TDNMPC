"""Scenario runner orchestrating forecasts, NMPC, and logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from coord.admm import ADMMConfig
from nmpc.controller import NMPCConfig, NMPCController
from utils import ensure_run_dir
from utils.config import load_config
from utils.timewin import make_horizon


@dataclass
class ScenarioState:
    controller: NMPCController
    tso_targets: list[np.ndarray]
    dso_targets: list[np.ndarray]
    rho: float
    history: list[dict[str, Any]]
    step_ref: dict[str, int]


def _build_controller(cfg: dict[str, Any]) -> ScenarioState:
    time_cfg = cfg["time"]
    horizon = make_horizon(time_cfg["start"], time_cfg["steps"], time_cfg["dt_min"])

    signals = cfg["signals"]
    tso_targets = [np.asarray(x, dtype=float) for x in signals["tso_targets"]]
    dso_targets = [np.asarray(x, dtype=float) for x in signals["dso_targets"]]
    size = tso_targets[0].shape[0]
    rho = float(cfg["admm"]["rho"])

    step_ref = {"idx": 0}

    def tso_solver(v: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        target = tso_targets[step_ref["idx"]]
        value = (target + rho * v) / (1.0 + rho)
        return value, {"theta": value}

    def dso_solver(v: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        target = dso_targets[step_ref["idx"]]
        value = (target + rho * v) / (1.0 + rho)
        return value, {"control": value}

    admm_cfg = ADMMConfig(
        size=size,
        rho=rho,
        max_iters=int(cfg["admm"].get("max_iters", 50)),
        tol_primal=float(cfg["admm"].get("tol_primal", 1e-4)),
        tol_dual=float(cfg["admm"].get("tol_dual", 1e-4)),
    )

    env_cfg = cfg.get("envelope", {})
    nmpc_cfg = NMPCConfig(
        size=size,
        admm=admm_cfg,
        tso_solver=tso_solver,
        dso_solver=dso_solver,
        envelope_margin=float(env_cfg.get("margin", 0.05)),
        envelope_alpha=float(env_cfg.get("alpha", 0.3)),
    )

    controller = NMPCController(nmpc_cfg)
    return ScenarioState(
        controller=controller,
        tso_targets=tso_targets,
        dso_targets=dso_targets,
        rho=rho,
        history=[],
        step_ref=step_ref,
    )


def simulate_step(state: dict[str, Any], t: int) -> dict[str, Any]:
    scenario: ScenarioState = state["scenario"]
    scenario.step_ref["idx"] = t
    result = scenario.controller.run_step()

    scenario.history.append(
        {
            "step": t,
            "tso_vector": result.tso_vector.tolist(),
            "dso_vector": result.dso_vector.tolist(),
            "residual_max": result.residuals["max"],
            "residual_mean": result.residuals["mean"],
            "envelope_upper": result.envelope.upper.tolist(),
            "envelope_lower": result.envelope.lower.tolist(),
        }
    )
    return {
        "result": result,
        "history": scenario.history,
    }


def _simulate(cfg: dict[str, Any]) -> dict[str, Any]:
    state = {"scenario": _build_controller(cfg)}

    run_cfg = cfg.get("run", {})
    run_dir = ensure_run_dir(tag=run_cfg.get("tag"), base=run_cfg.get("base", "runs"))

    steps = cfg["time"]["steps"]
    for t in range(steps):
        simulate_step(state, t)

    history = state["scenario"].history
    df = pd.DataFrame(history)
    df.to_csv(run_dir / "logs.csv", index=False)
    summary = {"final_residual": history[-1]["residual_max"], "steps": steps}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return {
        "history": history,
        "run_dir": run_dir,
        "final_residual": history[-1]["residual_max"],
    }


def simulate_scenario(cfg_path: Path | str) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    return _simulate(cfg)

