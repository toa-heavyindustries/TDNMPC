"""Lean scenario runner: orchestrate steps and write artifacts.

Controller construction is delegated to ``sim.scenario`` and I/O helpers to
``sim.io_utils`` to keep complexity low and responsibilities separated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from sim.io_utils import (
    init_run_and_logging,
    write_history_and_metrics,
    write_summary_and_figs,
    write_wide_trace,
)
from sim.scenario import ScenarioState, build_scenario
from utils.config import load_config
from utils.random import set_global_seed
from viz.plots import plot_convergence

logger = logging.getLogger(__name__)


def simulate_step(state: dict[str, Any], t: int) -> dict[str, Any]:
    scenario: ScenarioState = state["scenario"]
    if scenario.step_ref is not None:
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
            # Optional horizon-wise data if provided by solvers
            "tso_h": getattr(result, "tso_metadata", {}).get("tso_h") if hasattr(result, "tso_metadata") else None,
            "dso_h": getattr(result, "dso_metadata", {}).get("dso_h") if hasattr(result, "dso_metadata") else None,
            "dso_voltages": getattr(result, "dso_metadata", {}).get("dso_voltages") if hasattr(result, "dso_metadata") else None,
        }
    )
    return {
        "result": result,
        "history": scenario.history,
    }


def simulate(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)

    state = {"scenario": build_scenario(cfg)}

    time_cfg = cfg.get("time", {})
    steps = int(time_cfg.get("steps", 0))
    dt_min = time_cfg.get("dt_min")

    run_dir, log_dir, logger = init_run_and_logging(cfg)
    logger.info(
        "Simulation run created | seed=%d | steps=%d | dt_min=%s | run_dir=%s",
        seed,
        steps,
        dt_min,
        run_dir,
    )

    admm_histories: list[list[dict[str, float]]] = []
    for t in range(steps):
        ret = simulate_step(state, t)
        result = ret.get("result")
        if result is not None:
            if hasattr(result, "residuals"):
                res = result.residuals
                logger.info(
                    "Step %d/%d | residual_max=%.3e | residual_mean=%.3e",
                    t + 1,
                    steps,
                    float(res.get("max", 0.0)),
                    float(res.get("mean", 0.0)),
                )
            if hasattr(result, "admm_history"):
                admm_histories.append(result.admm_history)

    history = state["scenario"].history
    df = write_history_and_metrics(
        history,
        run_dir,
        log_dir,
        logger,
        metrics_file=cfg.get("logging", {}).get("metrics_file", "metrics.csv"),
    )
    write_wide_trace(df, run_dir)
    write_summary_and_figs(df, history, run_dir)

    combined_rows: list[dict[str, float]] = []
    for t, hist in enumerate(admm_histories):
        if not hist:
            continue
        hdf = pd.DataFrame(hist)
        csv_path = run_dir / f"admm_history_step_{t}.csv"
        hdf.to_csv(csv_path, index=False)
        try:
            plot_convergence(hdf.set_index("iter"), run_dir / f"admm_conv_step_{t}.png")
        except Exception:
            pass
        hdf = hdf.copy()
        hdf["step"] = t
        combined_rows.extend(hdf.to_dict(orient="records"))

    if combined_rows:
        all_df = pd.DataFrame(combined_rows)
        all_df.to_csv(run_dir / "admm_history_all.csv", index=False)
        try:
            from viz.plots import plot_convergence_multi

            plot_convergence_multi(all_df, run_dir / "admm_convergence_all.png")
        except Exception:
            pass

    return {
        "history": history,
        "run_dir": run_dir,
        "final_residual": history[-1]["residual_max"],
    }


def simulate_scenario(cfg_path: Path | str) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    return simulate(cfg)

