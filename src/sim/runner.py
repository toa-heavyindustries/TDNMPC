"""Scenario runner orchestrating forecasts, NMPC, and logging.

Two operation modes are supported:
- Toy consensus mode (backward compatible with the tests): ``cfg`` provides
  ``signals.tso_targets`` and ``signals.dso_targets`` arrays and the runner
  uses simple proximal updates without calling Pyomo.
- Integrated physics mode (recommended for experiments): ``cfg`` provides
  a TSO DC case and a list of DSO parameters. The runner builds and solves
  Pyomo models inside the ADMM callbacks, passing the ADMM iterate ``v`` as
  boundary targets to the TSO model at every update, and solving the DSO
  models per step to produce aggregated boundary injections.
"""

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
from opt.pyomo_tso import (
    TSOParameters,
    build_tso_model,
    extract_solution as extract_tso_solution,
    solve_tso_model,
)
from opt.pyomo_dso import (
    DSOParameters,
    build_dso_model,
    extract_solution as extract_dso_solution,
    solve_dso_model,
)


@dataclass
class ScenarioState:
    controller: NMPCController
    history: list[dict[str, Any]]
    # Toy mode references
    tso_targets: list[np.ndarray] | None = None
    dso_targets: list[np.ndarray] | None = None
    rho: float | None = None
    step_ref: dict[str, int] | None = None
    # Integrated Pyomo mode references
    tso_base: dict[str, Any] | None = None  # admittance, injections, boundary, cost_coeff
    dso_params: list[DSOParameters] | None = None


def _build_controller(cfg: dict[str, Any]) -> ScenarioState:
    """Build an NMPC controller in either toy or integrated mode.

    Integrated mode is activated when both ``cfg['tso']`` and ``cfg['dsos']``
    are present. Otherwise, the function falls back to the toy proximal mode
    used by tests.
    """
    time_cfg = cfg["time"]
    _ = make_horizon(time_cfg["start"], time_cfg["steps"], time_cfg["dt_min"])  # reserved

    admm_dict = cfg.get("admm", {})
    env_cfg = cfg.get("envelope", {})

    if "tso" in cfg and "dsos" in cfg:
        # Integrated physics mode: prepare TSO base parameters and DSO parameter list.
        tso_cfg = cfg["tso"]
        Y = np.asarray(tso_cfg["admittance"], dtype=float)
        injections = np.asarray(tso_cfg["injections"], dtype=float)
        boundary = np.asarray(tso_cfg["boundary"], dtype=int)
        cost_coeff = float(tso_cfg.get("cost_coeff", 30.0))

        dsos_cfg = cfg["dsos"]
        if len(dsos_cfg) != len(boundary):
            raise ValueError("Number of DSOs must match the number of TSO boundary buses")

        dso_params: list[DSOParameters] = []
        for d in dsos_cfg:
            sens = {
                "Rp": np.asarray(d["sens"]["Rp"], dtype=float),
                "Rq": np.asarray(d["sens"].get("Rq", np.zeros_like(d["sens"]["Rp"])), dtype=float),
                "vm_base": np.asarray(d["sens"]["vm_base"], dtype=float),
            }
            horizon_steps = int(time_cfg["steps"])  # use global horizon for now
            # Profiles: accept inline arrays or a CSV path
            prof = d.get("profiles", {})
            if "csv" in prof:
                # Lightweight loader: expect same columns saved by save_profiles
                import pandas as pd

                df = pd.read_csv(prof["csv"])  # type: ignore[arg-type]
                df = df[["load", "pv"]]
            else:
                load_arr = np.asarray(prof.get("load", [0.0] * horizon_steps), dtype=float)
                pv_arr = np.asarray(prof.get("pv", [0.0] * horizon_steps), dtype=float)
                import pandas as pd

                df = pd.DataFrame({"load": load_arr, "pv": pv_arr})
            # Ensure length >= steps
            if len(df) < horizon_steps:
                raise ValueError("profiles length shorter than horizon steps")
            horizon_obj = make_horizon(time_cfg["start"], horizon_steps, time_cfg["dt_min"])
            dso_params.append(
                DSOParameters(
                    sens=sens,
                    profiles=df,
                    horizon=horizon_obj,
                    vmin=float(d.get("vmin", 0.95)),
                    vmax=float(d.get("vmax", 1.05)),
                    penalty_voltage=float(d.get("penalty_voltage", 1e4)),
                    cost_coeff=float(d.get("cost_coeff", 50.0)),
                )
            )

        size = len(boundary)
        rho = float(admm_dict.get("rho", 1.0))

        solvers = cfg.get("solvers", {})
        tso_solver_name = str(solvers.get("tso", "ipopt"))
        dso_solver_name = str(solvers.get("dso", "ipopt"))

        def tso_solver(v: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            """Solve the DC TSO Pyomo model using ADMM iterate ``v`` as boundary targets."""
            params = TSOParameters(
                admittance=Y,
                injections=injections,
                boundary=boundary,
                boundary_targets=np.asarray(v, dtype=float),
                cost_coeff=cost_coeff,
            )
            model = build_tso_model(params)
            solve_tso_model(model, solver=tso_solver_name)
            res = extract_tso_solution(model, params)
            flows_vec = res.flows.loc[boundary.tolist()].to_numpy()
            meta = {"theta": res.theta.to_dict(), "obj": res.objective if hasattr(res, "objective") else None}
            return flows_vec, meta

        def dso_solver(v: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            """Solve each DSO Pyomo model once and aggregate an injection per interface.

            Current aggregation uses the total active power injection at the first
            horizon step as the boundary response for each DSO.
            """
            injections_vec = []
            meta: dict[str, Any] = {"dso_objs": []}
            for p in dso_params:
                model = build_dso_model(p)
                # Note: if GLPK is not available, this will raise; configure solvers via cfg["solvers"].
                solve_dso_model(model, solver=dso_solver_name)
                res = extract_dso_solution(model, p)
                # Aggregate pg at step 0 across buses
                total_pg = float(res.p_injections.iloc[0, :].sum())
                injections_vec.append(total_pg)
                meta["dso_objs"].append(res.objective)
            return np.asarray(injections_vec, dtype=float), meta

        admm_cfg = ADMMConfig(
            size=size,
            rho=rho,
            max_iters=int(admm_dict.get("max_iters", 50)),
            tol_primal=float(admm_dict.get("tol_primal", 1e-4)),
            tol_dual=float(admm_dict.get("tol_dual", 1e-4)),
        )

        nmpc_cfg = NMPCConfig(
            size=size,
            admm=admm_cfg,
            tso_solver=tso_solver,
            dso_solver=dso_solver,
            envelope_margin=float(env_cfg.get("margin", 0.05)),
            envelope_alpha=float(env_cfg.get("alpha", 0.3)),
        )
        controller = NMPCController(nmpc_cfg)
        return ScenarioState(controller=controller, history=[], tso_base={"Y": Y, "inj": injections, "boundary": boundary, "cost": cost_coeff}, dso_params=dso_params)

    # Fallback: toy consensus mode used by tests
    signals = cfg["signals"]
    tso_targets = [np.asarray(x, dtype=float) for x in signals["tso_targets"]]
    dso_targets = [np.asarray(x, dtype=float) for x in signals["dso_targets"]]
    size = tso_targets[0].shape[0]
    rho = float(admm_dict.get("rho", 1.0))

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
        max_iters=int(admm_dict.get("max_iters", 50)),
        tol_primal=float(admm_dict.get("tol_primal", 1e-4)),
        tol_dual=float(admm_dict.get("tol_dual", 1e-4)),
    )

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
        history=[],
        tso_targets=tso_targets,
        dso_targets=dso_targets,
        rho=rho,
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
