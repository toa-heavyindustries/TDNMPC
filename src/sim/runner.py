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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from coord.admm import ADMMConfig
from nmpc.controller import NMPCConfig, NMPCController
from nmpc.greedy import GreedyConfig, GreedyController
from utils import ensure_run_dir
from utils.config import load_config
from utils.logging_utils import setup_structured_logging
from utils.random import set_global_seed
from utils.timewin import make_horizon, align_profiles
from viz.plots import plot_convergence
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
    apply_envelope_pg_bounds,
)
from sim.forecast import sample_forecast
from coord.ti_env import compute_bounds_from_scenarios


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
    # Coupling mapping: length=size arrays mapping each TSO boundary slot to (dso_idx, dso_bus_idx)
    dso_indices: np.ndarray | None = None
    dso_bus_indices: np.ndarray | None = None
    # Algorithm selection
    algorithm: str = "OUR"
    # Precomputed TI envelope specs per DSO index
    dso_env_specs: dict[int, dict[str, Any]] | None = None


def _build_controller(cfg: dict[str, Any]) -> ScenarioState:
    """Build an NMPC controller in either toy or integrated mode.

    Integrated mode is activated when both ``cfg['tso']`` and ``cfg['dsos']``
    are present. Otherwise, the function falls back to the toy proximal mode
    used by tests.
    """
    time_cfg = cfg["time"]
    # Allow shorter prediction horizon than total simulation steps
    n_pred = int(cfg.get("N_pred", time_cfg["steps"]))
    _ = make_horizon(time_cfg["start"], n_pred, time_cfg["dt_min"])  # reserved

    admm_dict = cfg.get("admm", {})
    env_cfg = cfg.get("envelope", {})
    alg = str(cfg.get("algorithm", "OUR")).upper()

    if "tso" in cfg and "dsos" in cfg:
        # Integrated physics mode: prepare TSO base parameters and DSO parameter list.
        tso_cfg = cfg["tso"]
        Y = np.asarray(tso_cfg["admittance"], dtype=float)
        injections = np.asarray(tso_cfg["injections"], dtype=float)
        boundary = np.asarray(tso_cfg["boundary"], dtype=int)
        cost_coeff = float(tso_cfg.get("cost_coeff", 30.0))
        bounds_cfg = tso_cfg.get("bounds")
        lower_bounds = None
        upper_bounds = None
        if bounds_cfg:
            if "lower" in bounds_cfg:
                lower_bounds = np.asarray(bounds_cfg["lower"], dtype=float)
            if "upper" in bounds_cfg:
                upper_bounds = np.asarray(bounds_cfg["upper"], dtype=float)

        dsos_cfg = cfg["dsos"]
        # Allow multiple interfaces per DSO; mapping will define slot ownership.

        dso_params: list[DSOParameters] = []
        for d in dsos_cfg:
            sens = {
                "Rp": np.asarray(d["sens"]["Rp"], dtype=float),
                "Rq": np.asarray(d["sens"].get("Rq", np.zeros_like(d["sens"]["Rp"])), dtype=float),
                "vm_base": np.asarray(d["sens"]["vm_base"], dtype=float),
            }
            horizon_steps = int(n_pred)
            # Profiles: accept inline arrays or a CSV path
            prof = d.get("profiles", {})
            if "csv" in prof:
                # Load CSV with time column and align to horizon index
                import pandas as pd

                df_raw = pd.read_csv(prof["csv"], parse_dates=["time"])  # type: ignore[arg-type]
                df_raw = df_raw.set_index("time")
                idx = pd.date_range(start=time_cfg["start"], periods=horizon_steps, freq=f"{time_cfg['dt_min']}min")
                aligned = align_profiles(idx, {"load": df_raw["load"], "pv": df_raw["pv"]})
                df = aligned.reset_index(drop=True)
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
        # Coupling mapping: map each boundary bus to (dso_idx, dso_bus_idx)
        mapping = cfg.get("mapping", {})
        if not mapping:
            # Default: one DSO per boundary slot in order, bus 0
            dso_indices = np.arange(size, dtype=int)
            dso_bus_indices = np.zeros(size, dtype=int)
        else:
            dso_indices = np.empty(size, dtype=int)
            dso_bus_indices = np.empty(size, dtype=int)
            for k, bus in enumerate(boundary.tolist()):
                if str(bus) in mapping:
                    pair = mapping[str(bus)]
                elif bus in mapping:
                    pair = mapping[bus]
                else:
                    raise ValueError(f"Missing mapping for boundary bus {bus}")
                dso_indices[k] = int(pair[0])
                dso_bus_indices[k] = int(pair[1])
        rho = float(admm_dict.get("rho", 1.0))

        solvers = cfg.get("solvers", {})
        tso_solver_name = str(solvers.get("tso", "ipopt"))
        tso_solver_options = solvers.get("tso_options", {})
        dso_solver_name = str(solvers.get("dso", "ipopt"))

        # Precompute TI envelope bounds if enabled or required (OUR/B3)
        dso_env_specs: dict[int, dict[str, Any]] = {}
        ti_cfg = cfg.get("ti_envelope", {}) | env_cfg
        ti_enabled = bool(ti_cfg.get("enabled", alg in {"OUR", "B3"}))
        scen_count = int(ti_cfg.get("scenario_count", 8))
        alpha = float(ti_cfg.get("alpha", 1.0))
        margin = float(ti_cfg.get("margin", 0.0))
        fcfg = cfg.get("forecast", {})
        sigL = float(fcfg.get("sigma_load", 0.05))
        sigPV = float(fcfg.get("sigma_pv", 0.15))
        rho_err = float(fcfg.get("rho", 0.6))

        # Build mapping slots per DSO index
        slot_groups: dict[int, dict[str, Any]] = {}
        size = len(boundary)
        for slot in range(size):
            di = int(dso_indices[slot]) if 'dso_indices' in locals() or 'dso_indices' in globals() else slot
            bi = int(dso_bus_indices[slot]) if 'dso_bus_indices' in locals() or 'dso_bus_indices' in globals() else 0
            g = slot_groups.setdefault(di, {"slots": [], "bus_indices": []})
            g["slots"].append(slot)
            g["bus_indices"].append(bi)

        if ti_enabled:
            for di, group in slot_groups.items():
                p = dso_params[di]
                T = int(p.horizon.steps)
                load_series = p.profiles["load"].iloc[:T]
                pv_series = p.profiles["pv"].iloc[:T]
                # Sample per-scenario series (assumed same across buses)
                L = sample_forecast(load_series, sigma=sigL, rho=rho_err, horizon=T, n=scen_count, mode="relative", clamp_nonneg=False)
                PV = sample_forecast(pv_series, sigma=sigPV, rho=rho_err, horizon=T, n=scen_count, mode="relative", clamp_nonneg=True)
                # Net generation need (kW) assumed ~ max(load - pv, 0)
                G = np.maximum(L - PV, 0.0)
                # Build bounds per slot (bus)
                m = len(group["bus_indices"])
                # Scenarios are identical across buses; tile across bus dimension
                G3 = np.repeat(G[:, :, None], m, axis=2)
                low, up = compute_bounds_from_scenarios(G3, alpha=alpha, margin=margin)
                dso_env_specs[di] = {"bus_indices": group["bus_indices"], "lower": low, "upper": up}

        def tso_solver(v: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            """Solve the DC TSO Pyomo model using ADMM iterate ``v`` as boundary targets."""
            params = TSOParameters(
                admittance=Y,
                injections=injections,
                boundary=boundary,
                boundary_targets=np.asarray(v, dtype=float),
                rho=rho,
                cost_coeff=cost_coeff,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            model = build_tso_model(params)
            try:
                solve_tso_model(model, solver=tso_solver_name, options=tso_solver_options)
                res = extract_tso_solution(model, params)
                flows_vec = res.flows.loc[boundary.tolist()].to_numpy()
                obj = res.objective if hasattr(res, "objective") else None
                theta_meta = res.theta.to_dict()
            except Exception as exc:
                # Solver unavailable or failed: produce zero vector fallback
                flows_vec = np.zeros(len(boundary), dtype=float)
                obj = None
                theta_meta = {}
            # Build a horizon-repeated matrix for convenience (same static DC per step)
            T = int(dso_params[0].horizon.steps) if dso_params else 1
            flows_mat = np.tile(flows_vec[None, :], (T, 1))
            meta = {
                "theta": theta_meta,
                "obj": obj,
                "tso_h": flows_mat.tolist(),
                "solver": tso_solver_name,
            }
            return flows_vec, meta

        def dso_solver(v: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            """Solve each DSO Pyomo model once and aggregate an injection per interface.

            Current aggregation uses the total active power injection at the first
            horizon step as the boundary response for each DSO.
            """
            # Solve each distinct DSO once, then assemble per-interface vector by mapping bus indices
            meta: dict[str, Any] = {"dso_objs": [], "dso_voltages": []}
            solved: list[tuple[DSOParameters, Any]] = []
            for di, p in enumerate(dso_params):
                model = build_dso_model(p)
                # Apply TI envelope bounds if configured for this DSO
                if ti_enabled and alg in {"OUR", "B3"} and di in dso_env_specs:
                    spec = dso_env_specs[di]
                    buses = list(spec["bus_indices"])  # bus indices within DSO model
                    low = np.asarray(spec["lower"], dtype=float)
                    up = np.asarray(spec["upper"], dtype=float)
                    pen = float(ti_cfg.get("penalty", 1000.0)) if alg == "OUR" else None
                    dt_hours = p.horizon.dt_min / 60.0
                    apply_envelope_pg_bounds(model, buses, low, up, penalty=pen, dt_hours=dt_hours)
                try:
                    solve_dso_model(model, solver=dso_solver_name)
                    res = extract_dso_solution(model, p)
                except Exception as exc:
                    # Fallback: zero injections time series with correct shape
                    import pandas as _pd
                    Tloc = int(p.horizon.steps)
                    buses = sorted(range(len(p.sens["vm_base"])))
                    zero = _pd.DataFrame(np.zeros((Tloc, len(buses))), index=range(Tloc), columns=buses)
                    class _Res:
                        def __init__(self, pg: _pd.DataFrame):
                            self.p_injections = pg
                            self.voltage = pg  # Use same shape for voltage fallback
                            self.objective = 0.0
                    res = _Res(zero)
                solved.append((p, res))
                meta["dso_objs"].append(res.objective)
                meta["dso_voltages"].append(res.voltage.to_dict(orient="split"))

            # Build interface vector ordered by TSO boundary slots, and full-horizon matrix
            T = int(dso_params[0].horizon.steps) if dso_params else 1
            mat = np.zeros((T, size), dtype=float)
            for slot in range(size):
                di = int(dso_indices[slot])
                bi = int(dso_bus_indices[slot])
                _, res = solved[di]
                series = res.p_injections.iloc[:, bi].to_numpy()
                if series.shape[0] < T:
                    # Should not happen; guard by truncating/padding
                    tmp = np.zeros(T)
                    tmp[: series.shape[0]] = series
                    series = tmp
                mat[:, slot] = series[:T]
            vec = mat[0, :].copy()
            meta["dso_h"] = mat.tolist()
            return vec, meta

        admm_cfg = ADMMConfig(
            size=size,
            rho=rho,
            max_iters=int(admm_dict.get("max_iters", 50)),
            tol_primal=float(admm_dict.get("tol_primal", 1e-4)),
            tol_dual=float(admm_dict.get("tol_dual", 1e-4)),
        )

        # Algorithm-specific controller
        if alg == "B0":
            gcfg = GreedyConfig(
                size=size,
                dso_solver=dso_solver,
                envelope_margin=float(env_cfg.get("margin", 0.05)),
                envelope_alpha=float(env_cfg.get("alpha", 0.3)),
            )
            controller = GreedyController(gcfg)  # type: ignore[assignment]
        else:
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
            controller=controller,  # type: ignore[arg-type]
            history=[],
            tso_base={"Y": Y, "inj": injections, "boundary": boundary, "cost": cost_coeff},
            dso_params=dso_params,
            dso_indices=dso_indices,
            dso_bus_indices=dso_bus_indices,
            algorithm=alg,
            dso_env_specs=dso_env_specs if ti_enabled else None,
        )

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
    if scenario.step_ref is not None:
        scenario.step_ref["idx"] = t
    # Centralized (B1): one-shot DSO then TSO
    if getattr(scenario, "algorithm", "OUR") == "B1":
        # Compute local DSO response ignoring TSO signal first
        def _get_dso_vec() -> tuple[np.ndarray, Any]:
            # Reuse the controller's dso_solver via a small proxy ADMM call
            ctrl = scenario.controller
            # Access underlying solver by using a zero vector through the public API
            # Fall back: if controller lacks attribute, call run_step
            try:
                dso_solver = ctrl.config.dso_solver  # type: ignore[attr-defined]
                dso_vec, dso_meta = dso_solver(np.zeros(ctrl.config.size))  # type: ignore[attr-defined]
            except Exception:
                step = ctrl.run_step()
                dso_vec, dso_meta = step.dso_vector, step.dso_metadata
            return dso_vec, dso_meta

        dso_vec, dso_meta = _get_dso_vec()
        # Feed to TSO as boundary targets
        def _get_tso_vec() -> tuple[np.ndarray, Any]:
            try:
                tso_solver = scenario.controller.config.tso_solver  # type: ignore[attr-defined]
                tso_vec, tso_meta = tso_solver(dso_vec)
            except Exception:
                tso_vec, tso_meta = dso_vec.copy(), {}
            return tso_vec, tso_meta

        tso_vec, tso_meta = _get_tso_vec()

        # Craft result in NMPCStepResult-like shape
        from nmpc.controller import NMPCStepResult
        from coord.ti_env import update_envelope, create_envelope

        if not hasattr(scenario.controller, "envelope"):
            scenario.controller.envelope = create_envelope(size=dso_vec.shape[0])  # type: ignore[attr-defined]
        scenario.controller.envelope = update_envelope(scenario.controller.envelope, dso_vec)  # type: ignore[attr-defined]
        env = scenario.controller.envelope  # type: ignore[attr-defined]
        result = NMPCStepResult(
            tso_vector=tso_vec,
            dso_vector=dso_vec,
            residuals={"max": float(np.max(np.abs(tso_vec - dso_vec))), "mean": float(np.mean(np.abs(tso_vec - dso_vec))), "l2": float(np.linalg.norm(tso_vec - dso_vec))},
            admm_history=[],
            envelope=env,
            tso_metadata=tso_meta,
            dso_metadata=dso_meta,
        )
    else:
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


def _simulate(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)

    state = {"scenario": _build_controller(cfg)}

    time_cfg = cfg.get("time", {})
    steps = int(time_cfg.get("steps", 0))
    dt_min = time_cfg.get("dt_min")

    run_cfg = cfg.get("run", {})
    run_dir = ensure_run_dir(tag=run_cfg.get("tag"), base=run_cfg.get("base", "runs"))

    log_cfg = cfg.get("logging", {})
    log_base = Path(log_cfg.get("base_dir", "logs"))
    log_dir = log_base / run_dir.name
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = setup_structured_logging(
        log_dir,
        log_file=log_cfg.get("log_file", "runner.log"),
        level=level,
    )
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
        # Capture ADMM iteration history for this step (if available)
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
    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(run_dir / "logs.csv", index=False)
    if not df.empty:
        metrics_cols = [col for col in ["step", "residual_max", "residual_mean"] if col in df.columns]
        if metrics_cols:
            metrics_df = df[metrics_cols].copy()
            metrics_df.to_csv(
                log_dir / log_cfg.get("metrics_file", "metrics.csv"),
                index=False,
            )
            logger.info("Metrics written to %s", log_dir / log_cfg.get("metrics_file", "metrics.csv"))

    # Expand vectors to wide trace for convenience
    if not df.empty:
        try:
            import numpy as _np
            tso_mat = _np.vstack(df["tso_vector"].apply(_np.asarray).to_list())
            dso_mat = _np.vstack(df["dso_vector"].apply(_np.asarray).to_list())
            env_up = _np.vstack(df["envelope_upper"].apply(_np.asarray).to_list())
            env_lo = _np.vstack(df["envelope_lower"].apply(_np.asarray).to_list())
            cols_tso = {i: f"tso_{i}" for i in range(tso_mat.shape[1])}
            cols_dso = {i: f"dso_{i}" for i in range(dso_mat.shape[1])}
            cols_up = {i: f"env_up_{i}" for i in range(env_up.shape[1])}
            cols_lo = {i: f"env_lo_{i}" for i in range(env_lo.shape[1])}
            wide = pd.DataFrame(index=df.index)
            for i, name in cols_tso.items():
                wide[name] = tso_mat[:, i]
            for i, name in cols_dso.items():
                wide[name] = dso_mat[:, i]
            for i, name in cols_up.items():
                wide[name] = env_up[:, i]
            for i, name in cols_lo.items():
                wide[name] = env_lo[:, i]
            wide["residual_max"] = df["residual_max"].values
            wide["residual_mean"] = df["residual_mean"].values
            wide["step"] = df["step"].values
            wide.to_parquet(run_dir / "trace.parquet", index=False)
        except Exception:
            pass

    summary = {
        "final_residual": float(history[-1]["residual_max"]),
        "mean_residual": float(df["residual_mean"].mean()) if not df.empty else None,
        "steps": steps,
        "interfaces": int(len(history[-1]["tso_vector"])) if history else 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Summary | final_residual=%.3e | mean_residual=%s | interfaces=%d",
        summary["final_residual"],
        f"{summary['mean_residual']:.3e}" if summary["mean_residual"] is not None else "n/a",
        summary["interfaces"],
    )

    # Figures
    figs = run_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    try:
        # Plot interface 0 tso vs dso if present
        if not df.empty and len(df["tso_vector"].iloc[0]) > 0:
            s0_tso = [row[0] for row in df["tso_vector"]]
            s0_dso = [row[0] for row in df["dso_vector"]]
            plot_df = pd.DataFrame({"tso_0": s0_tso, "dso_0": s0_dso})
            from viz.plots import plot_timeseries

            plot_timeseries(plot_df, ["tso_0", "dso_0"], figs / "pcc_timeseries.png")
    except Exception:
        pass

    # Summary markdown
    try:
        md = ["# Run Summary", "", f"- Steps: {summary['steps']}", f"- Interfaces: {summary['interfaces']}", f"- Final residual: {summary['final_residual']:.3e}", f"- Mean residual: {summary['mean_residual']:.3e}" if summary["mean_residual"] is not None else "- Mean residual: n/a"]
        (run_dir / "summary.md").write_text("\n".join(md))
    except Exception:
        pass

    # Save ADMM per-iteration histories and combined artifacts
    import pandas as pd
    combined_rows: list[dict[str, float]] = []
    for t, hist in enumerate(admm_histories):
        if not hist:
            continue
        hdf = pd.DataFrame(hist)
        # Per-step CSV/PNG (preserve for detailed inspection)
        csv_path = run_dir / f"admm_history_step_{t}.csv"
        hdf.to_csv(csv_path, index=False)
        try:
            plot_convergence(hdf.set_index("iter"), run_dir / f"admm_conv_step_{t}.png")
        except Exception:
            pass
        # Collect for combined outputs
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
    return _simulate(cfg)
