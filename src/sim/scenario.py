"""Scenario construction utilities.

Defines the ScenarioState container and helpers to build controllers for
either toy consensus mode (used by tests) or integrated physics mode that
wraps Pyomo-based TSO/DSO solvers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from coord.admm import ADMMConfig
from coord.ti_env import compute_bounds_from_scenarios
from nmpc.b1 import B1Controller
from nmpc.base import BaseController
from nmpc.controller import NMPCConfig, NMPCController
from nmpc.greedy import GreedyConfig, GreedyController
from opt.pyomo_dso import (
    DSOParameters,
    apply_envelope_pg_bounds,
    build_dso_model,
    solve_dso_model,
)
from opt.pyomo_dso import (
    extract_solution as extract_dso_solution,
)
from opt.pyomo_tso import (
    TSOParameters,
    build_tso_model,
    solve_tso_model,
)
from opt.pyomo_tso import (
    extract_solution as extract_tso_solution,
)
from sim.forecast import sample_forecast
from utils.timewin import align_profiles, make_horizon

logger = logging.getLogger(__name__)


@dataclass
class ScenarioState:
    controller: BaseController
    history: list[dict[str, Any]]
    # Toy mode references
    tso_targets: list[np.ndarray] | None = None
    dso_targets: list[np.ndarray] | None = None
    rho: float | None = None
    step_ref: dict[str, int] | None = None
    # Integrated Pyomo mode references
    tso_base: dict[str, Any] | None = None
    dso_params: list[DSOParameters] | None = None
    dso_indices: np.ndarray | None = None
    dso_bus_indices: np.ndarray | None = None
    algorithm: str = "OUR"
    dso_env_specs: dict[int, dict[str, Any]] | None = None


def _setup_dso_params(
    dsos_cfg: list[dict[str, Any]], time_cfg: dict[str, Any], n_pred: int
) -> list[DSOParameters]:
    dso_params: list[DSOParameters] = []
    for d in dsos_cfg:
        sens = {
            "Rp": np.asarray(d["sens"]["Rp"], dtype=float),
            "Rq": np.asarray(d["sens"].get("Rq", np.zeros_like(d["sens"]["Rp"])), dtype=float),
            "vm_base": np.asarray(d["sens"]["vm_base"], dtype=float),
        }
        horizon_steps = int(n_pred)
        prof = d.get("profiles", {})
        if "csv" in prof:
            df_raw = pd.read_csv(prof["csv"], parse_dates=["time"])  # type: ignore[arg-type]
            df_raw = df_raw.set_index("time")
            idx = pd.date_range(
                start=time_cfg["start"], periods=horizon_steps, freq=f"{time_cfg['dt_min']}min"
            )
            aligned = align_profiles(idx, {"load": df_raw["load"], "pv": df_raw["pv"]})
            df = aligned.reset_index(drop=True)
        else:
            load_arr = np.asarray(prof.get("load", [0.0] * horizon_steps), dtype=float)
            pv_arr = np.asarray(prof.get("pv", [0.0] * horizon_steps), dtype=float)
            df = pd.DataFrame({"load": load_arr, "pv": pv_arr})
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
    return dso_params


def _compute_ti_envelopes(
    dso_params: list[DSOParameters],
    dso_indices: np.ndarray,
    dso_bus_indices: np.ndarray,
    ti_cfg: dict[str, Any],
    forecast_cfg: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    scen_count = int(ti_cfg.get("scenario_count", 8))
    alpha = float(ti_cfg.get("alpha", 1.0))
    margin = float(ti_cfg.get("margin", 0.0))
    sigL = float(forecast_cfg.get("sigma_load", 0.05))
    sigPV = float(forecast_cfg.get("sigma_pv", 0.15))
    rho_err = float(forecast_cfg.get("rho", 0.6))

    slot_groups: dict[int, dict[str, Any]] = {}
    size = len(dso_indices)
    for slot in range(size):
        di = int(dso_indices[slot])
        bi = int(dso_bus_indices[slot])
        group = slot_groups.setdefault(di, {"slots": [], "bus_indices": []})
        group["slots"].append(slot)
        group["bus_indices"].append(bi)

    dso_env_specs: dict[int, dict[str, Any]] = {}
    for di, group in slot_groups.items():
        p = dso_params[di]
        T = int(p.horizon.steps)
        load_series = p.profiles["load"].iloc[:T]
        pv_series = p.profiles["pv"].iloc[:T]
        # Robustness: pad with last value if series shorter than T (e.g., edge alignment)
        def _pad_series(s: pd.Series, length: int) -> pd.Series:
            s = s.dropna()
            if s.size >= length:
                return s.iloc[:length]
            if s.size == 0:
                return pd.Series([0.0] * length)
            last = float(s.iloc[-1])
            pad = pd.Series([last] * (length - s.size))
            return pd.concat([s, pad], ignore_index=True)

        load_series = _pad_series(load_series, T)
        pv_series = _pad_series(pv_series, T)
        L = sample_forecast(
            load_series, sigma=sigL, rho=rho_err, horizon=T, n=scen_count, mode="relative", clamp_nonneg=False
        )
        PV = sample_forecast(
            pv_series, sigma=sigPV, rho=rho_err, horizon=T, n=scen_count, mode="relative", clamp_nonneg=True
        )
        G = np.maximum(L - PV, 0.0)
        m = len(group["bus_indices"])
        G3 = np.repeat(G[:, :, None], m, axis=2)
        low, up = compute_bounds_from_scenarios(G3, alpha=alpha, margin=margin)
        dso_env_specs[di] = {"bus_indices": group["bus_indices"], "lower": low, "upper": up}
    return dso_env_specs


def build_scenario(cfg: dict[str, Any]) -> ScenarioState:
    """Build a ScenarioState with an appropriate controller for a given config."""
    time_cfg = cfg["time"]
    n_pred = int(cfg.get("N_pred", time_cfg["steps"]))
    _ = make_horizon(time_cfg["start"], n_pred, time_cfg["dt_min"])  # reserved

    admm_dict = cfg.get("admm", {})
    env_cfg = cfg.get("envelope", {})
    alg = str(cfg.get("algorithm", "OUR")).upper()

    if "tso" in cfg and "dsos" in cfg:
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
        dso_params = _setup_dso_params(dsos_cfg, time_cfg, n_pred)

        size = len(boundary)
        mapping = cfg.get("mapping", {})
        if not mapping:
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

        ti_cfg = cfg.get("ti_envelope", {}) | env_cfg
        ti_enabled = bool(ti_cfg.get("enabled", alg in {"OUR", "B3"}))
        dso_env_specs: dict[int, dict[str, Any]] = {}
        if ti_enabled:
            dso_env_specs = _compute_ti_envelopes(
                dso_params, dso_indices, dso_bus_indices, ti_cfg, cfg.get("forecast", {})
            )

        def tso_solver(v: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
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
                logger.exception("TSO solver failed; returning zero vector fallback")
                flows_vec = np.zeros(len(boundary), dtype=float)
                obj = None
                theta_meta = {"error": str(exc)}
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
            meta: dict[str, Any] = {"dso_objs": [], "dso_voltages": []}
            solved: list[tuple[DSOParameters, Any]] = []
            for di, p in enumerate(dso_params):
                model = build_dso_model(p)
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
                    meta.setdefault("errors", []).append({"dso_index": di, "error": str(exc)})
                    logger.exception("DSO solver failed for index %d; using zero fallback", di)
                    Tloc = int(p.horizon.steps)
                    buses = sorted(range(len(p.sens["vm_base"])))
                    zero = pd.DataFrame(np.zeros((Tloc, len(buses))), index=range(Tloc), columns=buses)

                    class _Res:
                        def __init__(self, pg: pd.DataFrame, error: str):
                            self.p_injections = pg
                            self.voltage = pg
                            self.objective = 0.0
                            self.error = error

                    res = _Res(zero, str(exc))
                solved.append((p, res))
                meta["dso_objs"].append(res.objective)
                meta["dso_voltages"].append(res.voltage.to_dict(orient="split"))

            T = int(dso_params[0].horizon.steps) if dso_params else 1
            mat = np.zeros((T, size), dtype=float)
            for slot in range(size):
                di = int(dso_indices[slot])
                bi = int(dso_bus_indices[slot])
                _, res = solved[di]
                series = res.p_injections.iloc[:, bi].to_numpy()
                if series.shape[0] < T:
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

        ctrl_name = alg
        if ctrl_name == "GREEDY":
            controller = GreedyController(GreedyConfig(size=size))
        elif ctrl_name == "B1":
            controller = B1Controller(
                NMPCConfig(
                    size=size,
                    admm=admm_cfg,
                    tso_solver=tso_solver,
                    dso_solver=dso_solver,
                    envelope_margin=float(env_cfg.get("margin", 0.05)),
                    envelope_alpha=float(env_cfg.get("alpha", 0.3)),
                )
            )
        else:
            controller = NMPCController(
                NMPCConfig(
                    size=size,
                    admm=admm_cfg,
                    tso_solver=tso_solver,
                    dso_solver=dso_solver,
                    envelope_margin=float(env_cfg.get("margin", 0.05)),
                    envelope_alpha=float(env_cfg.get("alpha", 0.3)),
                )
            )

        return ScenarioState(
            controller=controller,
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

    controller = NMPCController(
        NMPCConfig(
            size=size,
            admm=admm_cfg,
            tso_solver=tso_solver,
            dso_solver=dso_solver,
            envelope_margin=float(env_cfg.get("margin", 0.05)),
            envelope_alpha=float(env_cfg.get("alpha", 0.3)),
        )
    )
    return ScenarioState(
        controller=controller,
        history=[],
        tso_targets=tso_targets,
        dso_targets=dso_targets,
        rho=rho,
        step_ref=step_ref,
    )
