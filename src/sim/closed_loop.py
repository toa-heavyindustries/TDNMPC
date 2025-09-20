"""Closed-loop simulation helpers coupling simple TSO DC-OPF with DSO tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import pandapower as pp
import pyomo.environ as pyo

from dso.network import build_cigre_feeder
from interface import apply_reference, build_box_envelope, envelopes_to_bounds, measure_boundary
from interface.envelope import BoxEnvelope
from opt.pyomo_tso import TSOParameters, build_tso_model, extract_solution, solve_tso_model
from tso.network import build_tso_case, mark_boundary_buses


@dataclass(slots=True)
class ClosedLoopResult:
    """Aggregated outputs from a multi-step closed-loop simulation."""

    history: pd.DataFrame
    summary: dict[str, float]
    envelopes: list[BoxEnvelope]
    boundary_order: list[int]


def _ensure_solver_available(name: str) -> None:
    solver = pyo.SolverFactory(name)
    if solver is None or not solver.available():  # pragma: no cover - skip handled in tests
        raise RuntimeError(f"Solver '{name}' is not available")


def run_closed_loop(
    *,
    steps: int = 6,
    amplitude: float = 1.0,
    boundary_ids: Sequence[int] | None = None,
    feeder_type: str = "mv",
    feeder_peak_mw: float = 20.0,
    solver: str = "glpk",
    envelope_margin: float = 0.5,
    dt_min: float = 5.0,
    load_profile: Sequence[float] | None = None,
) -> ClosedLoopResult:
    """Execute a simple closed-loop tracking sequence using box envelopes."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    _ensure_solver_available(solver)

    if boundary_ids is None:
        boundary_ids = [1, 3, 5]
    boundary_ids = list(boundary_ids)

    n_bus = max(boundary_ids) + 3
    tso_case = build_tso_case(n_bus)
    tso_case = mark_boundary_buses(tso_case, boundary_ids)
    boundary_order = tso_case["boundary"].tolist()

    feeders = [build_cigre_feeder(feeder_type, target_peak_mw=feeder_peak_mw) for _ in boundary_order]
    envelopes: list[BoxEnvelope] = []
    for feeder, bus in zip(feeders, boundary_order):
        env = build_box_envelope(feeder, p_margin=None, q_margin=None)
        env.metadata["boundary_bus"] = bus
        envelopes.append(env)

    lower_bounds, upper_bounds = envelopes_to_bounds(envelopes, boundary_order)

    records: list[dict[str, float]] = []
    dt_hours = dt_min / 60.0
    soc: dict[int, float] = {int(bus): 0.0 for bus in boundary_order}

    for step in range(steps):
        phase = 2.0 * math.pi * (step / max(steps, 1))
        offset = amplitude * math.sin(phase)

        dso_targets: dict[int, float] = {}
        voltage_min = float("inf")
        voltage_max = float("-inf")

        for env, feeder, bus in zip(envelopes, feeders, boundary_order):
            base_p, base_q, _ = measure_boundary(feeder.net)
            if load_profile is not None and step < len(load_profile):
                target_raw = float(load_profile[step])
            else:
                target_raw = base_p + offset
            target_p = target_raw
            target_p = min(env.p_max - envelope_margin, max(env.p_min + envelope_margin, target_p))
            measurement = apply_reference(feeder.net, target_p, base_q)
            dso_targets[bus] = measurement.p_mw

            pp.runpp(feeder.net, algorithm="nr", numba=False)
            voltage_min = min(voltage_min, float(feeder.net.res_bus.vm_pu.min()))
            voltage_max = max(voltage_max, float(feeder.net.res_bus.vm_pu.max()))

        boundary_targets = np.asarray([dso_targets[int(bus)] for bus in boundary_order], dtype=float)

        params = TSOParameters(
            admittance=tso_case["admittance"],
            injections=tso_case["injections"],
            boundary=tso_case["boundary"],
            boundary_targets=boundary_targets,
            rho=1.0,
            cost_coeff=10.0,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        model = build_tso_model(params)
        solve_tso_model(model, solver=solver)
        result = extract_solution(model, params)

        flows = result.flows.loc[boundary_order]
        for idx, bus in enumerate(boundary_order):
            soc[bus] += -flows.iloc[idx] * dt_hours
            records.append(
                {
                    "step": step,
                    "bus": int(bus),
                    "p_target": boundary_targets[idx],
                    "p_tso": flows.iloc[idx],
                    "residual": flows.iloc[idx] - boundary_targets[idx],
                    "p_request": target_raw,
                    "v_min": voltage_min,
                    "v_max": voltage_max,
                    "soc": soc[bus],
                }
            )

    history = pd.DataFrame(records)
    summary = {
        "max_abs_residual": float(history["residual"].abs().max()) if not history.empty else 0.0,
        "mean_abs_residual": float(history["residual"].abs().mean()) if not history.empty else 0.0,
        "min_voltage": float(history["v_min"].min()) if not history.empty else float("nan"),
        "max_voltage": float(history["v_max"].max()) if not history.empty else float("nan"),
        "soc_min": float(history["soc"].min()) if not history.empty else float("nan"),
        "soc_max": float(history["soc"].max()) if not history.empty else float("nan"),
        "soc_range": float(history["soc"].max() - history["soc"].min()) if not history.empty else float("nan"),
    }

    return ClosedLoopResult(history=history, summary=summary, envelopes=envelopes, boundary_order=boundary_order)
